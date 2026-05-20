/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package backend contains bare-metal backend discovery for the EPP.
//
// FileProvider replaces the Kubernetes EndpointSlice / Pod / InferencePool
// reconcilers when the EPP runs without a Kubernetes API server. It reads a
// YAML file declaring vLLM backends and feeds the datastore through the
// EndpointUpsert / EndpointDelete entry points that the datastore exposes
// specifically for non-Kubernetes discovery sources.
//
// When HealthCheckConfig.Interval > 0, FileProvider also actively probes
// each declared backend's HTTP /health endpoint on a separate goroutine.
// Backends that fail N consecutive probes are removed from rotation, and
// re-admitted after a single successful probe.
package backend

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
)

// BackendsConfig is the schema of the YAML file consumed by FileProvider.
//
// Example:
//
//	poolName: llama3-pool
//	poolNamespace: default
//	targetPort: 8000
//	backends:
//	  - address: "192.168.1.10:8000"
//	    labels:
//	      llm-d.ai/role: decode
type BackendsConfig struct {
	PoolName      string         `json:"poolName"      yaml:"poolName"`
	PoolNamespace string         `json:"poolNamespace" yaml:"poolNamespace"`
	TargetPort    int            `json:"targetPort"    yaml:"targetPort"`
	Backends      []BackendEntry `json:"backends"      yaml:"backends"`
}

// BackendEntry describes one vLLM endpoint declared in the YAML file.
type BackendEntry struct {
	// Address is either "host:port" or just "host". When the port is omitted,
	// the top-level TargetPort is used.
	Address string `json:"address" yaml:"address"`
	// Labels are propagated onto the synthetic endpoint metadata so filters
	// and scorers can match on the same llm-d.ai/role, hardware-type, etc.
	// labels they use in Kubernetes mode.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	// MetricsPort optionally overrides the port the EPP scrapes /metrics from.
	// Defaults to the same port as Address.
	MetricsPort int `json:"metricsPort,omitempty" yaml:"metricsPort,omitempty"`
}

// HealthCheckConfig configures active health probing of declared backends.
// When Interval is zero, the entire health-check goroutine is skipped and
// backends are admitted purely based on what's in the YAML file.
type HealthCheckConfig struct {
	// Interval between probe sweeps. Zero disables health checking entirely.
	Interval time.Duration
	// Path probed on each backend, e.g. "/health" or "/v1/models".
	Path string
	// Timeout for each individual probe.
	Timeout time.Duration
	// FailureThreshold is the number of consecutive failed probes after which
	// an admitted backend is removed via EndpointDelete. One successful probe
	// re-admits it.
	FailureThreshold int
	// HTTPClient is used to perform probe requests. Test code injects a
	// custom client; nil means use a default client with the configured
	// timeout.
	HTTPClient *http.Client
}

// backendState carries per-backend bookkeeping for the health-check loop.
type backendState struct {
	metadata            *fwkdl.EndpointMetadata // latest from YAML; used to re-upsert on recovery
	consecutiveFailures int
	admitted            bool // true if the datastore currently holds this endpoint
}

// FileProvider polls a YAML file and reconciles its contents into a Datastore.
//
// It does not implement the Datastore interface itself; it wraps an existing
// Datastore and pushes EndpointMetadata through the non-Kubernetes
// EndpointUpsert / EndpointDelete entry points. The scheduling / scorer /
// filter code path is identical to Kubernetes mode.
type FileProvider struct {
	path     string
	interval time.Duration
	ds       datastore.Datastore
	logger   logr.Logger

	health HealthCheckConfig

	mu      sync.Mutex
	known   map[string]*backendState // backendKey -> state
	current BackendsConfig
}

// Options configures a FileProvider.
type Options struct {
	Path        string
	Interval    time.Duration
	Logger      logr.Logger
	HealthCheck HealthCheckConfig
}

// New returns a FileProvider. It does not perform an initial read; call
// ReloadOnce before Start if you need the first reconcile to be synchronous
// with startup.
func New(ds datastore.Datastore, opts Options) (*FileProvider, error) {
	if ds == nil {
		return nil, errors.New("datastore must not be nil")
	}
	if opts.Path == "" {
		return nil, errors.New("backends file path must not be empty")
	}
	interval := opts.Interval
	if interval <= 0 {
		interval = 10 * time.Second
	}

	hc := opts.HealthCheck
	if hc.Interval > 0 {
		// Health checking is enabled; fill in sensible defaults for fields
		// the caller left zero.
		if hc.Path == "" {
			hc.Path = "/health"
		}
		if hc.Timeout <= 0 {
			hc.Timeout = 2 * time.Second
		}
		if hc.FailureThreshold <= 0 {
			hc.FailureThreshold = 3
		}
		if hc.HTTPClient == nil {
			hc.HTTPClient = &http.Client{Timeout: hc.Timeout}
		}
	}

	return &FileProvider{
		path:     opts.Path,
		interval: interval,
		ds:       ds,
		logger:   opts.Logger,
		health:   hc,
		known:    map[string]*backendState{},
	}, nil
}

// ReloadOnce reads the file and applies a single reconcile pass. It is
// intended to be called once at startup so the datastore is populated before
// the gRPC server starts serving requests.
func (p *FileProvider) ReloadOnce(ctx context.Context) error {
	cfg, err := LoadBackendsFile(p.path)
	if err != nil {
		return err
	}
	return p.apply(ctx, cfg)
}

// LoadBackendsFile reads a BackendsConfig from disk without constructing a
// FileProvider. It is exported so the runner can read pool identity (poolName,
// namespace, targetPort) before the datastore is built.
func LoadBackendsFile(path string) (BackendsConfig, error) {
	return loadBackendsFile(path)
}

// Start blocks, periodically re-reading the backends file and (if enabled)
// running active health probes. It returns when ctx is cancelled.
func (p *FileProvider) Start(ctx context.Context) error {
	logger := p.logger.WithName("file-provider")
	logger.Info("starting backend file watcher",
		"path", p.path,
		"interval", p.interval,
		"healthCheckEnabled", p.health.Interval > 0,
		"healthCheckInterval", p.health.Interval,
	)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		p.runFileWatcher(ctx, logger)
	}()
	if p.health.Interval > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.runHealthChecks(ctx, logger.WithName("health-check"))
		}()
	}
	wg.Wait()
	return nil
}

func (p *FileProvider) runFileWatcher(ctx context.Context, logger logr.Logger) {
	t := time.NewTicker(p.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			cfg, err := LoadBackendsFile(p.path)
			if err != nil {
				logger.Error(err, "failed to read backends file; keeping previous state")
				continue
			}
			if err := p.apply(ctx, cfg); err != nil {
				logger.Error(err, "failed to apply backends file")
			}
		}
	}
}

// Current returns a snapshot of the most recently applied configuration.
func (p *FileProvider) Current() BackendsConfig {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.current
}

// apply reconciles a freshly read BackendsConfig into the datastore. Compared
// with the original behavior, this version:
//
//   - Calls EndpointUpsert only for NEW backends (admitted=false→true). For
//     backends already in `known`, just refreshes the cached metadata so the
//     next health-check recovery or label update sees up-to-date data; the
//     health checker decides re-admission.
//   - Skips EndpointDelete for backends that the health checker already
//     removed (admitted=false) — the datastore already lacks them.
//   - Calls EndpointDelete for backends dropped from the YAML, regardless of
//     their admitted state (idempotent at the datastore level).
func (p *FileProvider) apply(ctx context.Context, cfg BackendsConfig) error {
	if cfg.PoolName == "" {
		return errors.New("poolName is required in backends file")
	}
	if cfg.PoolNamespace == "" {
		cfg.PoolNamespace = "default"
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	desired := map[string]*fwkdl.EndpointMetadata{}
	for i, b := range cfg.Backends {
		host, port, err := splitAddress(b.Address, cfg.TargetPort)
		if err != nil {
			p.logger.Error(err, "skipping invalid backend", "index", i, "address", b.Address)
			continue
		}
		key := host + ":" + strconv.Itoa(port)
		metricsPort := b.MetricsPort
		if metricsPort == 0 {
			metricsPort = port
		}
		labels := make(map[string]string, len(b.Labels))
		for k, v := range b.Labels {
			labels[k] = v
		}
		desired[key] = &fwkdl.EndpointMetadata{
			NamespacedName: endpointID(cfg.PoolNamespace, key),
			PodName:        endpointID(cfg.PoolNamespace, key).Name,
			Address:        host,
			Port:           strconv.Itoa(port),
			MetricsHost:    net.JoinHostPort(host, strconv.Itoa(metricsPort)),
			Labels:         labels,
		}
	}

	// Adds and label refreshes.
	for key, meta := range desired {
		state, exists := p.known[key]
		if !exists {
			// New backend. Admit eagerly; the health checker will remove it
			// later if it turns out to be unhealthy.
			p.ds.EndpointUpsert(ctx, meta)
			p.known[key] = &backendState{metadata: meta, admitted: true}
			continue
		}
		// Existing backend — keep the cached metadata up-to-date. Push to
		// datastore only if currently admitted (otherwise the datastore
		// doesn't have it and the health checker owns re-admission).
		state.metadata = meta
		if state.admitted {
			p.ds.EndpointUpsert(ctx, meta)
		}
	}

	// Removals: anything we had last cycle but is no longer desired.
	for key, state := range p.known {
		if _, still := desired[key]; still {
			continue
		}
		if state.admitted {
			p.ds.EndpointDelete(state.metadata.NamespacedName)
		}
		delete(p.known, key)
	}

	p.current = cfg
	return nil
}

// ---------------------------------------------------------------------------
// Active health-check loop
// ---------------------------------------------------------------------------

func (p *FileProvider) runHealthChecks(ctx context.Context, logger logr.Logger) {
	logger.Info("starting health-check loop",
		"interval", p.health.Interval,
		"path", p.health.Path,
		"timeout", p.health.Timeout,
		"failureThreshold", p.health.FailureThreshold,
	)
	t := time.NewTicker(p.health.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			p.probeAll(ctx)
		}
	}
}

// probeAll probes every currently-known backend in parallel and applies the
// state-machine update for each. Probes hit the network without holding the
// FileProvider lock; only the apply step takes the lock briefly.
func (p *FileProvider) probeAll(ctx context.Context) {
	// Snapshot under lock so we don't probe while apply() is rewriting the
	// map. We capture (key, address, port) — enough for the probe URL.
	type target struct {
		key     string
		address string
		port    string
	}
	p.mu.Lock()
	targets := make([]target, 0, len(p.known))
	for key, state := range p.known {
		targets = append(targets, target{
			key:     key,
			address: state.metadata.Address,
			port:    state.metadata.Port,
		})
	}
	p.mu.Unlock()

	var wg sync.WaitGroup
	for _, tgt := range targets {
		wg.Add(1)
		go func(tgt target) {
			defer wg.Done()
			ok := p.probeOne(ctx, tgt.address, tgt.port)
			p.applyHealthResult(ctx, tgt.key, ok)
		}(tgt)
	}
	wg.Wait()
}

func (p *FileProvider) probeOne(ctx context.Context, address, port string) bool {
	url := fmt.Sprintf("http://%s%s", net.JoinHostPort(address, port), p.health.Path)
	reqCtx, cancel := context.WithTimeout(ctx, p.health.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := p.health.HTTPClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// applyHealthResult mutates the per-backend state in response to a probe
// outcome. It re-admits via EndpointUpsert on recovery and removes via
// EndpointDelete on the FailureThreshold-th consecutive failure.
func (p *FileProvider) applyHealthResult(ctx context.Context, key string, ok bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	state, exists := p.known[key]
	if !exists {
		// Backend was removed from YAML while the probe was in flight.
		return
	}
	logger := p.logger.WithName("health-check").WithValues("backend", key)

	if ok {
		if !state.admitted {
			p.ds.EndpointUpsert(ctx, state.metadata)
			state.admitted = true
			logger.Info("backend recovered; re-admitted",
				"priorConsecutiveFailures", state.consecutiveFailures)
		}
		state.consecutiveFailures = 0
		return
	}

	state.consecutiveFailures++
	if state.admitted && state.consecutiveFailures >= p.health.FailureThreshold {
		p.ds.EndpointDelete(state.metadata.NamespacedName)
		state.admitted = false
		logger.Info("backend failed health checks; removed from rotation",
			"consecutiveFailures", state.consecutiveFailures,
			"failureThreshold", p.health.FailureThreshold)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func loadBackendsFile(path string) (BackendsConfig, error) {
	var cfg BackendsConfig
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read backends file %q: %w", path, err)
	}
	// sigs.k8s.io/yaml handles both YAML and JSON and respects json: tags.
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return cfg, fmt.Errorf("parse backends file %q: %w", path, err)
	}
	return cfg, nil
}

// splitAddress accepts "host:port" or "host" and returns (host, port).
// If the port is missing on the entry, defaultPort is used.
func splitAddress(addr string, defaultPort int) (string, int, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "", 0, errors.New("address is empty")
	}
	if host, portStr, err := net.SplitHostPort(addr); err == nil {
		port, perr := strconv.Atoi(portStr)
		if perr != nil {
			return "", 0, fmt.Errorf("invalid port in %q: %w", addr, perr)
		}
		return host, port, nil
	}
	if defaultPort == 0 {
		return "", 0, fmt.Errorf("address %q has no port and targetPort is unset", addr)
	}
	return addr, defaultPort, nil
}

// endpointID turns "10.0.0.1:8000" + namespace into a stable NamespacedName.
// The transform is irrelevant to the scheduler; it only needs to be stable so
// removals on the next reconcile target the same entry that was added.
func endpointID(namespace, key string) types.NamespacedName {
	r := strings.NewReplacer(".", "-", ":", "-")
	return types.NamespacedName{Name: "vllm-" + r.Replace(key), Namespace: namespace}
}

// suppress "imported and not used" if types is only referenced via endpointID
var _ = types.NamespacedName{}
