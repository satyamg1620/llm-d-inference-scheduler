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

package runner

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-inference-scheduler/internal/runnable"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/backend"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/tracing"
	backendmetrics "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/backend/metrics"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datastore"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts"
	fccontroller "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/controller"
	fcregistry "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metrics"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metrics/collectors"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/scheduling"
	runserver "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/server"
	"github.com/llm-d/llm-d-inference-scheduler/version"
)

// RunStandalone starts the EPP in bare-metal mode: no controller-runtime
// manager, no Kubernetes API client, no CRD reconcilers. Backend discovery is
// performed by reading a YAML file on disk.
//
// The lifecycle is:
//
//	load backends.yaml → build datastore → build scheduler →
//	  build admission controller → build director → start metrics HTTP,
//	  health gRPC, ext-proc gRPC, and file-provider polling loops.
//
// The caller is expected to have already parsed CLI flags (Run() does this
// before dispatching). The function blocks until ctx is cancelled, then waits
// for goroutines to drain before returning.
func (r *Runner) RunStandalone(ctx context.Context, opts *runserver.Options) error {
	setupLog.Info(r.eppExecutableName+" bare-metal build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

	if !opts.BareMetal {
		return errors.New("RunStandalone invoked without --baremetal")
	}

	if opts.Tracing {
		if err := tracing.InitTracing(ctx, setupLog, "llm-d-inference-scheduler/epp"); err != nil {
			return fmt.Errorf("failed to init tracing: %w", err)
		}
	}

	// --- Pre-read the backends file once so we know the pool identity. ---
	initial, err := backend.LoadBackendsFile(opts.BackendsFile)
	if err != nil {
		return fmt.Errorf("read backends file: %w", err)
	}
	if initial.PoolName == "" {
		return errors.New("backends file must declare a non-empty poolName")
	}
	if initial.PoolNamespace == "" {
		initial.PoolNamespace = runserver.DefaultPoolNamespace
	}
	if initial.TargetPort <= 0 {
		return errors.New("backends file must declare a positive targetPort")
	}

	// --- Parse plugin config / feature gates. ---
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	if err != nil {
		return fmt.Errorf("parse configuration: %w", err)
	}

	useNewMetrics := !r.featureGates[datalayer.EnableLegacyMetricsFeatureGate]
	pmc, err := backendmetrics.NewPodMetricsClientImpl(setupLog, backendmetrics.Config{
		ModelServerMetricsScheme:        opts.ModelServerMetricsScheme,
		ModelServerMetricsHTTPSInsecure: opts.ModelServerMetricsHTTPSInsecure,
		ModelServerMetricsPath:          opts.ModelServerMetricsPath,
		TotalQueuedRequestsMetric:       opts.TotalQueuedRequestsMetric,
		TotalRunningRequestsMetric:      opts.TotalRunningRequestsMetric,
		KVCacheUsagePercentageMetric:    opts.KVCacheUsagePercentageMetric,
		LoRAInfoMetric:                  opts.LoRAInfoMetric,
		CacheInfoMetric:                 opts.CacheInfoMetric,
	})
	if err != nil {
		return err
	}
	epf := r.setupMetricsCollection(useNewMetrics, opts, pmc)

	// --- Build EndpointPool from the backends file (no InferencePool CRD). ---
	endpointPool := datalayer.NewEndpointPool(initial.PoolNamespace, initial.PoolName)
	endpointPool.TargetPorts = []int{initial.TargetPort}

	// --- Build datastore. ---
	ds := datastore.NewDatastore(ctx, epf, int32(opts.ModelServerMetricsPort)).WithEndpointPool(endpointPool)

	// --- Phase-two config parsing (registers plugins, parser, scheduler). ---
	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	if err != nil {
		return fmt.Errorf("parse configuration (phase two): %w", err)
	}

	// --- Prometheus metrics registration. ---
	r.customCollectors = append(r.customCollectors, collectors.NewInferencePoolMetricsCollector(ds))
	metrics.Register(r.customCollectors...)
	metrics.RecordInferenceExtensionInfo(version.CommitSHA, version.BuildRef)

	if r.schedulerConfig == nil {
		return errors.New("scheduler config must be set either by config api or through code")
	}
	scheduler := scheduling.NewSchedulerWithConfig(r.schedulerConfig)

	// --- Data layer (no controller manager — Runtime.Start now tolerates nil). ---
	datalayerMetricsEnabled := !r.featureGates[datalayer.EnableLegacyMetricsFeatureGate]
	if err := r.configureAndStartDatalayer(ctx, datalayerMetricsEnabled, eppConfig.DataConfig, nil); err != nil {
		return fmt.Errorf("init data layer: %w", err)
	}

	// --- Admission control + director. ---
	var admissionController requestcontrol.AdmissionController
	var endpointCandidates contracts.EndpointCandidates
	endpointCandidates = requestcontrol.NewDatastoreEndpointCandidates(ds,
		requestcontrol.WithDisableEndpointSubsetFilter(opts.DisableEndpointSubsetFilter))
	if r.featureGates[flowcontrol.FeatureGate] {
		endpointCandidates = requestcontrol.NewCachedEndpointCandidates(ctx, endpointCandidates, 50*time.Millisecond)
		registry, err := fcregistry.NewFlowRegistry(eppConfig.FlowControlConfig.Registry, setupLog)
		if err != nil {
			return fmt.Errorf("init flow registry: %w", err)
		}
		fc, err := fccontroller.NewFlowController(
			ctx,
			initial.PoolName,
			eppConfig.FlowControlConfig.Controller,
			fccontroller.Deps{
				Registry:           registry,
				SaturationDetector: eppConfig.SaturationDetector,
				EndpointCandidates: endpointCandidates,
				UsageLimitPolicy:   eppConfig.FlowControlConfig.UsageLimitPolicy,
			},
		)
		if err != nil {
			return fmt.Errorf("init flow controller: %w", err)
		}
		go registry.Run(ctx)
		admissionController = requestcontrol.NewFlowControlAdmissionController(fc, initial.PoolName)
	} else {
		admissionController = requestcontrol.NewLegacyAdmissionController(eppConfig.SaturationDetector, endpointCandidates)
	}
	director := requestcontrol.NewDirectorWithConfig(ds, scheduler, admissionController, endpointCandidates, r.requestControlConfig)

	// --- Build (but do not register with a manager) the ExtProc server runner. ---
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         opts.GRPCPort,
		Datastore:                        ds,
		ControllerCfg:                    runserver.NewControllerConfig(false),
		SecureServing:                    opts.SecureServing,
		HealthChecking:                   opts.HealthChecking,
		CertPath:                         opts.CertPath,
		EnableCertReload:                 opts.EnableCertReload,
		RefreshPrometheusMetricsInterval: opts.RefreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:        opts.MetricsStalenessThreshold,
		Director:                         director,
		Parser:                           r.parser,
		SaturationDetector:               eppConfig.SaturationDetector,
		UseExperimentalDatalayerV2: r.featureGates[datalayer.ExperimentalDatalayerFeatureGate] ||
			!r.featureGates[datalayer.EnableLegacyMetricsFeatureGate],
	}

	// --- Construct the FileProvider that drives backend discovery. ---
	fp, err := backend.New(ds, backend.Options{
		Path:     opts.BackendsFile,
		Interval: opts.BackendsPollInterval,
		Logger:   ctrl.Log.WithName("backends"),
		HealthCheck: backend.HealthCheckConfig{
			Interval:         opts.HealthCheckInterval,
			Path:             opts.HealthCheckPath,
			Timeout:          opts.HealthCheckTimeout,
			FailureThreshold: opts.HealthCheckFailureThreshold,
		},
	})
	if err != nil {
		return fmt.Errorf("init file provider: %w", err)
	}
	if err := fp.ReloadOnce(ctx); err != nil {
		return fmt.Errorf("initial backends reconcile: %w", err)
	}

	// --- Bare-metal is always "leader" for readiness. ---
	isLeader := &atomic.Bool{}
	isLeader.Store(true)

	// --- Spin up the long-running goroutines. ---
	g, gctx := errgroup.WithContext(ctx)

	// FileProvider polling loop.
	g.Go(func() error { return fp.Start(gctx) })

	// Prometheus /metrics HTTP server.
	g.Go(func() error { return runMetricsServer(gctx, opts.MetricsPort) })

	// Ext-proc gRPC server.
	g.Go(func() error {
		return serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc")).Start(gctx)
	})

	// Health gRPC server.
	g.Go(func() error {
		return runStandaloneHealthServer(gctx, ds, opts.GRPCHealthPort, isLeader, opts.EnableLeaderElection, r.parser)
	})

	setupLog.Info("Bare-metal EPP started",
		"grpcPort", opts.GRPCPort,
		"metricsPort", opts.MetricsPort,
		"healthPort", opts.GRPCHealthPort,
		"backendsFile", opts.BackendsFile,
		"pool", initial.PoolName,
		"namespace", initial.PoolNamespace,
		"targetPort", initial.TargetPort,
	)
	if err := g.Wait(); err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
	setupLog.Info("Bare-metal EPP terminated")
	return nil
}

// runMetricsServer exposes the Prometheus registry over plain HTTP. In
// Kubernetes mode this is wired into the controller-runtime metrics server.
//
// Note: the EPP registers metrics against the controller-runtime registry
// (sigs.k8s.io/controller-runtime/pkg/metrics.Registry), NOT the global
// prometheus default registry. Use HandlerFor with that registry so all
// inference_extension_*, inference_pool_*, etc. metrics show up.
func runMetricsServer(ctx context.Context, port int) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{}))
	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("metrics server: %w", err)
	}
	return nil
}

// runStandaloneHealthServer starts the gRPC health server outside of a
// controller-runtime manager. It mirrors registerHealthServer but uses the
// errgroup lifecycle instead of mgr.Add.
func runStandaloneHealthServer(
	ctx context.Context,
	ds datastore.Datastore,
	port int,
	isLeader *atomic.Bool,
	leaderElectionEnabled bool,
	supporter appProtocolSupporter,
) error {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger:                ctrl.Log.WithName("health"),
		datastore:             ds,
		isLeader:              isLeader,
		leaderElectionEnabled: leaderElectionEnabled,
		supporter:             supporter,
	})
	return runnable.GRPCServer("health", srv, port).Start(ctx)
}
