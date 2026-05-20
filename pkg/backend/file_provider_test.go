/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-inference-scheduler/apix/v1alpha2"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
)

// fakeDatastore captures the calls FileProvider makes against the
// EndpointUpsert / EndpointDelete seam.
type fakeDatastore struct {
	mu        sync.Mutex
	endpoints map[types.NamespacedName]*fwkdl.EndpointMetadata
	deleted   []types.NamespacedName
}

func newFakeDatastore() *fakeDatastore {
	return &fakeDatastore{endpoints: map[types.NamespacedName]*fwkdl.EndpointMetadata{}}
}

func (f *fakeDatastore) EndpointUpsert(_ context.Context, meta *fwkdl.EndpointMetadata) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.endpoints[meta.NamespacedName] = meta.Clone()
}

func (f *fakeDatastore) EndpointDelete(id types.NamespacedName) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.endpoints, id)
	f.deleted = append(f.deleted, id)
}

func (f *fakeDatastore) snapshot() map[types.NamespacedName]*fwkdl.EndpointMetadata {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[types.NamespacedName]*fwkdl.EndpointMetadata, len(f.endpoints))
	for k, v := range f.endpoints {
		out[k] = v
	}
	return out
}

// fakeAdapter wraps fakeDatastore with stub implementations of the rest of
// the datastore.Datastore interface. FileProvider only calls EndpointUpsert
// and EndpointDelete, so every other method panics if invoked.
type fakeAdapter struct {
	*fakeDatastore
	unimplementedDatastore
}

// Compile-time check that fakeAdapter satisfies datastore.Datastore.
var _ datastore.Datastore = fakeAdapter{}

func newProvider(t *testing.T, ds *fakeDatastore, path string) *FileProvider {
	t.Helper()
	p, err := New(fakeAdapter{fakeDatastore: ds}, Options{
		Path:   path,
		Logger: logr.Discard(),
	})
	require.NoError(t, err)
	return p
}

func writeFile(t *testing.T, dir, contents string) string {
	t.Helper()
	path := filepath.Join(dir, "backends.yaml")
	require.NoError(t, os.WriteFile(path, []byte(contents), 0o600))
	return path
}

func TestFileProvider_InitialLoad(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolName: test-pool
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
    labels:
      llm-d.ai/role: decode
  - address: "10.0.0.2:8000"
    labels:
      llm-d.ai/role: decode
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.NoError(t, p.ReloadOnce(context.Background()))

	got := ds.snapshot()
	require.Len(t, got, 2)

	id := types.NamespacedName{Name: "vllm-10-0-0-1-8000", Namespace: "default"}
	require.Contains(t, got, id)
	require.Equal(t, "10.0.0.1", got[id].Address)
	require.Equal(t, "8000", got[id].Port)
	require.Equal(t, "decode", got[id].Labels["llm-d.ai/role"])
	require.Equal(t, "10.0.0.1:8000", got[id].MetricsHost)
}

func TestFileProvider_AddRemove(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolName: p
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.NoError(t, p.ReloadOnce(context.Background()))
	require.Len(t, ds.snapshot(), 1)

	require.NoError(t, os.WriteFile(path, []byte(`
poolName: p
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.2:8000"
`), 0o600))
	require.NoError(t, p.ReloadOnce(context.Background()))

	got := ds.snapshot()
	require.Len(t, got, 1)
	require.Contains(t, got, types.NamespacedName{Name: "vllm-10-0-0-2-8000", Namespace: "default"})
	require.Equal(t, []types.NamespacedName{
		{Name: "vllm-10-0-0-1-8000", Namespace: "default"},
	}, ds.deleted)
}

func TestFileProvider_RelabelInPlace(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolName: p
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
    labels:
      llm-d.ai/role: decode
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.NoError(t, p.ReloadOnce(context.Background()))

	require.NoError(t, os.WriteFile(path, []byte(`
poolName: p
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
    labels:
      llm-d.ai/role: prefill
`), 0o600))
	require.NoError(t, p.ReloadOnce(context.Background()))

	got := ds.snapshot()
	require.Len(t, got, 1)
	id := types.NamespacedName{Name: "vllm-10-0-0-1-8000", Namespace: "default"}
	require.Equal(t, "prefill", got[id].Labels["llm-d.ai/role"])
	require.Empty(t, ds.deleted, "relabel should not trigger a delete")
}

func TestFileProvider_DefaultTargetPort(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolName: p
poolNamespace: default
targetPort: 9999
backends:
  - address: "10.0.0.1"
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.NoError(t, p.ReloadOnce(context.Background()))

	got := ds.snapshot()
	id := types.NamespacedName{Name: "vllm-10-0-0-1-9999", Namespace: "default"}
	require.Contains(t, got, id)
	require.Equal(t, "9999", got[id].Port)
}

func TestFileProvider_RejectsMissingPoolName(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.Error(t, p.ReloadOnce(context.Background()))
}

func TestFileProvider_CustomMetricsPort(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, `
poolName: p
poolNamespace: default
targetPort: 8000
backends:
  - address: "10.0.0.1:8000"
    metricsPort: 9000
`)
	ds := newFakeDatastore()
	p := newProvider(t, ds, path)
	require.NoError(t, p.ReloadOnce(context.Background()))

	got := ds.snapshot()
	id := types.NamespacedName{Name: "vllm-10-0-0-1-8000", Namespace: "default"}
	require.Equal(t, "10.0.0.1:9000", got[id].MetricsHost)
}

func TestSplitAddress(t *testing.T) {
	cases := []struct {
		in       string
		defPort  int
		wantHost string
		wantPort int
		wantErr  bool
	}{
		{"10.0.0.1:8000", 0, "10.0.0.1", 8000, false},
		{"vllm-0:8001", 0, "vllm-0", 8001, false},
		{"10.0.0.1", 8000, "10.0.0.1", 8000, false},
		{"10.0.0.1", 0, "", 0, true},
		{"", 8000, "", 0, true},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			host, port, err := splitAddress(c.in, c.defPort)
			if c.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, c.wantHost, host)
			require.Equal(t, c.wantPort, port)
		})
	}
}

// ---------------------------------------------------------------------------
// Health-check tests
//
// These tests call probeAll() directly so the assertions are deterministic
// — we don't have to race against the goroutine's ticker.
// ---------------------------------------------------------------------------

func newProviderWithHealth(t *testing.T, ds *fakeDatastore, path string, hc HealthCheckConfig) *FileProvider {
	t.Helper()
	p, err := New(fakeAdapter{fakeDatastore: ds}, Options{
		Path:        path,
		Logger:      logr.Discard(),
		HealthCheck: hc,
	})
	require.NoError(t, err)
	return p
}

// hostOnly extracts "127.0.0.1:54321" from "http://127.0.0.1:54321".
func hostOnly(t *testing.T, u string) string {
	t.Helper()
	parsed, err := url.Parse(u)
	require.NoError(t, err)
	return parsed.Host
}

func TestFileProvider_HealthFailRemovesAfterThreshold(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer bad.Close()

	dir := t.TempDir()
	path := writeFile(t, dir, "poolName: p\npoolNamespace: default\ntargetPort: 1\nbackends:\n  - address: \""+hostOnly(t, bad.URL)+"\"\n")

	ds := newFakeDatastore()
	p := newProviderWithHealth(t, ds, path, HealthCheckConfig{
		Interval:         5 * time.Millisecond,
		Path:             "/",
		Timeout:          500 * time.Millisecond,
		FailureThreshold: 3,
		HTTPClient:       bad.Client(),
	})
	require.NoError(t, p.ReloadOnce(context.Background()))
	require.Len(t, ds.snapshot(), 1, "backend should be admitted on initial load")

	// First two failures stay admitted.
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 1, "still admitted after 1 failure")
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 1, "still admitted after 2 failures")

	// Third failure crosses the threshold and removes.
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 0, "should be removed after 3 failures")
	require.Len(t, ds.deleted, 1, "exactly one delete should have been issued")
}

func TestFileProvider_HealthRecoverReadmits(t *testing.T) {
	var serverHealthy atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if serverHealthy.Load() {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	dir := t.TempDir()
	path := writeFile(t, dir, "poolName: p\npoolNamespace: default\ntargetPort: 1\nbackends:\n  - address: \""+hostOnly(t, srv.URL)+"\"\n")

	ds := newFakeDatastore()
	p := newProviderWithHealth(t, ds, path, HealthCheckConfig{
		Interval:         5 * time.Millisecond,
		Path:             "/",
		Timeout:          500 * time.Millisecond,
		FailureThreshold: 2,
		HTTPClient:       srv.Client(),
	})
	require.NoError(t, p.ReloadOnce(context.Background()))

	// Drive past threshold.
	p.probeAll(context.Background())
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 0, "should be removed after threshold failures")
	require.Len(t, ds.deleted, 1)

	// Server recovers.
	serverHealthy.Store(true)
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 1, "should be re-admitted on first successful probe")
}

func TestFileProvider_YAMLRemovesUnhealthyBackend(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer bad.Close()

	dir := t.TempDir()
	path := writeFile(t, dir, "poolName: p\npoolNamespace: default\ntargetPort: 1\nbackends:\n  - address: \""+hostOnly(t, bad.URL)+"\"\n")

	ds := newFakeDatastore()
	p := newProviderWithHealth(t, ds, path, HealthCheckConfig{
		Interval:         5 * time.Millisecond,
		Path:             "/",
		Timeout:          500 * time.Millisecond,
		FailureThreshold: 2,
		HTTPClient:       bad.Client(),
	})
	require.NoError(t, p.ReloadOnce(context.Background()))

	// Push the backend to unhealthy.
	p.probeAll(context.Background())
	p.probeAll(context.Background())
	require.Len(t, ds.snapshot(), 0)
	require.Len(t, ds.deleted, 1)

	// YAML now removes the backend entirely. apply() must NOT issue a second
	// EndpointDelete — the health checker already did it.
	require.NoError(t, os.WriteFile(path, []byte("poolName: p\npoolNamespace: default\ntargetPort: 1\nbackends: []\n"), 0o600))
	require.NoError(t, p.ReloadOnce(context.Background()))
	require.Len(t, ds.deleted, 1, "no second delete on already-removed backend")
}

func TestFileProvider_HealthDisabled_NoProbingFields(t *testing.T) {
	// When the health check is disabled, the HTTPClient should remain nil
	// (we don't construct one) and probeAll is safe to call (it just walks
	// an empty target set).
	dir := t.TempDir()
	path := writeFile(t, dir, "poolName: p\npoolNamespace: default\ntargetPort: 8000\nbackends:\n  - address: \"10.0.0.1:8000\"\n")
	ds := newFakeDatastore()
	p := newProviderWithHealth(t, ds, path, HealthCheckConfig{Interval: 0})
	require.Nil(t, p.health.HTTPClient, "HTTPClient should not be initialized when health-check is disabled")
	require.NoError(t, p.ReloadOnce(context.Background()))
	require.Len(t, ds.snapshot(), 1, "backend admitted by file load")
}

// ---- stub implementations of the parts of datastore.Datastore the
// FileProvider does not touch. They panic to surface accidental coupling. ----

type unimplementedDatastore struct{}

func (unimplementedDatastore) PoolSet(context.Context, client.Reader, *datalayer.EndpointPool) error {
	panic("unimplemented")
}
func (unimplementedDatastore) PoolGet() (*datalayer.EndpointPool, error) { panic("unimplemented") }
func (unimplementedDatastore) PoolHasSynced() bool                       { panic("unimplemented") }
func (unimplementedDatastore) PoolLabelsMatch(map[string]string) bool    { panic("unimplemented") }
func (unimplementedDatastore) WithEndpointPool(*datalayer.EndpointPool) datastore.Datastore {
	panic("unimplemented")
}
func (unimplementedDatastore) ObjectiveSet(*v1alpha2.InferenceObjective)        { panic("unimplemented") }
func (unimplementedDatastore) ObjectiveGet(string) *v1alpha2.InferenceObjective { panic("unimplemented") }
func (unimplementedDatastore) ObjectiveDelete(types.NamespacedName)             { panic("unimplemented") }
func (unimplementedDatastore) ObjectiveGetAll() []*v1alpha2.InferenceObjective  { panic("unimplemented") }
func (unimplementedDatastore) ModelRewriteSet(*v1alpha2.InferenceModelRewrite)  { panic("unimplemented") }
func (unimplementedDatastore) ModelRewriteDelete(types.NamespacedName)          { panic("unimplemented") }
func (unimplementedDatastore) ModelRewriteGet(string) (*v1alpha2.InferenceModelRewriteRule, string) {
	panic("unimplemented")
}
func (unimplementedDatastore) ModelRewriteGetAll() []*v1alpha2.InferenceModelRewrite {
	panic("unimplemented")
}
func (unimplementedDatastore) PodList(func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	panic("unimplemented")
}
func (unimplementedDatastore) PodUpdateOrAddIfNotExist(context.Context, *corev1.Pod) bool {
	panic("unimplemented")
}
func (unimplementedDatastore) PodDelete(string) { panic("unimplemented") }
func (unimplementedDatastore) Clear()           { panic("unimplemented") }
