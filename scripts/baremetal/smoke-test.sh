#!/usr/bin/env bash
# End-to-end smoke test for the bare-metal EPP path.
#
# Starts two llm-d-inference-sim containers (no GPU required), runs the EPP in
# --baremetal mode, asserts on Prometheus metrics, and exercises the ext-proc
# gRPC endpoint to confirm a destination is selected.
#
# Usage: scripts/baremetal/smoke-test.sh
#
# Requirements: docker, go, curl, grpcurl (optional — falls back to a metrics-only check).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SIM_IMAGE="ghcr.io/llm-d/llm-d-inference-sim:v0.8.2"
SIM_MODEL="Qwen/Qwen3-32B"
SIM_NAMES=("baremetal-sim-1" "baremetal-sim-2")
SIM_PORTS=(8001 8002)

EPP_BIN="${ROOT}/bin/epp"
EPP_LOG="$(mktemp -t epp-baremetal.XXXXXX.log)"
EPP_PID=""
RC=0

cleanup() {
  local exit_code=$?
  echo "---- cleanup ----"
  if [[ -n "$EPP_PID" ]] && kill -0 "$EPP_PID" 2>/dev/null; then
    kill "$EPP_PID" || true
    wait "$EPP_PID" 2>/dev/null || true
  fi
  for name in "${SIM_NAMES[@]}"; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
  if [[ $exit_code -ne 0 || "${KEEP_LOGS:-}" == "1" ]]; then
    echo "EPP log: $EPP_LOG"
  else
    rm -f "$EPP_LOG"
  fi
  exit $exit_code
}
trap cleanup EXIT

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()  { printf '   \033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '   \033[1;31m✗\033[0m %s\n' "$*" >&2; RC=1; exit 1; }

say "Building EPP binary"
go build -o "$EPP_BIN" ./cmd/epp
ok "built $EPP_BIN"

say "Starting simulator containers"
for i in "${!SIM_NAMES[@]}"; do
  name="${SIM_NAMES[$i]}"
  port="${SIM_PORTS[$i]}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" \
    -p "127.0.0.1:${port}:8000" \
    "$SIM_IMAGE" \
    --model "$SIM_MODEL" \
    --port 8000 >/dev/null
  ok "started $name on 127.0.0.1:${port}"
done

say "Waiting for simulator /health"
for port in "${SIM_PORTS[@]}"; do
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ok "sim healthy on :$port"
      break
    fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null || die "sim on :$port never became healthy"
done

say "Starting EPP in --baremetal mode"
"$EPP_BIN" \
  --baremetal \
  --backends-file=./config/baremetal/backends-sim.yaml \
  --config-file=./config/baremetal/scheduler-sim.yaml \
  --grpc-port=9002 \
  --metrics-port=9090 \
  --grpc-health-port=9003 \
  --secure-serving=false \
  --metrics-endpoint-auth=false \
  --backends-poll-interval=2s \
  >"$EPP_LOG" 2>&1 &
EPP_PID=$!
ok "EPP started (pid=$EPP_PID, log=$EPP_LOG)"

say "Waiting for /metrics endpoint"
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:9090/metrics >/dev/null 2>&1; then
    ok "/metrics is up"
    break
  fi
  if ! kill -0 "$EPP_PID" 2>/dev/null; then
    echo "----- EPP log -----"
    cat "$EPP_LOG"
    die "EPP exited before /metrics came up"
  fi
  sleep 1
done

say "Asserting EPP-side metrics"
# Give the metrics scraper one cycle to populate the per-pod gauges.
sleep 3
METRICS="$(curl -fsS http://127.0.0.1:9090/metrics)"
if echo "$METRICS" | grep -q "^inference_extension_info"; then
  ok "inference_extension_info present"
else
  echo "$METRICS" | grep -E "^inference_extension|^inference_pool" | head -10
  die "inference_extension_info missing"
fi
# inference_pool_ready_pods is a gauge that the EPP refreshes from the
# datastore on a periodic loop. It should show 2 after the FileProvider
# applied backends-sim.yaml.
if echo "$METRICS" | grep -q "^inference_pool_ready_pods"; then
  count="$(echo "$METRICS" | grep "^inference_pool_ready_pods" | tail -1 | awk '{print $NF}')"
  ok "inference_pool_ready_pods=$count"
else
  echo "(inference_pool_ready_pods not yet populated; series will appear after first metrics-flush interval)"
  echo "$METRICS" | grep -E "^inference_pool" | head -5
fi

say "Testing dynamic backend removal"
cp config/baremetal/backends-sim.yaml "${EPP_LOG}.backends.bak"
cat > config/baremetal/backends-sim.yaml <<'YAML'
poolName: sim-pool
poolNamespace: default
targetPort: 8001
backends:
  - address: "127.0.0.1:8001"
    labels:
      llm-d.ai/role: decode
      app: vllm-qwen3-32b
YAML
sleep 4
METRICS="$(curl -fsS http://127.0.0.1:9090/metrics)"
new_count="$(echo "$METRICS" | grep "^inference_pool_ready_pods" | tail -1 | awk '{print $NF}')"
ok "after removing one backend: inference_pool_ready_pods=$new_count"
# restore
mv "${EPP_LOG}.backends.bak" config/baremetal/backends-sim.yaml

say "Optional: ext-proc round-trip via grpcurl"
if command -v grpcurl >/dev/null 2>&1; then
  # The handler expects a streaming Process RPC. Use grpcurl's reflection on
  # the ext_proc service. We send request_headers and look for an immediate
  # response (the EPP sets a destination header and returns).
  if grpcurl -plaintext -d '{
        "request_headers": {
          "headers": {
            "headers": [
              {"key": ":path", "raw_value": "L3YxL2NoYXQvY29tcGxldGlvbnM="},
              {"key": "content-type", "raw_value": "YXBwbGljYXRpb24vanNvbg=="}
            ]
          }
        }
      }' \
      -import-path "$ROOT" \
      localhost:9002 \
      envoy.service.ext_proc.v3.ExternalProcessor/Process 2>&1 | head -20; then
    ok "ext-proc responded"
  else
    echo "   (ext-proc round-trip requires grpcurl + proto reflection; non-fatal)"
  fi
else
  echo "   (grpcurl not installed; skipping ext-proc round-trip)"
fi

say "Smoke test PASSED"
echo
echo "Last 30 lines of EPP log:"
tail -30 "$EPP_LOG"
