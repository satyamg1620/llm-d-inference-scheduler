#!/usr/bin/env bash
# Bring up the bare-metal llm-d stack with REAL vLLM:
#   2× vLLM (Qwen3-8B, one per GPU) + EPP (--baremetal) + agentgateway.
#
# This variant BUILDS the EPP image locally from Dockerfile.epp on each run.
# For the "pull pre-built image" variant, use start-stack-real-vllm-pull.sh.
#
# No Kubernetes. No docker-compose. Plain `docker run` on a shared bridge net.
# Verified end-to-end on x86_64 + 2× NVIDIA L40S; see
# `bare-metal-x86-verification.md` at the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

NET=${NET:-llmd-baremetal}
EPP_IMAGE=${EPP_IMAGE:-llm-d-epp:baremetal}
AGW_IMAGE=${AGW_IMAGE:-ghcr.io/agentgateway/agentgateway:v1.2.0-alpha.2}
# Pinned to v0.21.0 — the version used during the x86 verification. Override
# via VLLM_IMAGE=... to upgrade once you've re-verified with a newer release.
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.21.0}
MODEL=${MODEL:-Qwen/Qwen3-8B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
HF_CACHE_VOL=${HF_CACHE_VOL:-llmd-hf-cache}
# First run downloads ~16 GB of Qwen3-8B weights + torch.compile cache.
# Subsequent runs hit the volume and are fast.
VLLM_READY_TIMEOUT=${VLLM_READY_TIMEOUT:-900}

log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '   \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

log "Build the EPP image from Dockerfile.epp (this is the build-local variant)"
docker build -f Dockerfile.epp -t "$EPP_IMAGE" . >/dev/null
ok "$EPP_IMAGE built"

log "Resetting any prior stack"
for c in agentgw epp vllm-0 vllm-1; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done
docker network rm "$NET" >/dev/null 2>&1 || true

log "Creating Docker network $NET"
docker network create "$NET" >/dev/null
ok "network up"

log "Ensuring HF cache volume $HF_CACHE_VOL exists"
docker volume create "$HF_CACHE_VOL" >/dev/null
ok "volume ready"

start_vllm() {
  local name=$1 gpu_id=$2
  log "Starting $name on GPU $gpu_id (model=$MODEL)"
  # IMPORTANT: --kv-events-config endpoint must contain a literal '*' to bind.
  # `tcp://0.0.0.0:5557` is treated as a CONNECT target by vLLM, not a bind.
  # See bare-metal-x86-verification.md §6.2.
  docker run -d \
    --name "$name" \
    --network "$NET" \
    --gpus "device=$gpu_id" \
    --shm-size=8g \
    --ipc=host \
    -v "${HF_CACHE_VOL}:/root/.cache/huggingface" \
    ${HF_TOKEN:+-e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN} \
    "$VLLM_IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --block-size 16 \
    --kv-events-config "{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\", \"endpoint\": \"tcp://*:5557\", \"topic\": \"kv@${name}\"}" \
    >/dev/null
  ok "$name started"
}

start_vllm vllm-0 0
start_vllm vllm-1 1

log "Waiting for vLLM /health (timeout=${VLLM_READY_TIMEOUT}s; first run downloads ~16GB)"
wait_ready() {
  local name=$1 deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if docker exec "$name" curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      ok "$name healthy"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      docker logs --tail 60 "$name" || true
      die "$name container exited"
    fi
    sleep 5
  done
  docker logs --tail 60 "$name" || true
  die "$name did not become healthy within ${VLLM_READY_TIMEOUT}s"
}
wait_ready vllm-0
wait_ready vllm-1

log "Resolving vLLM container IPs (agentgateway passthrough needs IP:port)"
ip_of() {
  docker inspect -f "{{(index .NetworkSettings.Networks \"$NET\").IPAddress}}" "$1"
}
VLLM_0_IP=$(ip_of vllm-0); [[ -n $VLLM_0_IP ]] || die "no IP for vllm-0"
VLLM_1_IP=$(ip_of vllm-1); [[ -n $VLLM_1_IP ]] || die "no IP for vllm-1"
ok "vllm-0 = ${VLLM_0_IP}:8000"
ok "vllm-1 = ${VLLM_1_IP}:8000"

log "Generating backends-real-vllm.generated.yaml"
sed -e "s|__VLLM_0_IP__|${VLLM_0_IP}|g" \
    -e "s|__VLLM_1_IP__|${VLLM_1_IP}|g" \
    "${ROOT}/config/baremetal/backends-real-vllm.template.yaml" \
    > "${ROOT}/config/baremetal/backends-real-vllm.generated.yaml"
ok "wrote config/baremetal/backends-real-vllm.generated.yaml"

log "Starting EPP (image: $EPP_IMAGE)"
docker run -d \
  --name epp \
  --network "$NET" \
  -p 127.0.0.1:9090:9090 \
  -p 127.0.0.1:9002:9002 \
  -p 127.0.0.1:9003:9003 \
  -v "${ROOT}/config/baremetal:/etc/llm-d:ro" \
  "$EPP_IMAGE" \
  --baremetal \
  --backends-file=/etc/llm-d/backends-real-vllm.generated.yaml \
  --backends-poll-interval=5s \
  --config-file=/etc/llm-d/scheduler.yaml \
  --grpc-port=9002 \
  --grpc-health-port=9003 \
  --metrics-port=9090 \
  --secure-serving=false \
  --metrics-endpoint-auth=false \
  --health-checking \
  --v=2 \
  >/dev/null
ok "epp container started"

log "Waiting for EPP /metrics"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:9090/metrics >/dev/null 2>&1; then
    ok "EPP /metrics is up"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx epp; then
    docker logs --tail 80 epp || true
    die "epp container exited"
  fi
  sleep 2
done

log "Confirming EPP discovered both backends"
sleep 3
if curl -s http://127.0.0.1:9090/metrics | grep -q 'inference_pool_per_pod_queue_size'; then
  curl -s http://127.0.0.1:9090/metrics | grep 'inference_pool_per_pod_queue_size{' | head -4
  ok "EPP knows the pods"
else
  warn "no inference_pool_per_pod_queue_size yet — check EPP logs:  docker logs epp"
fi

log "Starting agentgateway (image: $AGW_IMAGE)"
docker run -d \
  --name agentgw \
  --network "$NET" \
  -p 127.0.0.1:8080:8080 \
  -v "${ROOT}/config/baremetal/agentgateway-real-vllm.yaml:/etc/agw/config.yaml:ro" \
  "$AGW_IMAGE" \
  -f /etc/agw/config.yaml \
  >/dev/null
ok "agentgw started"

log "Waiting for agentgateway to bind :8080"
for _ in $(seq 1 30); do
  if curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null | grep -qE '^(2|4)..$'; then
    ok "agentgateway is listening"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx agentgw; then
    docker logs --tail 80 agentgw || true
    die "agentgw exited"
  fi
  sleep 1
done

cat <<EOF

────────────────────────────────────────────────────────────────────────
  bare-metal llm-d stack (real vLLM) is up
────────────────────────────────────────────────────────────────────────
  network          $NET
  vllm-0           ${VLLM_0_IP}:8000   (GPU 0)
  vllm-1           ${VLLM_1_IP}:8000   (GPU 1)
  EPP gRPC         127.0.0.1:9002
  EPP health       127.0.0.1:9003
  EPP metrics      127.0.0.1:9090/metrics
  agentgateway     127.0.0.1:8080

  Try:
    curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"${MODEL}","messages":[{"role":"user","content":"hi"}],"max_tokens":32}' | jq .

  Logs:
    docker logs -f epp
    docker logs -f agentgw
    docker logs -f vllm-0
    docker logs -f vllm-1

  Tear down:
    ./scripts/baremetal/stop-stack.sh
────────────────────────────────────────────────────────────────────────
EOF
