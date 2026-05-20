# Bare-metal llm-d-inference-scheduler

Run the EPP (Endpoint Picker) without Kubernetes.

## Three ways to bring up the stack (pick one)

1. **Simulator** (no GPU, fastest) — uses `llm-d-inference-sim` containers in
   place of real vLLM. Good for development, plugin testing, and CI smoke.
   See [`scripts/baremetal/start-stack.sh`](../../scripts/baremetal/start-stack.sh).

2. **Real vLLM, build EPP locally** — two vLLMs on real GPUs, EPP image built
   from `Dockerfile.epp` on the spot. Good for first-time real bring-up.
   See [`scripts/baremetal/start-stack-real-vllm.sh`](../../scripts/baremetal/start-stack-real-vllm.sh).

3. **Real vLLM, pull pre-built EPP** — same as #2 but pulls
   `quay.io/satyam16/llm-d-epp:baremetal` instead of building. Good for
   shared dev machines where you don't want to rebuild on every clone.
   See [`scripts/baremetal/start-stack-real-vllm-pull.sh`](../../scripts/baremetal/start-stack-real-vllm-pull.sh).

All three share the same teardown:
[`scripts/baremetal/stop-stack.sh`](../../scripts/baremetal/stop-stack.sh).

---

## Manual quick start (no scripts) — real vLLM

```bash
# 1. Build the EPP binary (same binary as K8s mode).
make build-epp        # or: go build -o bin/epp ./cmd/epp

# 2. Start vLLM with KV events. The endpoint must contain a literal '*'
#    (not 0.0.0.0) — vLLM treats anything without '*' as a connect target,
#    not a bind. See `bare-metal-x86-verification.md` §6.2.
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --port 8000 --host 0.0.0.0 \
  --block-size 16 \
  --enable-prefix-caching \
  --kv-events-config '{
    "enable_kv_cache_events": true,
    "publisher": "zmq",
    "endpoint": "tcp://*:5557",
    "topic": "kv@vllm-0"
  }'

# 3. Edit config/baremetal/backends.yaml to match your vLLM IPs. The address
#    must be IP:port (not hostname) so agentgateway's passthrough can dial it.

# 4. Run the EPP in bare-metal mode.
./bin/epp \
  --baremetal \
  --backends-file=./config/baremetal/backends.yaml \
  --config-file=./config/baremetal/scheduler.yaml \
  --grpc-port=9002 \
  --metrics-port=9090 \
  --secure-serving=false \
  --metrics-endpoint-auth=false

# 5. Point your gateway (agentgateway or Envoy) at 127.0.0.1:9002 via ext-proc.
```

The EPP **subscribes** to each backend's ZMQ publisher (SUB dials, PUB binds).
You do NOT need a centralized ZMQ socket on the EPP side, and you do NOT set
`kvEventsConfig.zmqEndpoint` in `scheduler.yaml` — that field is a no-op in
`kv-cache@v0.8.0` and earlier docs that suggested centralized mode were wrong.

## What's different from Kubernetes mode

| Concern              | Kubernetes mode                                 | Bare-metal mode                                  |
| -------------------- | ----------------------------------------------- | ------------------------------------------------ |
| Backend discovery    | InferencePool CRD + EndpointSlice reconciler    | `backends.yaml` polled every 10s                 |
| Backend health       | Pod readiness probe (instant)                   | HTTP `/metrics` poll (slower)                    |
| Pool identity        | InferencePool CR                                | `poolName` + `poolNamespace` in `backends.yaml`  |
| Backend labels       | Pod labels (read by PodReconciler)              | `labels:` block per backend                      |
| ZMQ KV events        | `discoverPods: true` + RBAC                     | `discoverPods: false`; SUB dials per backend     |
| Tokenizer            | UDS sidecar (`/tmp/tokenizer/...`)              | vLLM HTTP `/tokenize` (default), HF cache, or UDS |
| Manager / reconciler | controller-runtime                              | none                                             |

## HuggingFace model access (`HF_TOKEN`)

The default model (`Qwen/Qwen3-8B`) is public and needs no authentication.
For gated models — `meta-llama/*`, `mistralai/*`, etc. — export `HF_TOKEN`
before launching the real-vLLM stack; both `start-stack-real-vllm.sh` and
`start-stack-real-vllm-pull.sh` forward it into the vLLM containers as
`HUGGING_FACE_HUB_TOKEN`:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
MODEL=meta-llama/Llama-3.1-8B-Instruct ./scripts/baremetal/start-stack-real-vllm-pull.sh
```

The UDS tokenizer mode (`TOKENIZER=uds ./scripts/baremetal/start-stack.sh`)
forwards `HF_TOKEN` to the `tokenizer-uds` container the same way. The
simulator-based stack ignores `HF_TOKEN` because the simulator doesn't
download model files.

## Tokenizer backends

The `token-producer` plugin (used by `precise-prefix-cache-scorer`) supports
three backends; pick one in `scheduler.yaml`:

| Mode | `parameters` shape | Extra container | When to use |
|---|---|---|---|
| vLLM HTTP /tokenize (recommended for bare-metal) | `vllm: { http: http://<host>:<port> }` | None — vLLM is already there | Default; simplest. |
| In-process HuggingFace cache | `hf: { tokenizersCacheDir: /var/lib/llm-d/tokenizers }` | None | Pre-populated cache; isolated from vLLM. |
| UDS tokenizer sidecar | `udsTokenizerConfig: { socketFile: /tmp/tokenizer/tokenizer-uds.socket }` | `ghcr.io/llm-d/llm-d-uds-tokenizer:v0.7.1` | K8s parity; image-size discipline. See `scheduler-uds.yaml`. |

## Active health checking 

The FileProvider can actively probe each backend's HTTP `/health` and remove
non-responsive backends from rotation. Disabled by default; enable by setting
`--health-check-interval` to a non-zero duration on the EPP.

| Flag | Default | What it does |
|------|---------|--------------|
| `--health-check-interval` | `0` (off) | How often to probe each backend. `0` disables the entire feature. |
| `--health-check-path` | `/health` | HTTP path to probe. Use `/v1/models` if your server doesn't expose `/health`. |
| `--health-check-timeout` | `2s` | Per-probe HTTP timeout. A probe that exceeds this counts as a failure. |
| `--health-check-failure-threshold` | `3` | Consecutive failed probes before `EndpointDelete`. One success re-admits via `EndpointUpsert`. |

Recommended production setup:

```bash
./bin/epp --baremetal ... \
  --health-check-interval=5s \
  --health-check-path=/health \
  --health-check-timeout=2s \
  --health-check-failure-threshold=3
```

Verify it's running:

```bash
docker logs epp 2>&1 | grep "starting health-check loop"
# expected: {"msg":"starting health-check loop","interval":5,...}
```

Per-transition log entries:

```
{"msg":"backend failed health checks; removed from rotation",
 "backend":"172.18.0.2:8000","consecutiveFailures":3,"failureThreshold":3}
{"msg":"backend recovered; re-admitted",
 "backend":"172.18.0.2:8000","priorConsecutiveFailures":22}
```

## Limitations

- LoRA adapter routing (InferenceModel CRD) is not supported in bare-metal.
- Autoscaling: external (operator edits `backends.yaml`, ~10s propagation by default; instant with active health-check enabled).
- Multi-pool: run a second EPP process with its own `backends.yaml`.
- **Session-affinity auto-rotation requires response-phase ext-proc.**
  agentgateway's `inferenceRouting.destinationMode: passthrough` doesn't
  forward response phases to the EPP, so the response-side hook that
  writes `x-session-token` back to clients doesn't fire. The
  `session-affinity-scorer` plugin still pins requests when a client
  explicitly sends `x-session-token: <base64(<namespace>/<pod-name>)>`;
  only the automatic round-trip is missing. Tracked for upstream
  agentgateway.
