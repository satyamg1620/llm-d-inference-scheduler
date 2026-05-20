# Bare-metal targets. Plain `docker run` per component on a shared bridge
# network — no Kubernetes, no docker-compose. Backing scripts live under
# scripts/baremetal/. See docs/BAREMETAL_IMPLEMENTATION_PLAN.md.

BAREMETAL_EPP_IMAGE      ?= llm-d-epp:baremetal
BAREMETAL_EPP_IMAGE_PULL ?= quay.io/satyam16/llm-d-epp:baremetal

# The main Makefile `export`s EPP_IMAGE=ghcr.io/.../llm-d-router-endpoint-picker:dev
# which would otherwise leak into the bare-metal scripts (whose own default is
# `llm-d-epp:baremetal`). Every recipe below explicitly sets the right value.

##@ Bare-metal (no Kubernetes)

.PHONY: baremetal-help
baremetal-help: ## Show bare-metal targets and how they relate
	@printf '\n\033[1mBare-metal target groups\033[0m\n'
	@printf '  image       baremetal-epp-image\n'
	@printf '  test        baremetal-test, baremetal-smoke-test\n'
	@printf '  bring up    baremetal-start-sim, baremetal-start-uds,\n'
	@printf '              baremetal-start-real-vllm, baremetal-start-real-vllm-pull\n'
	@printf '  tear down   baremetal-stop, baremetal-stop-purge\n\n'
	@printf 'See docs/BAREMETAL_IMPLEMENTATION_PLAN.md for the phased plan,\n'
	@printf '    docs/BAREMETAL_HANDOFF.md for x86 reproduction,\n'
	@printf '    docs/BAREMETAL_OPERATIONS.md for the runbook.\n\n'

.PHONY: baremetal-epp-image
baremetal-epp-image: ## Build the EPP image (linux/$(TARGETARCH)) from Dockerfile.epp tagged $(BAREMETAL_EPP_IMAGE)
	docker build -f Dockerfile.epp -t $(BAREMETAL_EPP_IMAGE) .

.PHONY: baremetal-test
baremetal-test: ## go test the bare-metal-touched packages (no docker required)
	go test ./pkg/backend/... ./cmd/epp/runner/... ./pkg/epp/datalayer/...

.PHONY: baremetal-smoke-test
baremetal-smoke-test: ## End-to-end simulator smoke test (no GPU; pulls llm-d-inference-sim)
	EPP_IMAGE=$(BAREMETAL_EPP_IMAGE) ./scripts/baremetal/smoke-test.sh

.PHONY: baremetal-start-sim
baremetal-start-sim: baremetal-epp-image ## Bring up the simulator-based stack (2× sim + EPP + agentgateway)
	EPP_IMAGE=$(BAREMETAL_EPP_IMAGE) ./scripts/baremetal/start-stack.sh

.PHONY: baremetal-start-uds
baremetal-start-uds: baremetal-epp-image ## Same as start-sim but with the UDS tokenizer sidecar (5 containers)
	EPP_IMAGE=$(BAREMETAL_EPP_IMAGE) TOKENIZER=uds ./scripts/baremetal/start-stack.sh

.PHONY: baremetal-start-real-vllm
baremetal-start-real-vllm: baremetal-epp-image ## Bring up REAL vLLM stack (build EPP locally). Needs 2 NVIDIA GPUs.
	EPP_IMAGE=$(BAREMETAL_EPP_IMAGE) ./scripts/baremetal/start-stack-real-vllm.sh

.PHONY: baremetal-start-real-vllm-pull
baremetal-start-real-vllm-pull: ## Bring up REAL vLLM stack (pull pre-built EPP image). Needs 2 NVIDIA GPUs.
	EPP_IMAGE=$(BAREMETAL_EPP_IMAGE_PULL) ./scripts/baremetal/start-stack-real-vllm-pull.sh

.PHONY: baremetal-stop
baremetal-stop: ## Tear down any bare-metal stack (idempotent; volumes preserved)
	./scripts/baremetal/stop-stack.sh

.PHONY: baremetal-stop-purge
baremetal-stop-purge: ## Tear down AND drop all named volumes (HF + tokenizer caches)
	PURGE_VOLUMES=1 ./scripts/baremetal/stop-stack.sh
