#!/usr/bin/env bash
# Tear down everything started by start-stack.sh, start-stack-real-vllm.sh, or
# start-stack-real-vllm-pull.sh. Removes containers + the llmd-baremetal docker
# network. Idempotent.
#
# Docker volumes are PRESERVED by default — the HF model cache and tokenizer
# cache are expensive to rebuild. Pass PURGE_VOLUMES=1 to drop them too.
set -euo pipefail

NETWORK="llmd-baremetal"

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()  { printf '   \033[1;32m✓\033[0m %s\n' "$*"; }

say "Removing containers"
for name in agentgw epp tokenizer-uds vllm-0 vllm-1; do
  if docker rm -f "$name" >/dev/null 2>&1; then
    ok "removed $name"
  fi
done

say "Removing network"
if docker network rm "$NETWORK" >/dev/null 2>&1; then
  ok "removed $NETWORK"
else
  ok "$NETWORK was already gone"
fi

# Volumes are kept by default so model caches survive restarts.
# Pass PURGE_VOLUMES=1 to also remove them.
if [[ "${PURGE_VOLUMES:-}" == "1" ]]; then
  say "Removing docker volumes"
  for vol in llmd-tokenizer-uds llmd-tokenizer-models llmd-hf-cache; do
    if docker volume rm "$vol" >/dev/null 2>&1; then
      ok "removed $vol"
    fi
  done
fi
