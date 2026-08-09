#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL [ATTEMPTS] [INTERVAL_SECONDS]" >&2
  exit 2
fi

model=$1
attempts=${2:-240}
interval_s=${3:-60}
runtime_repo=${RWKV_STRICT_FROZEN_RUNTIME:-}
run_id=${RWKV_STRICT_RUN_ID:-}
if [[ -z "$runtime_repo" || -z "$run_id" ]]; then
  echo "RWKV_STRICT_FROZEN_RUNTIME and RWKV_STRICT_RUN_ID are mandatory for the model audit" >&2
  exit 42
fi
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
executing_repo=$(dirname "$(dirname "$(dirname "$script_path")")")
runtime_repo=$(readlink -f -- "$runtime_repo")
if [[ "$executing_repo" != "$runtime_repo" ]]; then
  echo "model audit must execute from the frozen runtime" >&2
  exit 42
fi
approval=${RWKV_GLOBAL_PROTOCOL_APPROVAL:-}
if [[ -z "$approval" ]]; then
  echo "RWKV_GLOBAL_PROTOCOL_APPROVAL is mandatory for the model audit" >&2
  exit 42
fi
frozen_approval="$runtime_repo/ops/g1i_strict46/approvals/$(basename "$approval")"
python=$(/usr/bin/python3 -I "$runtime_repo/ops/g1i_strict46/require_global_protocol_gate.py" \
  --repo "$runtime_repo" \
  --lock "$runtime_repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$runtime_repo" \
  --print-frozen-python)
state_root=$("$python" -I "$runtime_repo/ops/g1i_strict46/runtime_state.py" \
  --run-id "$run_id")
if [[ "${RWKV_STRICT_STATE_ROOT:-}" != "$state_root" ]]; then
  echo "RWKV_STRICT_STATE_ROOT does not match the verified per-run state" >&2
  exit 42
fi
safe_model=$(printf '%s' "$model" | tr -c 'A-Za-z0-9_.-' '_')
# Model handoff gates may run concurrently.  A per-model file prevents two
# audit writers from truncating the shared current-audit artifact at once.
audit_output="$state_root/logs/audits/g1i_strict46_gate_${safe_model}.json"

for attempt in $(seq 1 "$attempts"); do
  if (
    cd "$runtime_repo"
    RWKV_G1I_AUDIT_LOCK_PATH="$state_root/locks/g1i_strict46_audit.lock" \
    PYTHONPATH="$runtime_repo" "$python" ops/g1i_strict46/audit_current.py \
      --output "$audit_output" --require-model-complete "$model" >/dev/null
  ); then
    exit 0
  fi
  if (( attempt < attempts )); then
    echo "$model strict-46 audit incomplete (attempt $attempt/$attempts); waiting for replay/score commit" >&2
    sleep "$interval_s"
  fi
done

echo "$model did not pass the strict 46/46 database audit after $attempts attempts" >&2
exit 24
