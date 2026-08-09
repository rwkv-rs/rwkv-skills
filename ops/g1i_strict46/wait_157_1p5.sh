#!/usr/bin/env bash
set -euo pipefail

# Unprivileged handoff requester.  It may finish/recover the current model and
# attest evidence, but it never stops/starts a service.  A separately installed
# root-owned system orchestrator must consume and independently verify the
# content-addressed request.

runtime_repo=${RWKV_STRICT_FROZEN_RUNTIME:-}
approval=${RWKV_GLOBAL_PROTOCOL_APPROVAL:-}
run_id=${RWKV_STRICT_RUN_ID:-}
if [[ -z "$runtime_repo" || -z "$approval" || -z "$run_id" ]]; then
  echo "frozen runtime, global approval, and run id are mandatory" >&2
  exit 42
fi
runtime_repo=$(readlink -f -- "$runtime_repo")
if [[ "${RWKV_STRICT_WAITER_REEXEC:-0}" != 1 ]]; then
  exec env RWKV_STRICT_WAITER_REEXEC=1 \
    RWKV_STRICT_FROZEN_RUNTIME="$runtime_repo" \
    RWKV_GLOBAL_PROTOCOL_APPROVAL="$approval" \
    RWKV_STRICT_RUN_ID="$run_id" \
    RWKV_STRICT_STATE_ROOT="${RWKV_STRICT_STATE_ROOT:-}" \
    "$runtime_repo/ops/g1i_strict46/wait_157_1p5.sh"
fi
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
repo=$(dirname "$(dirname "$(dirname "$script_path")")")
if [[ "$repo" != "$runtime_repo" ]]; then
  echo "handoff requester must execute from the frozen runtime" >&2
  exit 42
fi
frozen_approval="$repo/ops/g1i_strict46/approvals/$(basename "$approval")"
python=$(/usr/bin/python3 -I "$repo/ops/g1i_strict46/require_global_protocol_gate.py" \
  --repo "$repo" --lock "$repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$repo" --print-frozen-python)
state_root=$("$python" -I "$repo/ops/g1i_strict46/runtime_state.py" \
  --run-id "$run_id")
if [[ "${RWKV_STRICT_STATE_ROOT:-}" != "$state_root" ]]; then
  echo "RWKV_STRICT_STATE_ROOT does not match the verified per-run state" >&2
  exit 42
fi

current_model=rwkv7-g1i-2.9b-20260805-ctx16384
next_model=rwkv7-g1i-1.5b-20260805-ctx16384
endpoint=http://127.0.0.1:19439/v1
gate="$repo/ops/g1i_strict46/require_global_protocol_gate.py"

"$python" -I "$gate" --repo "$repo" \
  --lock "$repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$repo" \
  --phase audit --model "$current_model" --require-current-python

"$repo/ops/g1i_strict46/ensure_model_complete.sh" \
  "$current_model" "$endpoint" strict46-29-recovery strict46_29_recovery \
  240 60 3

"$repo/ops/g1i_strict46/handoff_idle_guard.sh" \
  "$current_model" 19439 - - 2 10 12

"$python" -I "$gate" --repo "$repo" \
  --lock "$repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$repo" \
  --phase attest --model "$current_model" \
  --infer-base-url "$endpoint" --infer-api-key rwkv-skills \
  --require-current-python
"$python" -I "$gate" --repo "$repo" \
  --lock "$repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$repo" \
  --phase launch --model "$next_model" --require-current-python

cd "$repo"
set +e
"$python" -m ops.g1i_strict46.handoff_request \
  --transition 157-2p9-to-1p5 --run-id "$run_id" \
  --frozen-runtime "$repo" --approval "$frozen_approval"
status=$?
set -e
if [[ $status -ne 75 ]]; then
  exit "$status"
fi
echo "handoff request recorded; root orchestrator has not changed any service" >&2
exit 75
