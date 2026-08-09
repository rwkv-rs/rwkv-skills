#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 MODEL INFER_BASE_URL RECOVERY_LABEL_PREFIX RECOVERY_TAG_PREFIX [AUDIT_ATTEMPTS] [AUDIT_INTERVAL_SECONDS] [RECOVERY_ROUNDS]" >&2
  exit 2
fi

model=$1
infer_base_url=$2
recovery_label_prefix=$3
recovery_tag_prefix=$4
audit_attempts=${5:-240}
audit_interval_s=${6:-60}
recovery_rounds=${7:-3}
runtime_repo=${RWKV_STRICT_FROZEN_RUNTIME:-}
run_id=${RWKV_STRICT_RUN_ID:-}
bootstrap_python=/usr/bin/python3
if [[ -z "$runtime_repo" || -z "$run_id" ]]; then
  echo "RWKV_STRICT_FROZEN_RUNTIME and RWKV_STRICT_RUN_ID are mandatory for audit recovery" >&2
  exit 42
fi
if [[ "${RWKV_STRICT_ENSURE_REEXEC:-0}" != 1 ]]; then
  exec env RWKV_STRICT_ENSURE_REEXEC=1 \
    "$runtime_repo/ops/g1i_strict46/ensure_model_complete.sh" "$@"
fi
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
executing_repo=$(dirname "$(dirname "$(dirname "$script_path")")")
runtime_repo=$(readlink -f -- "$runtime_repo")
if [[ "$executing_repo" != "$runtime_repo" ]]; then
  echo "frozen runtime path does not match executing recovery launcher" >&2
  exit 42
fi
approval=${RWKV_GLOBAL_PROTOCOL_APPROVAL:-}
if [[ -z "$approval" ]]; then
  echo "RWKV_GLOBAL_PROTOCOL_APPROVAL is mandatory for audit recovery" >&2
  exit 42
fi
frozen_approval="$runtime_repo/ops/g1i_strict46/approvals/$(basename "$approval")"
python=$("$bootstrap_python" -I "$runtime_repo/ops/g1i_strict46/require_global_protocol_gate.py" \
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
exec 9>"$state_root/locks/${safe_model}.lock"
# A restarted waiter must join the existing model gate rather than scheduling
# a second fresh recovery lane for the same strict-46 cells.
flock 9

for value_name in audit_attempts audit_interval_s recovery_rounds; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done

wait_for_audit() {
  RWKV_STRICT_PYTHON="$python" \
  RWKV_GLOBAL_PROTOCOL_APPROVAL="$frozen_approval" \
  "$runtime_repo/ops/g1i_strict46/wait_for_model_audit.sh" \
    "$model" "$audit_attempts" "$audit_interval_s"
}

require_runtime_attestation() {
  "$python" -I "$runtime_repo/ops/g1i_strict46/require_global_protocol_gate.py" \
    --repo "$runtime_repo" \
    --lock "$runtime_repo/ops/g1i_strict46/protocol_gate.lock.json" \
    --approval "$frozen_approval" --frozen-runtime "$runtime_repo" \
    --phase attest --model "$model" \
    --infer-base-url "$infer_base_url" --infer-api-key rwkv-skills \
    --require-current-python
}

# A scheduler can exit before its final score/replay transaction becomes
# visible.  Give the database the full audit window before creating any fresh
# recovery tasks; otherwise a healthy, merely delayed cell is duplicated.
if wait_for_audit; then
  exit 0
fi

for recovery_attempt in $(seq 1 "$recovery_rounds"); do
  recovery_label="${recovery_label_prefix}-${recovery_attempt}"
  recovery_tag="${recovery_tag_prefix}_${recovery_attempt}"

  # runtime_attestation is a mandatory parent-side preflight before even the
  # synchronous recovery runner is entered. run_audit_missing.py repeats the
  # proof in phase=recovery immediately before task creation.  Recovery stays
  # in this unprivileged process; it never asks a user manager to manufacture
  # a mutable transient unit.
  require_runtime_attestation
  if ! RWKV_STRICT_FROZEN_RUNTIME="$runtime_repo" \
      RWKV_STRICT_RUN_ID="$run_id" \
      RWKV_STRICT_STATE_ROOT="$state_root" \
      RWKV_STRICT_PYTHON="$python" \
      RWKV_GLOBAL_PROTOCOL_APPROVAL="$frozen_approval" \
      "$python" "$runtime_repo/ops/g1i_strict46/run_audit_missing.py" \
      "$model" "$infer_base_url" "$recovery_tag" \
      --audit-output "$state_root/logs/audits/g1i_strict46_recovery_${safe_model}_${recovery_attempt}.json"; then
    echo "synchronous recovery failed: $recovery_label" >&2
    continue
  fi

  if wait_for_audit; then
    exit 0
  fi
done

echo "$model did not pass the strict 46/46 audit after $recovery_rounds targeted recovery rounds" >&2
exit 24
