#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CURRENT_UNIT MODEL URL RECOVERY_TAG FOLLOWUP_UNIT FOLLOWUP_TAG" >&2
  exit 2
fi

current_unit=$1
model=$2
url=$3
recovery_tag=$4
followup_unit=$5
followup_tag=$6
repo=/home/rwkv/chase/rwkv-skills

# Only the dispatcher is SIGSTOP'ed. Its already-launched runners continue and
# must drain before the old unit is stopped.
while pgrep -f "src.eval.tasks.maths.runner.*--infer-model ${model}" >/dev/null; do
  sleep 30
done

systemctl --user stop "$current_unit"

# Recover only a failed/missing member of the eight-task first wave. Completed
# scores are skipped, so this cannot duplicate successful generation.
"$repo/ops/g1i_strict46/run_current_math_wave_recover.sh" \
  "$model" "$url" "$recovery_tag"

systemctl --user stop "$followup_unit" 2>/dev/null || true
systemd-run --user --unit="$followup_unit" --working-directory="$repo" \
  "$repo/ops/g1i_strict46/run_followup_29_72.sh" \
  "$model" "$url" "$followup_tag"
