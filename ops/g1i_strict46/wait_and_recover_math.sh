#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 MAIN_SERVICE MODEL_NAME INFER_BASE_URL TAG DATASET..." >&2
  exit 2
fi

main_service=$1
model_name=$2
infer_base_url=$3
tag=$4
shift 4
datasets=("$@")
repo=/home/rwkv/chase/rwkv-skills

while systemctl --user is-active --quiet "$main_service"; do
  sleep 20
done

main_result=$(systemctl --user show "$main_service" -p Result --value)
if [[ "$main_result" != "success" ]]; then
  echo "$main_service ended with Result=$main_result; preserving endpoint for diagnosis" >&2
  exit 21
fi

cd "$repo"
export PG_DBNAME=chase_rwkv_skills_frontend46_20260804
export RWKV_BENCHMARK_CONFIG_ROOT="$repo/configs/g1h"
export PYTHONPATH="$repo"

"$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
  --log-dir "logs/scheduler/$tag" \
  --pid-dir "logs/pids/$tag" \
  --run-log-dir "logs/runs/$tag" \
  --only-jobs \
    free_response_naive \
    free_response_judge_naive \
    instruction_following_naive \
  --only-datasets "${datasets[@]}" \
  --infer-base-url "$infer_base_url" \
  --infer-models "$model_name" \
  --infer-api-key rwkv-skills \
  --infer-timeout-s 1800 \
  --infer-max-workers 64 \
  --infer-slots-per-model 8 \
  --infer-protocol vllm \
  --infer-seed-policy omit \
  --remote-batch-size 64 \
  --plain-choice-batch-size 128 \
  --math-judge-max-workers 32 \
  --run-mode rerun \
  --disable-checker \
  --disable-infer-backpressure \
  --dispatch-poll-seconds 3

# Refresh the shared evidence after this recovery lane.  Multiple independent
# recovery lanes for one model may finish in either order, so an individual
# lane must not require the *other* lanes to have finished.  The consolidated
# model hand-off waiter is the single 46/46 acceptance gate.
"$repo/.venv/bin/python" "$repo/ops/g1i_strict46/audit_current.py" \
  --output "$repo/logs/audits/g1i_strict46_current.json"
