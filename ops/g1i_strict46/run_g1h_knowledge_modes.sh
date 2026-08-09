#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_NAME INFER_BASE_URL TAG" >&2
  exit 2
fi

model=$1
url=$2
tag=$3
repo=/home/rwkv/chase/rwkv-skills
common=(
  --infer-base-url "$url" --infer-models "$model" --infer-api-key rwkv-skills
  --infer-timeout-s 1800 --infer-max-workers 64 --infer-slots-per-model 8
  --infer-protocol vllm --infer-seed-policy omit --remote-batch-size 64
  --plain-choice-batch-size 128 --run-mode fresh --disable-checker
  --disable-infer-backpressure --dispatch-poll-seconds 3
)
knowledge=(
  mmlu mmlu_pro mmlu_redux mmlu_sr_question_and_answer
  gpqa_diamond gpqa_main gpqa_extended arc_challenge arc_easy
  hellaswag bbh_mcq agieval_mcq truthfulqa_mc1 winogrande
  openbookqa commonsense_qa ceval cmmlu kmmlu medqa medmcqa
)

cd "$repo"
export PG_DBNAME=chase_rwkv_skills_frontend46_20260804
export RWKV_BENCHMARK_CONFIG_ROOT="$repo/configs/g1h"
export PYTHONPATH="$repo"

"$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
  --log-dir "logs/scheduler/${tag}_cot" \
  --pid-dir "logs/pids/${tag}_cot" \
  --run-log-dir "logs/runs/${tag}_cot" \
  --only-jobs multi_choice_cot_naive \
  --only-datasets "${knowledge[@]}" \
  "${common[@]}"

# The replay audit found exactly five G1h 7.2B NoCoT cells with >5% missing
# predictions. Other G1h NoCoT cells are preserved and never regenerated.
if [[ "$model" == *"7.2b"* ]]; then
  "$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
    --log-dir "logs/scheduler/${tag}_nocot_retest" \
    --pid-dir "logs/pids/${tag}_nocot_retest" \
    --run-log-dir "logs/runs/${tag}_nocot_retest" \
    --only-jobs multi_choice_plain_naive \
    --only-datasets agieval_mcq ceval cmmlu medmcqa mmlu_pro \
    "${common[@]}"
fi
