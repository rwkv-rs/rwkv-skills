#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_NAME INFER_BASE_URL TAG" >&2
  exit 2
fi

model_name=$1
infer_base_url=$2
tag=$3
repo=/home/rwkv/chase/rwkv-skills

cd "$repo"
export PG_DBNAME=chase_rwkv_skills_frontend46_20260804
export RWKV_BENCHMARK_CONFIG_ROOT="$repo/configs/g1h"
export PYTHONPATH="$repo"

exec "$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
  --log-dir "logs/scheduler/$tag" \
  --pid-dir "logs/pids/$tag" \
  --run-log-dir "logs/runs/$tag" \
  --only-jobs multi_choice_plain_naive \
  --only-datasets \
    mmlu mmlu_pro mmlu_redux mmlu_sr_question_and_answer \
    gpqa_diamond gpqa_main gpqa_extended \
    arc_challenge arc_easy hellaswag bbh_mcq agieval_mcq truthfulqa_mc1 \
    winogrande openbookqa commonsense_qa ceval cmmlu kmmlu medqa medmcqa \
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
  --run-mode fresh \
  --disable-checker \
  --disable-infer-backpressure \
  --dispatch-poll-seconds 3
