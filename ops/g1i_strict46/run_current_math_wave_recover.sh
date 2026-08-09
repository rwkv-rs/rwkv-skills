#!/usr/bin/env bash
set -euo pipefail

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
  --only-jobs free_response_naive free_response_judge_naive \
  --only-datasets aime24 aime25 amc23 beyond_aime brumo25 gsm8k hmmt_feb25 math_500 \
  --infer-base-url "$infer_base_url" \
  --infer-models "$model_name" \
  --infer-api-key rwkv-skills \
  --infer-timeout-s 1800 --infer-max-workers 64 --infer-slots-per-model 8 \
  --infer-protocol vllm --infer-seed-policy omit --remote-batch-size 64 \
  --math-judge-max-workers 32 --run-mode missing --disable-checker \
  --disable-infer-backpressure --dispatch-poll-seconds 3
