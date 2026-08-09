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

# The initial corrected wave already owns eight Math datasets.  The persistent
# strict-46 protocol does not include Knowledge CoT, so this follow-up contains
# only the 13 remaining strict cells: eight Math, four Coding, and IFBench.
# ``missing`` is intentional: if the first dispatcher was resumed for recovery
# and already launched any of these cells, reuse or resume that protocol-
# compatible work instead of creating a duplicate fresh score.
exec "$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
  --log-dir "logs/scheduler/$tag" \
  --pid-dir "logs/pids/$tag" \
  --run-log-dir "logs/runs/$tag" \
  --only-jobs \
    free_response_naive \
    free_response_judge_naive \
    code_human_eval_naive \
    code_mbpp_naive \
    instruction_following_naive \
  --only-datasets \
    answer_judge comp_math_24_25 gaokao2023en math_odyssey \
    minerva_math olympiadbench simpleqa svamp \
    human_eval_cn human_eval_fix human_eval_plus mbpp_plus ifbench \
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
  --coding-eval-workers 32 \
  --max-active-coding-runners 2 \
  --math-judge-max-workers 32 \
  --run-mode missing \
  --disable-checker \
  --disable-infer-backpressure \
  --dispatch-poll-seconds 3
