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

# The seven omitted datasets already have post-raw-protocol, fully audited
# scores for both 2.9B and 7.2B: MMLU, MMLU-Pro, C-Eval, HumanEval, MBPP,
# LiveCodeBench and IFEval.  Keep this recovery lane fresh so partial tasks
# created with the permanent Strategy-A think-close suppression cannot be
# resumed or mixed with corrected completions.
exec "$repo/.venv/bin/python" -m src.eval.scheduler.cli dispatch \
  --log-dir "logs/scheduler/$tag" \
  --pid-dir "logs/pids/$tag" \
  --run-log-dir "logs/runs/$tag" \
  --only-jobs \
    multi_choice_plain_naive \
    free_response_naive \
    free_response_judge_naive \
    code_human_eval_naive \
    code_mbpp_naive \
    code_livecodebench_plain_naive \
    instruction_following_naive \
  --only-datasets \
    arc_easy openbookqa cmmlu commonsense_qa truthfulqa_mc1 \
    hellaswag mmlu_redux winogrande agieval_mcq \
    mmlu_sr_question_and_answer bbh_mcq kmmlu gpqa_main gpqa_extended \
    medqa gpqa_diamond medmcqa arc_challenge \
    aime24 aime25 amc23 answer_judge beyond_aime brumo25 \
    comp_math_24_25 gaokao2023en gsm8k hmmt_feb25 math_500 \
    math_odyssey minerva_math olympiadbench simpleqa svamp \
    human_eval_cn human_eval_fix human_eval_plus mbpp_plus \
    ifbench \
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
  --run-mode fresh \
  --disable-checker \
  --disable-infer-backpressure \
  --dispatch-poll-seconds 3
