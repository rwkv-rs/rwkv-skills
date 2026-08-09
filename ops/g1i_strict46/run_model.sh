#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_NAME INFER_BASE_URL TAG" >&2
  exit 2
fi

model_name=$1
infer_base_url=$2
tag=$3
bootstrap_python=/usr/bin/python3

if [[ "${RWKV_STRICT_FROZEN_REEXEC:-0}" != 1 ]]; then
  frozen_runtime=${RWKV_STRICT_FROZEN_RUNTIME:-}
  approval=${RWKV_GLOBAL_PROTOCOL_APPROVAL:-}
  if [[ -z "$frozen_runtime" || -z "$approval" ]]; then
    echo "RWKV_STRICT_FROZEN_RUNTIME and RWKV_GLOBAL_PROTOCOL_APPROVAL are mandatory" >&2
    exit 42
  fi
  frozen_runtime=$(readlink -f -- "$frozen_runtime")
  frozen_gate="$frozen_runtime/ops/g1i_strict46/require_global_protocol_gate.py"
  frozen_lock="$frozen_runtime/ops/g1i_strict46/protocol_gate.lock.json"
  frozen_approval="$frozen_runtime/ops/g1i_strict46/approvals/$(basename "$approval")"
  sealed_python=$("$bootstrap_python" -I "$frozen_gate" \
    --repo "$frozen_runtime" --lock "$frozen_lock" --approval "$frozen_approval" \
    --frozen-runtime "$frozen_runtime" --print-frozen-python)
  exec env \
    RWKV_STRICT_FROZEN_REEXEC=1 \
    RWKV_STRICT_FROZEN_RUNTIME="$frozen_runtime" \
    RWKV_STRICT_PYTHON="$sealed_python" \
    RWKV_GLOBAL_PROTOCOL_APPROVAL="$frozen_approval" \
    "$frozen_runtime/ops/g1i_strict46/run_model.sh" \
    "$model_name" "$infer_base_url" "$tag"
fi

script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
repo=$(dirname "$(dirname "$(dirname "$script_path")")")
if [[ "$repo" != "$(readlink -f -- "${RWKV_STRICT_FROZEN_RUNTIME:?missing frozen runtime after re-exec}")" ]]; then
  echo "frozen runtime path does not match executing launcher" >&2
  exit 42
fi
frozen_approval="$repo/ops/g1i_strict46/approvals/$(basename "${RWKV_GLOBAL_PROTOCOL_APPROVAL:?missing frozen approval}")"
python=$("$bootstrap_python" -I "$repo/ops/g1i_strict46/require_global_protocol_gate.py" \
  --repo "$repo" --lock "$repo/ops/g1i_strict46/protocol_gate.lock.json" \
  --approval "$frozen_approval" --frozen-runtime "$repo" --print-frozen-python)
state_root=$("$python" -I "$repo/ops/g1i_strict46/runtime_state.py" \
  --run-id "$tag" --create)
export RWKV_STRICT_RUN_ID="$tag"
export RWKV_STRICT_STATE_ROOT="$state_root"

cd "$repo"
export PG_DBNAME=chase_rwkv_skills_frontend46_20260804
export RWKV_BENCHMARK_CONFIG_ROOT="$repo/configs/g1h"
export PYTHONPATH="$repo"
# Pin every inherited environment switch that can alter persisted generations
# or scores.  A systemd user manager can retain variables from older
# experiments; relying on "normally unset" made strict reruns non-reproducible.
export RWKV_EVAL_REFRESH_DATASET=0
export RWKV_DATASET_REFRESH=0
export RWKV_MATH_DISABLE_ORACLE_CASCADE=0
export RWKV_MATH_PRIMARY_ONLY=0
export RWKV_MATH_RUN_CHECKER=0
export RWKV_KNOWLEDGE_RUN_CHECKER=0
export RWKV_CODING_RUN_CHECKER=0
export RWKV_SKILLS_DISABLE_CHECKER=1
export DISABLE_CHECKER=1
export RWKV_OMIT_PENALTY_DECAY=0
export RWKV_MATH_VERIFY_TIMEOUT_S=2
unset RWKV_KNOWLEDGE_COT_STRATEGY

gate="$repo/ops/g1i_strict46/require_global_protocol_gate.py"
# Mandatory runtime_attestation is performed inside the dispatch gate after
# its final approval/lock reread: local listeners are traced to vLLM, while a
# forwarded 8222 endpoint proves both the 157 SSH listener and the 8222 vLLM.
# This is the last operation before scheduler exec and therefore before any
# task creation, runner spawn, or database write.
"$python" -I "$gate" \
  --phase dispatch --model "$model_name" \
  --infer-base-url="$infer_base_url" --infer-api-key=rwkv-skills \
  --frozen-runtime "$repo" --require-current-python

exec "$python" -m src.eval.scheduler.cli dispatch \
  --log-dir "$state_root/logs/scheduler" \
  --pid-dir "$state_root/logs/pids" \
  --run-log-dir "$state_root/logs/runs" \
  --only-jobs \
    multi_choice_plain_naive \
    free_response_naive \
    free_response_judge_naive \
    code_human_eval_naive \
    code_mbpp_naive \
    code_livecodebench_plain_naive \
    instruction_following_naive \
  --only-datasets \
    arc_easy mmlu openbookqa cmmlu commonsense_qa ceval truthfulqa_mc1 \
    mmlu_pro hellaswag mmlu_redux winogrande agieval_mcq \
    mmlu_sr_question_and_answer bbh_mcq kmmlu gpqa_main gpqa_extended \
    medqa gpqa_diamond medmcqa arc_challenge \
    aime24 aime25 amc23 answer_judge beyond_aime brumo25 \
    comp_math_24_25 gaokao2023en gsm8k hmmt_feb25 math_500 \
    math_odyssey minerva_math olympiadbench simpleqa svamp \
    human_eval human_eval_cn human_eval_fix human_eval_plus mbpp mbpp_plus \
    livecodebench ifeval ifbench \
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
