#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="local_6gpu_resume_20260622"
SMALL_SLOT_START="1"
SMALL_SLOTS="8"
INCLUDE_7B="1"
SEVENB_SLOTS="1"
INFER_BASE_URL="http://127.0.0.1:19083/v1"
INFER_MAX_WORKERS="32"
INFER_WORKER_PROFILE="fixed"
REMOTE_BATCH_SIZE="32"
PLAIN_CHOICE_BATCH_SIZE=""
PLAIN_CHOICE_TIMEOUT_S=""
CODING_EVAL_WORKERS=""
MAX_ACTIVE_CODING_RUNNERS=""
DISABLE_INFER_BACKPRESSURE=0
PID_DIR_OVERRIDE=""
RUN_LOG_DIR_OVERRIDE=""
JOB_SET="all"
FOREGROUND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --run-tag)
      RUN_TAG="${2:?missing run tag}"
      shift 2
      ;;
    --small-slots)
      SMALL_SLOTS="${2:?missing small slot count}"
      shift 2
      ;;
    --small-slot-start)
      SMALL_SLOT_START="${2:?missing small slot start}"
      shift 2
      ;;
    --include-7b)
      INCLUDE_7B="${2:?missing include-7b flag}"
      shift 2
      ;;
    --sevenb-slots)
      SEVENB_SLOTS="${2:?missing 7.2B slot count}"
      shift 2
      ;;
    --infer-base-url)
      INFER_BASE_URL="${2:?missing infer base url}"
      shift 2
      ;;
    --infer-max-workers)
      INFER_MAX_WORKERS="${2:?missing infer max workers}"
      shift 2
      ;;
    --infer-worker-profile)
      INFER_WORKER_PROFILE="${2:?missing infer worker profile}"
      shift 2
      ;;
    --remote-batch-size)
      REMOTE_BATCH_SIZE="${2:?missing remote batch size}"
      shift 2
      ;;
    --plain-choice-batch-size)
      PLAIN_CHOICE_BATCH_SIZE="${2:?missing plain choice batch size}"
      shift 2
      ;;
    --plain-choice-timeout-s)
      PLAIN_CHOICE_TIMEOUT_S="${2:?missing plain choice timeout seconds}"
      shift 2
      ;;
    --coding-eval-workers)
      CODING_EVAL_WORKERS="${2:?missing coding eval workers}"
      shift 2
      ;;
    --max-active-coding-runners)
      MAX_ACTIVE_CODING_RUNNERS="${2:?missing max active coding runners}"
      shift 2
      ;;
    --disable-infer-backpressure)
      DISABLE_INFER_BACKPRESSURE=1
      shift
      ;;
    --job-set)
      JOB_SET="${2:?missing job set}"
      shift 2
      ;;
    --pid-dir)
      PID_DIR_OVERRIDE="${2:?missing pid dir}"
      shift 2
      ;;
    --run-log-dir)
      RUN_LOG_DIR_OVERRIDE="${2:?missing run log dir}"
      shift 2
      ;;
    start)
      shift
      ;;
    *)
      echo "usage: $0 [start|--foreground] [--run-tag TAG] [--job-set all|naive] [--small-slots N] [--small-slot-start N] [--include-7b 0|1] [--sevenb-slots N] [--infer-base-url URL] [--infer-max-workers N] [--infer-worker-profile fixed|param-size] [--remote-batch-size N] [--plain-choice-batch-size N] [--plain-choice-timeout-s N] [--coding-eval-workers N] [--max-active-coding-runners N] [--disable-infer-backpressure] [--pid-dir DIR] [--run-log-dir DIR]" >&2
      exit 2
      ;;
  esac
done

DISPATCH_SESSION="rwkv_eval_${RUN_TAG}_s${SMALL_SLOTS}_7b${SEVENB_SLOTS}_w${INFER_MAX_WORKERS}_${INFER_WORKER_PROFILE}"
WATCHDOG_SESSION="rwkv_eval_19083_watchdog"

RUN_DIR="${ROOT_DIR}/logs/scheduler/${RUN_TAG}"
PID_DIR="${PID_DIR_OVERRIDE:-${RUN_DIR}/pids}"
RUN_LOG_DIR="${RUN_LOG_DIR_OVERRIDE:-${ROOT_DIR}/results/logs/${RUN_TAG}}"
DISPATCH_LOG="${RUN_DIR}/dispatcher_s${SMALL_SLOTS}_7b${SEVENB_SLOTS}_current.log"
WATCHDOG_LOG="${ROOT_DIR}/logs/forward_watchdog/watchdog_19083_tmux.log"

JOBS=(
  multi_choice_plain multi_choice_cot multi_choice_plain_naive multi_choice_cot_naive
  free_response free_response_judge free_response_naive free_response_judge_naive
  code_human_eval_naive code_mbpp_naive code_livecodebench_naive code_swe_bench code_swe_bench_naive
  instruction_following instruction_following_naive
  function_agentbench function_api_bank function_bfcl_ast function_bfcl_exec function_bfcl_v3
  function_browsecomp function_browsecomp_plus function_complexfuncbench function_longbench
  function_longcodebench function_mcp_bench function_tau_bench function_tau2_bench
  function_tau3_bench function_toolalpaca
)

JOB_ORDER=(
  code_human_eval_naive code_mbpp_naive code_livecodebench_naive code_swe_bench_naive
  multi_choice_plain_naive multi_choice_cot_naive multi_choice_plain multi_choice_cot
  free_response_naive free_response_judge_naive free_response free_response_judge
  instruction_following_naive code_swe_bench instruction_following
  function_agentbench function_api_bank function_bfcl_ast function_bfcl_exec
  function_bfcl_v3 function_browsecomp function_browsecomp_plus function_complexfuncbench
  function_longbench function_longcodebench function_mcp_bench function_tau_bench
  function_tau2_bench function_tau3_bench function_toolalpaca
)

DATASETS=(
  ceval cmmlu gpqa_main gpqa_extended gpqa_diamond include mmlu mmlu_pro mmlu_redux mmmlu supergpqa
  aime24 aime25 algebra222 answer_judge asdiv beyond_aime brumo25 college_math gsm_plus
  hendrycks_math hle hmmt_feb25 math_odyssey mawps omni_math polymath simpleqa svamp
  amc23 comp_math_24_25 gaokao2023en gsm8k math_500 minerva_math olympiadbench
  human_eval human_eval_cn human_eval_fix human_eval_plus mbpp mbpp_plus livecodebench
  swe_bench swe_bench_lite swe_bench_lite_bm25_13k swe_bench_lite_oracle swe_bench_verified
  ifeval ifbench
  agentbench_db agentbench_kg apibank_l1 apibank_l2 apibank_level1 apibank_level2
  bfcl_exec_multiple bfcl_exec_multiple_ast bfcl_exec_parallel bfcl_exec_parallel_multiple
  bfcl_exec_simple bfcl_exec_simple_ast bfcl_multiple bfcl_simple_python bfcl_v3
  browsecomp browsecomp_plus browsecomp_zh complexfuncbench_official complexfuncbench_subset
  longbench longbench_qa longbench_qa_balanced longcodeqa
  mcp_bench mcp_bench_multi_2server mcp_bench_multi_3server mcp_bench_single
  tau_bench_airline tau_bench_retail tau_bench_telecom
  tau2_bench_airline tau2_bench_retail tau2_bench_telecom
  tau3_bench_airline tau3_bench_banking_knowledge tau3_bench_mock tau3_bench_mock_long_context
  tau3_bench_retail tau3_bench_telecom toolalpaca_eval_real toolalpaca_eval_simulated
)

if [[ "${JOB_SET}" == "naive" ]]; then
  JOBS=(
    code_human_eval_naive code_mbpp_naive code_livecodebench_naive code_swe_bench_naive
    multi_choice_plain_naive multi_choice_cot_naive
    free_response_naive free_response_judge_naive
    instruction_following_naive
  )
  JOB_ORDER=("${JOBS[@]}")
elif [[ "${JOB_SET}" != "all" ]]; then
  echo "unknown --job-set '${JOB_SET}' (expected all or naive)" >&2
  exit 2
fi

build_slots() {
  local i suffix small_slot_end
  small_slot_end=$((SMALL_SLOT_START + SMALL_SLOTS - 1))
  for i in $(seq "${SMALL_SLOT_START}" "${small_slot_end}"); do
    printf -v suffix "%02d" "${i}"
    printf '%s\n' "g1f15_s${suffix}=rwkv7-g1f-1.5b-20260419-ctx8192"
  done
  for i in $(seq "${SMALL_SLOT_START}" "${small_slot_end}"); do
    printf -v suffix "%02d" "${i}"
    printf '%s\n' "g1g15_s${suffix}=rwkv7-g1g-1.5b-20260526-ctx8192"
  done
  for i in $(seq "${SMALL_SLOT_START}" "${small_slot_end}"); do
    printf -v suffix "%02d" "${i}"
    printf '%s\n' "g1f29_s${suffix}=rwkv7-g1f-2.9b-20260420-ctx8192"
  done
  for i in $(seq "${SMALL_SLOT_START}" "${small_slot_end}"); do
    printf -v suffix "%02d" "${i}"
    printf '%s\n' "g1g29_s${suffix}=rwkv7-g1g-2.9b-20260526-ctx8192"
  done
  if [[ "${INCLUDE_7B}" == "1" ]]; then
    for i in $(seq 1 "${SEVENB_SLOTS}"); do
      printf -v suffix "%02d" "${i}"
      printf '%s\n' "g1f72_s${suffix}=rwkv7-g1f-7.2b-20260414-ctx8192"
    done
    for i in $(seq 1 "${SEVENB_SLOTS}"); do
      printf -v suffix "%02d" "${i}"
      printf '%s\n' "g1g72_s${suffix}=rwkv7-g1g-7.2b-20260523-ctx8192"
    done
  fi
}

foreground_dispatch() {
  cd "${ROOT_DIR}"
  mkdir -p "${RUN_DIR}" "${PID_DIR}" "${RUN_LOG_DIR}"
  mapfile -t slots < <(build_slots)
  backpressure_args=()
  if [[ "${DISABLE_INFER_BACKPRESSURE}" == "1" ]]; then
    backpressure_args+=(--disable-infer-backpressure)
  fi
  coding_eval_args=()
  if [[ -n "${CODING_EVAL_WORKERS}" ]]; then
    coding_eval_args+=(--coding-eval-workers "${CODING_EVAL_WORKERS}")
  fi
  coding_limit_args=()
  if [[ -n "${MAX_ACTIVE_CODING_RUNNERS}" ]]; then
    coding_limit_args+=(--max-active-coding-runners "${MAX_ACTIVE_CODING_RUNNERS}")
  fi
  plain_choice_args=()
  if [[ -n "${PLAIN_CHOICE_BATCH_SIZE}" ]]; then
    plain_choice_args+=(--plain-choice-batch-size "${PLAIN_CHOICE_BATCH_SIZE}")
  fi
  if [[ -n "${PLAIN_CHOICE_TIMEOUT_S}" ]]; then
    plain_choice_args+=(--plain-choice-timeout-s "${PLAIN_CHOICE_TIMEOUT_S}")
  fi
  echo "$$" > "${RUN_DIR}/dispatcher.pid"
  exec .venv/bin/python -u -m src.eval.scheduler.cli dispatch \
    --log-dir "${RUN_DIR}" \
    --pid-dir "${PID_DIR}" \
    --run-log-dir "${RUN_LOG_DIR}" \
    --run-mode auto \
    --infer-base-url "${INFER_BASE_URL}" \
    --infer-models "${slots[@]}" \
    --only-jobs "${JOBS[@]}" \
    --job-order "${JOB_ORDER[@]}" \
    --only-datasets "${DATASETS[@]}" \
    --infer-timeout-s 900 \
    --infer-max-workers "${INFER_MAX_WORKERS}" \
    --infer-worker-profile "${INFER_WORKER_PROFILE}" \
    --infer-protocol "${INFER_PROTOCOL:-vllm}" \
    --infer-seed-policy "${INFER_SEED_POLICY:-preserve}" \
    --remote-batch-size "${REMOTE_BATCH_SIZE}" \
    "${plain_choice_args[@]}" \
    "${coding_eval_args[@]}" \
    "${coding_limit_args[@]}" \
    "${backpressure_args[@]}" \
    --dispatch-poll-seconds 20 \
    --skip-missing-dataset \
    --disable-checker \
    > "${DISPATCH_LOG}" 2>&1
}

start_watchdog() {
  mkdir -p "$(dirname "${WATCHDOG_LOG}")"
  if ! tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${WATCHDOG_SESSION}" -c "${ROOT_DIR}" \
      "exec scripts/watch_infer_forwards.sh --interval 30 --port 19083 >> '${WATCHDOG_LOG}' 2>&1"
  fi
}

start_dispatcher() {
  mkdir -p "${RUN_DIR}" "${PID_DIR}" "${RUN_LOG_DIR}"
  scripts/watch_infer_forwards.sh --once --port 19083
  start_watchdog
  local disable_backpressure_arg=()
  if [[ "${DISABLE_INFER_BACKPRESSURE}" == "1" ]]; then
    disable_backpressure_arg+=(--disable-infer-backpressure)
  fi
  local coding_eval_arg=()
  if [[ -n "${CODING_EVAL_WORKERS}" ]]; then
    coding_eval_arg+=(--coding-eval-workers "${CODING_EVAL_WORKERS}")
  fi
  local coding_limit_arg=()
  if [[ -n "${MAX_ACTIVE_CODING_RUNNERS}" ]]; then
    coding_limit_arg+=(--max-active-coding-runners "${MAX_ACTIVE_CODING_RUNNERS}")
  fi
  local plain_choice_arg=()
  if [[ -n "${PLAIN_CHOICE_BATCH_SIZE}" ]]; then
    plain_choice_arg+=(--plain-choice-batch-size "${PLAIN_CHOICE_BATCH_SIZE}")
  fi
  if [[ -n "${PLAIN_CHOICE_TIMEOUT_S}" ]]; then
    plain_choice_arg+=(--plain-choice-timeout-s "${PLAIN_CHOICE_TIMEOUT_S}")
  fi
  if tmux has-session -t "${DISPATCH_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${DISPATCH_SESSION}"
  fi
  tmux new-session -d -s "${DISPATCH_SESSION}" -c "${ROOT_DIR}" \
    "exec scripts/start_local_6gpu_dispatcher.sh --foreground --run-tag '${RUN_TAG}' --job-set '${JOB_SET}' --small-slots '${SMALL_SLOTS}' --small-slot-start '${SMALL_SLOT_START}' --include-7b '${INCLUDE_7B}' --sevenb-slots '${SEVENB_SLOTS}' --infer-base-url '${INFER_BASE_URL}' --infer-max-workers '${INFER_MAX_WORKERS}' --infer-worker-profile '${INFER_WORKER_PROFILE}' --remote-batch-size '${REMOTE_BATCH_SIZE}' ${plain_choice_arg[*]} ${coding_eval_arg[*]} ${coding_limit_arg[*]} ${disable_backpressure_arg[*]} --pid-dir '${PID_DIR}' --run-log-dir '${RUN_LOG_DIR}'"
  sleep 2
  echo "dispatch_session=${DISPATCH_SESSION}"
  echo "watchdog_session=${WATCHDOG_SESSION}"
  echo "dispatch_log=${DISPATCH_LOG}"
  echo "watchdog_log=${WATCHDOG_LOG}"
  echo "small_slots=${SMALL_SLOTS}"
  echo "small_slot_start=${SMALL_SLOT_START}"
  echo "include_7b=${INCLUDE_7B}"
  echo "sevenb_slots=${SEVENB_SLOTS}"
  echo "infer_max_workers=${INFER_MAX_WORKERS}"
  echo "infer_worker_profile=${INFER_WORKER_PROFILE}"
  echo "remote_batch_size=${REMOTE_BATCH_SIZE}"
  echo "plain_choice_batch_size=${PLAIN_CHOICE_BATCH_SIZE:-default}"
  echo "plain_choice_timeout_s=${PLAIN_CHOICE_TIMEOUT_S:-default}"
  echo "job_set=${JOB_SET}"
  echo "coding_eval_workers=${CODING_EVAL_WORKERS:-default}"
  echo "max_active_coding_runners=${MAX_ACTIVE_CODING_RUNNERS:-default}"
  echo "disable_infer_backpressure=${DISABLE_INFER_BACKPRESSURE}"
  echo "pid_dir=${PID_DIR}"
  echo "run_log_dir=${RUN_LOG_DIR}"
  if [[ -f "${RUN_DIR}/dispatcher.pid" ]]; then
    echo "dispatcher_pid=$(cat "${RUN_DIR}/dispatcher.pid")"
  fi
}

if [[ "${FOREGROUND}" == "1" ]]; then
  foreground_dispatch
else
  start_dispatcher
fi
