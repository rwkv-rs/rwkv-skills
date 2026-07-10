#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rwkv/chase/rwkv-skills}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/logs/agent_ready_missing_$STAMP}"
BASE_URL="${BASE_URL:-http://127.0.0.1:19183/v1}"
API_KEY="${API_KEY:-rwkv-skills}"
WAIT_SCREEN="${WAIT_SCREEN:-}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
MAX_PARALLEL_RUNS="${MAX_PARALLEL_RUNS:-1}"
INFER_MAX_WORKERS="${INFER_MAX_WORKERS:-1}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-1}"
DB_WRITE_QUEUE="${DB_WRITE_QUEUE:-8}"

MODELS=(
  "rwkv7-g1f-7.2b-20260414-ctx8192"
  "rwkv7-g1f-13.3b-20260415-ctx8192"
  "rwkv7-g1g-7.2b-20260523-ctx8192"
  "rwkv7-g1g-13.3b-20260523-ctx8192"
  "rwkv7-g1h-preview3121-7.2b-20260701-ctx8192"
  "rwkv7-g1h-preview4673-2.9b-20260701-ctx8192"
)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<<"$MODELS_OVERRIDE"
fi

FAST_AGENT_LOOP_DATASETS=(
  "widesearch"
  "deepsearchqa"
  "hle_with_tools"
)

BROWSECOMP_DATASETS=(
  "browsecomp"
  "browsecomp_zh"
)

BROWSECOMP_PLUS_DATASETS=(
  "browsecomp_plus"
)

COMPLEXFUNCBENCH_DATASETS=(
  "complexfuncbench_official"
  "complexfuncbench_subset"
)

LONG_CONTEXT_DATASETS=(
  "longbench"
  "longbench_qa"
  "longbench_qa_balanced"
)

DOCKER_AGENT_LOOP_DATASETS=(
  "terminal_bench_2_1"
  "nl2repo"
  "deepswe"
)

FORCE_FRESH_DATASETS="${FORCE_FRESH_DATASETS:-widesearch}"
REFRESH_DATASETS="${REFRESH_DATASETS:-}"
RERUN_SCORED_DATASETS="${RERUN_SCORED_DATASETS:-}"
ONLY_DATASETS="${ONLY_DATASETS:-}"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOG_DIR"
export PATH="$REPO_ROOT/.venv/bin:$PATH"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_DIR/summary.log"
}

has_score() {
  local dataset="$1"
  local model="$2"
  .venv/bin/python - "$dataset" "$model" <<'PY'
import os
import sys

import psycopg

dataset, model = sys.argv[1], sys.argv[2]
conn = psycopg.connect(
    host=os.getenv("PG_HOST", "127.0.0.1"),
    port=os.getenv("PG_PORT", "5432"),
    user=os.getenv("PG_USER", "postgres"),
    password=os.getenv("PG_PASSWORD", ""),
    dbname=os.getenv("PG_DBNAME") or "chase_rwkv_skills",
)
cur = conn.cursor()
cur.execute(
    """
    select 1
    from scores s
    join task t on t.task_id = s.task_id
    join benchmark b on b.benchmark_id = t.benchmark_id
    join model m on m.model_id = t.model_id
    where b.benchmark_name = %s
      and b.benchmark_split = 'test'
      and m.model_name = %s
      and coalesce(t.is_tmp, false) = false
    limit 1
    """,
    (dataset, model),
)
raise SystemExit(0 if cur.fetchone() else 1)
PY
}

has_live_runner() {
  local dataset="$1"
  local model="$2"
  pgrep -af "src.eval.tasks.function_calling.runner" \
    | grep -F -- "--infer-model $model" \
    | grep -E -- "--dataset (data/${dataset}/test\\.jsonl|${dataset}_test)" >/dev/null
}

force_fresh_dataset() {
  local dataset="$1"
  local item
  for item in $FORCE_FRESH_DATASETS; do
    if [[ "$item" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

rerun_scored_dataset() {
  local dataset="$1"
  local item
  for item in $RERUN_SCORED_DATASETS; do
    if [[ "$item" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

refresh_dataset() {
  local dataset="$1"
  local item
  for item in $REFRESH_DATASETS; do
    if [[ "$item" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

dataset_selected() {
  local dataset="$1"
  local item
  if [[ -z "$ONLY_DATASETS" ]]; then
    return 0
  fi
  for item in $ONLY_DATASETS; do
    if [[ "$item" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

run_dataset_model() {
  local dataset="$1"
  local model="$2"
  local kind="$3"
  local dataset_path="data/$dataset/test.jsonl"
  local dataset_arg="$dataset_path"
  local safe_model="${model//[^A-Za-z0-9_.-]/_}"
  local safe_dataset="${dataset//[^A-Za-z0-9_.-]/_}"
  local model_log="$LOG_DIR/${safe_dataset}__${safe_model}.log"
  local run_mode="auto"
  local decision_max_tokens=""
  local refresh_dataset="0"

  if ! dataset_selected "$dataset"; then
    log "skip_not_selected dataset=$dataset"
    return 0
  fi

  if force_fresh_dataset "$dataset"; then
    run_mode="fresh"
    dataset_arg="${dataset}_test"
  fi

  if refresh_dataset "$dataset"; then
    refresh_dataset="1"
    dataset_arg="${dataset}_test"
  fi

  if [[ ! -s "$dataset_path" && "$refresh_dataset" != "1" ]]; then
    log "missing_dataset dataset=$dataset path=$dataset_path"
    return 0
  fi

  if has_score "$dataset" "$model" && ! rerun_scored_dataset "$dataset"; then
    log "skip_score_exists dataset=$dataset model=$model"
    return 0
  fi
  if has_live_runner "$dataset" "$model"; then
    log "skip_live_runner dataset=$dataset model=$model"
    return 0
  fi
  if [[ "$kind" == "agent_loop" ]]; then
    decision_max_tokens="${AGENT_LOOP_DECISION_MAX_TOKENS:-4096}"
  fi
  if [[ "$dataset" == "widesearch" ]]; then
    decision_max_tokens="${WIDESEARCH_DECISION_MAX_TOKENS:-$decision_max_tokens}"
  fi

  log "run_start dataset=$dataset model=$model kind=$kind run_mode=$run_mode log=$model_log"
  export RWKV_TASK_DESC="agent_ready_missing_scores dataset=$dataset model=$model stamp=$STAMP"
  export RWKV_SKILLS_LOG_PATH="$model_log"

  local -a cmd=(
    .venv/bin/python -m src.eval.tasks.function_calling.runner
    --dataset "$dataset_arg"
    --run-mode "$run_mode"
    --infer-base-url "$BASE_URL"
    --infer-model "$model"
    --infer-api-key "$API_KEY"
    --infer-protocol completions
    --infer-seed-policy omit
    --infer-timeout-s 900
    --infer-max-workers "$INFER_MAX_WORKERS"
    --sample-workers "$SAMPLE_WORKERS"
    --history-max-chars 24000
    --long-doc-mode lexical
    --candidate-router-mode auto
    --max-steps 16
    --max-tool-errors 5
    --agent-loop-command-timeout-s "${AGENT_LOOP_COMMAND_TIMEOUT_S:-300}"
    --agent-loop-max-output-chars 8000
    --disable-checker
    --db-write-queue "$DB_WRITE_QUEUE"
  )

  if [[ "$kind" == "browsecomp" ]]; then
    cmd+=(--judge-max-workers "${JUDGE_MAX_WORKERS:-2}")
    decision_max_tokens="${BROWSECOMP_DECISION_MAX_TOKENS:-4096}"
  fi
  if [[ "$kind" == "browsecomp_plus" ]]; then
    cmd+=(--judge-max-workers "${JUDGE_MAX_WORKERS:-2}")
  fi
  if [[ -n "$decision_max_tokens" ]]; then
    cmd+=(--decision-max-tokens "$decision_max_tokens")
  fi
  if [[ "$kind" == "complexfuncbench" && "${COMPLEXFUNCBENCH_OFFLINE_COMPARE:-1}" == "1" ]]; then
    cmd+=(--complexfuncbench-offline-compare)
  fi

  local -a env_cmd=()
  if [[ "$refresh_dataset" == "1" ]]; then
    env_cmd+=(RWKV_EVAL_REFRESH_DATASET=1)
  fi
  if [[ "$kind" == "browsecomp" ]]; then
    env_cmd+=(RWKV_BROWSECOMP_AGENTIC="${RWKV_BROWSECOMP_AGENTIC:-1}")
  fi

  if env "${env_cmd[@]}" "${cmd[@]}" >"$model_log" 2>&1; then
    log "run_done dataset=$dataset model=$model"
  else
    local rc=$?
    log "run_failed rc=$rc dataset=$dataset model=$model"
  fi

  docker system df >>"$LOG_DIR/summary.log" 2>&1 || true
}

wait_for_parallel_slot() {
  local max_parallel="$1"
  if (( max_parallel <= 1 )); then
    return 0
  fi
  while (( $(jobs -pr | wc -l) >= max_parallel )); do
    wait -n || true
  done
}

launch_dataset_model() {
  local dataset="$1"
  local model="$2"
  local kind="$3"
  local max_parallel="${MAX_PARALLEL_RUNS:-1}"
  if (( max_parallel <= 1 )); then
    run_dataset_model "$dataset" "$model" "$kind"
    return
  fi
  wait_for_parallel_slot "$max_parallel"
  run_dataset_model "$dataset" "$model" "$kind" &
}

log "agent_ready_missing_start repo=$REPO_ROOT base_url=$BASE_URL"
if [[ -n "$WAIT_SCREEN" ]]; then
  while screen -ls | grep -q "$WAIT_SCREEN"; do
    log "waiting_for_screen screen=$WAIT_SCREEN sleep_s=$WAIT_SECONDS"
    sleep "$WAIT_SECONDS"
  done
  log "wait_screen_done screen=$WAIT_SCREEN"
fi

for dataset in "${FAST_AGENT_LOOP_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "agent_loop"
  done
done

for dataset in "${BROWSECOMP_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "browsecomp"
  done
done

for dataset in "${BROWSECOMP_PLUS_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "browsecomp_plus"
  done
done

for dataset in "${COMPLEXFUNCBENCH_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "complexfuncbench"
  done
done

for dataset in "${LONG_CONTEXT_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "longbench"
  done
done

for dataset in "${DOCKER_AGENT_LOOP_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_dataset_model "$dataset" "$model" "agent_loop"
  done
done

wait

log "agent_ready_missing_done"
