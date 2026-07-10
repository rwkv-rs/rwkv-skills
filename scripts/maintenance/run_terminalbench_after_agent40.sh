#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rwkv/chase/rwkv-skills}"
WAIT_SCREEN="${WAIT_SCREEN:-scheduler_agent40_full_seedfix_20260709_013442}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/logs/terminalbench_seq_$STAMP}"

MODELS=(
  "rwkv7-g1f-7.2b-20260414-ctx8192"
  "rwkv7-g1f-13.3b-20260415-ctx8192"
  "rwkv7-g1g-7.2b-20260523-ctx8192"
  "rwkv7-g1g-13.3b-20260523-ctx8192"
  "rwkv7-g1h-preview3121-7.2b-20260701-ctx8192"
  "rwkv7-g1h-preview4673-2.9b-20260701-ctx8192"
)

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

echo "terminalbench_seq_start $(date '+%F %T %Z')" | tee -a "$LOG_DIR/summary.log"
echo "repo=$REPO_ROOT" | tee -a "$LOG_DIR/summary.log"
echo "wait_screen=$WAIT_SCREEN" | tee -a "$LOG_DIR/summary.log"

while screen -ls | grep -q "$WAIT_SCREEN"; do
  echo "waiting_for_$WAIT_SCREEN $(date '+%F %T %Z')" | tee -a "$LOG_DIR/summary.log"
  sleep 300
done

.venv/bin/python - <<'PY' | tee -a "$LOG_DIR/summary.log"
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

paths = prepare_dataset("terminal_bench_2_1", Path("data"), "test")
path = paths[0]
rows = sum(1 for _ in path.open(encoding="utf-8"))
print(f"prepared_terminal_bench path={path} rows={rows}")
PY

for model in "${MODELS[@]}"; do
  safe_model="${model//[^A-Za-z0-9_.-]/_}"
  model_log="$LOG_DIR/${safe_model}.log"
  echo "model_start $model $(date '+%F %T %Z')" | tee -a "$LOG_DIR/summary.log"
  .venv/bin/python -m src.eval.tasks.function_calling.runner \
    --dataset data/terminal_bench_2_1/test.jsonl \
    --run-mode fresh \
    --infer-base-url http://127.0.0.1:19183/v1 \
    --infer-model "$model" \
    --infer-api-key rwkv-skills \
    --infer-protocol completions \
    --infer-seed-policy omit \
    --infer-timeout-s 900 \
    --infer-max-workers 1 \
    --sample-workers 1 \
    --history-max-chars 24000 \
    --long-doc-mode lexical \
    --candidate-router-mode auto \
    --max-steps 16 \
    --max-tool-errors 5 \
    --agent-loop-command-timeout-s 900 \
    --agent-loop-max-output-chars 8000 \
    --disable-checker \
    >"$model_log" 2>&1
  echo "model_done $model $(date '+%F %T %Z')" | tee -a "$LOG_DIR/summary.log"
  docker system df >>"$LOG_DIR/summary.log" 2>&1 || true
done

echo "terminalbench_seq_done $(date '+%F %T %Z')" | tee -a "$LOG_DIR/summary.log"
