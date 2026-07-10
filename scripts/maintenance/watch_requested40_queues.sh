#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rwkv/chase/rwkv-skills}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
RUN_ONCE="${RUN_ONCE:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/logs/requested40_watch_$STAMP}"
LOG_FILE="$LOG_DIR/hourly_summary.log"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOG_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

section() {
  printf '\n===== %s %s =====\n' "$(date '+%F %T %Z')" "$1"
}

collect_once() {
  section "screens"
  screen -ls || true

  section "runners"
  pgrep -af "src.eval.tasks.function_calling.runner|src.eval.tasks.coding.runner|src.eval.tasks.maths.runner|src.eval.tasks.knowledge.runner|src.main --config|run_ready_agent_missing_scores|run_requested40_missing_scores" || true

  section "agent_focus_matrix"
  .venv/bin/python scripts/maintenance/audit_agent_score_matrix.py \
    --dataset widesearch \
    --dataset deepsearchqa \
    --dataset terminal_bench_2_1 \
    --dataset nl2repo \
    --dataset deepswe || true

  section "requested40_summary"
  .venv/bin/python scripts/maintenance/audit_agent_score_matrix.py --suite requested40 \
    | awk 'BEGIN { stop = 0 } /^dataset[[:space:]]/ { stop = 1 } stop == 0 { print }' || true

  section "queue_logs"
  find results/logs -maxdepth 2 -type f -name summary.log \( \
      -path '*/agent_ready_missing_*/*' -o \
      -path '*/requested40_missing_*/*' \
    \) -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 8 \
    | cut -d' ' -f2- \
    | while IFS= read -r file; do
    printf '\n--- %s ---\n' "$file"
    tail -n 20 "$file" 2>/dev/null || true
  done

  section "web_search_proxy_health"
  curl -sS --max-time 5 "${RWKV_WEB_SEARCH_HEALTH_URL:-http://127.0.0.1:18901/health}" || true
  printf '\n'

  section "recent_errors"
  if command -v rg >/dev/null 2>&1; then
    find results/logs -maxdepth 2 -type f -name '*.log' \
      ! -name 'web_search_proxy_18901.log' \
      ! -path '*/requested40_watch_*/*' \
      -mmin -90 -print0 2>/dev/null \
      | xargs -0 -r rg -n "Traceback|RuntimeError|ValueError|ImportError|ModuleNotFoundError|RemoteHTTPError|Connection refused|ReadTimeout|ConnectTimeout|HTTP 5|CUDA out of memory" -S \
      | tail -n 80 || true
  else
    find results/logs -maxdepth 2 -type f -name '*.log' \
      ! -name 'web_search_proxy_18901.log' \
      ! -path '*/requested40_watch_*/*' \
      -mmin -90 -print0 2>/dev/null \
      | xargs -0 -r grep -EIn "Traceback|RuntimeError|ValueError|ImportError|ModuleNotFoundError|RemoteHTTPError|Connection refused|ReadTimeout|ConnectTimeout|HTTP 5|CUDA out of memory" \
      | tail -n 80 || true
  fi
}

while true; do
  collect_once >>"$LOG_FILE" 2>&1
  if [[ "$RUN_ONCE" == "1" ]]; then
    exit 0
  fi
  sleep "$(printf '%d' "$INTERVAL_SECONDS")"
done
