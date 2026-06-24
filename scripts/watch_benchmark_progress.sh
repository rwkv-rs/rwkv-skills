#!/usr/bin/env bash
set -uo pipefail

RUN_TAG="${RUN_TAG:-local_6gpu_resume_20260622}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
SCORE_ID_MIN="${SCORE_ID_MIN:-182}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/monitor/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/progress.log"
SCHEDULER_LOG_DIR="${ROOT_DIR}/logs/scheduler/${RUN_TAG}"
RUN_LOG_DIR="${ROOT_DIR}/results/logs/${RUN_TAG}"
PID_DIR="${SCHEDULER_LOG_DIR}/pids"
DISPATCHER_PID_FILE="${SCHEDULER_LOG_DIR}/dispatcher.pid"
ERROR_PATTERN='Traceback|RemoteHTTPError|timeout|Timeout|AttributeError|RuntimeError|ERROR|CUDA out of memory|HTTP 5|Connection refused|BrokenPipe|ReadTimeout|ConnectTimeout'

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PGHOST="${PG_HOST:-127.0.0.1}"
PGPORT="${PG_PORT:-5432}"
PGUSER="${PG_USER:-postgres}"
PGDATABASE="${PG_DBNAME:-rwkv-eval}"
export PGPASSWORD="${PG_PASSWORD:-}"

log_section() {
  printf '\n===== %s %s =====\n' "$(date '+%F %T %Z')" "$1"
}

run_psql() {
  psql -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${PGDATABASE}" -At -v ON_ERROR_STOP=1 -c "$1"
}

while true; do
  {
    log_section "process"
    if [[ -s "${DISPATCHER_PID_FILE}" ]]; then
      dispatcher_pid="$(cat "${DISPATCHER_PID_FILE}")"
      printf 'dispatcher_pid=%s\n' "${dispatcher_pid}"
      ps -p "${dispatcher_pid}" -o pid,stat,etime,cmd --no-headers || true
    else
      printf 'dispatcher_pid=missing\n'
    fi
    if [[ -d "${PID_DIR}" ]]; then
      printf 'runner_pid_files=%s\n' "$(find "${PID_DIR}" -maxdepth 1 -name '*.pid' | wc -l)"
    else
      printf 'runner_pid_files=0\n'
    fi

    log_section "db"
    run_psql "select max(score_id), count(*) from scores;" || true
    run_psql "select status, count(*) from task group by status order by status;" || true
    run_psql "select s.score_id, s.task_id, t.evaluator, b.benchmark_name, m.model_name, s.cot_mode, s.created_at, s.metrics from scores s join task t on t.task_id=s.task_id join benchmark b on b.benchmark_id=t.benchmark_id join model m on m.model_id=t.model_id where s.score_id >= ${SCORE_ID_MIN} order by s.score_id desc limit 20;" || true

    log_section "recent-running"
    run_psql "select t.task_id, t.evaluator, b.benchmark_name, m.model_name, t.status, coalesce((select count(*) from completions c where c.task_id=t.task_id),0) comps, coalesce((select count(*) from eval e join completions c on c.completions_id=e.completions_id where c.task_id=t.task_id),0) evals from task t join benchmark b on b.benchmark_id=t.benchmark_id join model m on m.model_id=t.model_id where t.status='Running' order by t.created_at desc limit 30;" || true

    log_section "backpressure"
    curl -fsS http://127.0.0.1:19083/v1/backpressure \
      | jq -r '.models | to_entries[] | [.key, (.value.status // ""), (.value.pending_queue // 0), (.value.service_queue // 0), (.value.engine_inbox // 0), (.value.active_records // 0), (.value.scheduler_waiting // 0), (.value.scheduler_running // 0)] | @tsv' \
      || true

    log_section "errors"
    if [[ -d "${SCHEDULER_LOG_DIR}" || -d "${RUN_LOG_DIR}" ]]; then
      rg -n "${ERROR_PATTERN}" "${SCHEDULER_LOG_DIR}" "${RUN_LOG_DIR}" -S | tail -n 80 || true
    fi

    log_section "sync-dry-run"
    .venv/bin/python scripts/sync_eval_db_to_remote.py --score-id-min "${SCORE_ID_MIN}" || true
  } >> "${LOG_FILE}" 2>&1

  sleep "${INTERVAL_SECONDS}"
done
