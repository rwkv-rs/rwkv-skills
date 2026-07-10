#!/usr/bin/env bash
set -uo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/rwkv/chase/rwkv-skills}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_ROOT/results/logs/agent_ready_missing_public_agent_fresh_20260709_143222}"

while true; do
  printf '===== %s =====\n' "$(date '+%F %T %Z')"

  echo "--- gpu: rwkv-2333 ---"
  ssh -S none -n rwkv-2333 "hostname; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true" || true

  echo "--- gpu: rwkv-8222 ---"
  ssh -S none -n rwkv-8222 "hostname; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true" || true

  echo "--- 157 runners ---"
  ssh -S none -n rwkv-157 "pgrep -af 'src.eval.tasks.function_calling.runner' | grep -v pgrep | wc -l; pgrep -af 'src.eval.tasks.function_calling.runner' | grep -v pgrep || true" || true

  echo "--- db progress ---"
  ssh -S none -n rwkv-157 "cd '$REMOTE_ROOT' && set -a && . ./.env && set +a && .venv/bin/python - <<'PY'
import os
import psycopg

wanted = ['widesearch', 'deepsearchqa', 'browsecomp', 'browsecomp_plus', 'terminal_bench_2_1', 'nl2repo', 'deepswe']
conn = psycopg.connect(
    host=os.getenv('PG_HOST', '127.0.0.1'),
    port=os.getenv('PG_PORT', '5432'),
    user=os.getenv('PG_USER', 'postgres'),
    password=os.getenv('PG_PASSWORD', ''),
    dbname=os.getenv('PG_DBNAME') or 'chase_rwkv_skills',
)
cur = conn.cursor()
cur.execute(
    '''
    select b.benchmark_name, m.model_name, t.status, count(c.completions_id) as completions,
           max(t.created_at) as task_created
    from task t
    join benchmark b on b.benchmark_id=t.benchmark_id
    join model m on m.model_id=t.model_id
    left join completions c on c.task_id=t.task_id
    where b.benchmark_split=%s and b.benchmark_name = any(%s)
      and coalesce(t.is_tmp,false)=false
      and t.created_at > now() - (%s)::interval
    group by b.benchmark_name, m.model_name, t.task_id, t.status
    order by task_created desc, b.benchmark_name, m.model_name
    limit 40
    ''',
    ('test', wanted, '3 hours'),
)
for row in cur.fetchall():
    print(' | '.join(str(item) for item in row))
PY" || true

  echo "--- fresh summary tail ---"
  ssh -S none -n rwkv-157 "cd '$REMOTE_ROOT' && tail -80 '$REMOTE_LOG_DIR/summary.log' 2>/dev/null || true" || true

  echo "--- fresh errors tail ---"
  ssh -S none -n rwkv-157 "cd '$REMOTE_ROOT' && grep -R 'run_failed\\|Traceback\\|PREP_FAIL\\|Network is unreachable\\|BrowseComp agentic mode requires' -n '$REMOTE_LOG_DIR' 2>/dev/null | tail -80 || true" || true

  echo
  sleep "$INTERVAL_SECONDS"
done
