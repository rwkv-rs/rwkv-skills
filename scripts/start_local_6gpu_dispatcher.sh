#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="local-6gpu-full"
FOREGROUND=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    start)
      shift
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --profile)
      PROFILE="${2:?missing profile}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

run_scheduler() {
  cd "${ROOT_DIR}"
  exec uv run rwkv-skills-scheduler run --profile "${PROFILE}" "${EXTRA_ARGS[@]}"
}

if [[ "${FOREGROUND}" == "1" ]]; then
  run_scheduler
fi

SESSION="rwkv_eval_${PROFILE//[^A-Za-z0-9_]/_}"
cd "${ROOT_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi
cmd=(scripts/start_local_6gpu_dispatcher.sh --foreground --profile "${PROFILE}" "${EXTRA_ARGS[@]}")
printf -v tmux_cmd '%q ' "${cmd[@]}"
tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" \
  "exec ${tmux_cmd}"
echo "dispatch_session=${SESSION}"
echo "profile=${PROFILE}"
