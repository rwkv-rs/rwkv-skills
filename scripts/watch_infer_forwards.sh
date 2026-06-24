#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/tmp/forward_watchdog"
LOG_DIR="${ROOT_DIR}/logs/forward_watchdog"
INTERVAL_SECONDS=30
ONCE=0
ONLY_PORT=""

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

usage() {
  cat <<'USAGE'
Usage: scripts/watch_infer_forwards.sh [--once] [--interval SECONDS] [--port 19083|19183]

Keeps local inference router forwards healthy:
  19083 -> chase@47.115.88.183:8222 127.0.0.1:19083
  19183 -> caizus@47.115.88.183:2333 127.0.0.1:19183

The watchdog checks local /healthz first. If local health fails, it confirms the
remote router is healthy and starts an ssh tunnel with ExitOnForwardFailure.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      shift
      ;;
    --interval)
      INTERVAL_SECONDS="${2:?missing interval}"
      shift 2
      ;;
    --port)
      ONLY_PORT="${2:?missing port}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ONLY_PORT}" != "" && "${ONLY_PORT}" != "19083" && "${ONLY_PORT}" != "19183" ]]; then
  echo "--port must be 19083 or 19183" >&2
  exit 2
fi

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_DIR}/watchdog.log"
}

profile_value() {
  local port="$1"
  local key="$2"

  case "${port}:${key}" in
    19083:name) echo "8222_19083" ;;
    19083:ssh_host) echo "chase@47.115.88.183" ;;
    19083:ssh_port) echo "8222" ;;
    19083:ssh_key) echo "" ;;
    19083:local_port) echo "19083" ;;
    19083:remote_port) echo "19083" ;;
    19083:health_url) echo "http://127.0.0.1:19083/healthz" ;;

    19183:name) echo "2333_19183" ;;
    19183:ssh_host) echo "caizus@47.115.88.183" ;;
    19183:ssh_port) echo "2333" ;;
    19183:ssh_key) echo "/home/chase/.ssh/id_server_new" ;;
    19183:local_port) echo "19183" ;;
    19183:remote_port) echo "19183" ;;
    19183:health_url) echo "http://127.0.0.1:19183/healthz" ;;

    *)
      echo "unknown profile ${port}:${key}" >&2
      exit 2
      ;;
  esac
}

local_health_ok() {
  local port="$1"
  local url
  url="$(profile_value "${port}" health_url)"
  curl -fsS --max-time 5 "${url}" >/dev/null
}

remote_health_ok() {
  local port="$1"
  local ssh_host ssh_port ssh_key remote_port
  ssh_host="$(profile_value "${port}" ssh_host)"
  ssh_port="$(profile_value "${port}" ssh_port)"
  ssh_key="$(profile_value "${port}" ssh_key)"
  remote_port="$(profile_value "${port}" remote_port)"

  local ssh_args=(
    -4
    -p "${ssh_port}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=10
  )
  if [[ -n "${ssh_key}" ]]; then
    ssh_args=(-i "${ssh_key}" "${ssh_args[@]}")
  fi

  ssh "${ssh_args[@]}" "${ssh_host}" "curl -fsS --max-time 5 http://127.0.0.1:${remote_port}/healthz >/dev/null"
}

start_tunnel() {
  local port="$1"
  local name ssh_host ssh_port ssh_key local_port remote_port tunnel_log
  name="$(profile_value "${port}" name)"
  ssh_host="$(profile_value "${port}" ssh_host)"
  ssh_port="$(profile_value "${port}" ssh_port)"
  ssh_key="$(profile_value "${port}" ssh_key)"
  local_port="$(profile_value "${port}" local_port)"
  remote_port="$(profile_value "${port}" remote_port)"
  tunnel_log="${LOG_DIR}/${name}.ssh.log"

  local ssh_args=(
    -4
    -f
    -N
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o ExitOnForwardFailure=yes
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -p "${ssh_port}"
    -L "${local_port}:127.0.0.1:${remote_port}"
    "${ssh_host}"
  )
  if [[ -n "${ssh_key}" ]]; then
    ssh_args=(-i "${ssh_key}" "${ssh_args[@]}")
  fi

  log "${name}: local health failed; starting ssh ${local_port}->${ssh_host}:${remote_port}"
  printf '%s starting ssh %s->%s:%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${local_port}" "${ssh_host}" "${remote_port}" >> "${tunnel_log}"
  ssh "${ssh_args[@]}"
}

ensure_forward() {
  local port="$1"
  local name
  name="$(profile_value "${port}" name)"

  if local_health_ok "${port}"; then
    log "${name}: local health ok"
    return 0
  fi

  if ! remote_health_ok "${port}"; then
    log "${name}: local health failed and remote health failed; not starting tunnel"
    return 1
  fi

  start_tunnel "${port}"

  for _ in {1..15}; do
    if local_health_ok "${port}"; then
      log "${name}: local health ok after ssh start"
      return 0
    fi
    sleep 1
  done

  log "${name}: ssh start attempted but local health still failed"
  return 1
}

selected_ports() {
  if [[ -n "${ONLY_PORT}" ]]; then
    printf '%s\n' "${ONLY_PORT}"
  else
    printf '%s\n' 19083 19183
  fi
}

main_once() {
  local failed=0
  local port
  while IFS= read -r port; do
    ensure_forward "${port}" || failed=1
  done < <(selected_ports)
  return "${failed}"
}

if [[ "${ONCE}" == "1" ]]; then
  main_once
else
  log "watchdog start interval=${INTERVAL_SECONDS}s port=${ONLY_PORT:-all}"
  while true; do
    main_once || true
    sleep "${INTERVAL_SECONDS}"
  done
fi
