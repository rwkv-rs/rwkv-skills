#!/usr/bin/env bash
set -euo pipefail

# Read-only final handoff barrier.  The caller must first complete the strict
# per-model database audit; this guard only observes runtime state and sockets.

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 MODEL ENDPOINT_PORT SCHEDULER_UNIT RECOVERY_UNIT_PREFIX [REQUIRED_IDLE_CHECKS] [INTERVAL_SECONDS] [MAX_OBSERVATIONS]" >&2
  exit 2
fi

model=$1
endpoint_port=$2
scheduler_unit=$3
recovery_unit_prefix=$4
required_idle_checks=${5:-2}
interval_s=${6:-10}
max_observations=${7:-12}
allow_collected_scheduler=${HANDOFF_ALLOW_COLLECTED_SCHEDULER:-0}

if [[ -z "$model" ]]; then
  echo "model must not be empty" >&2
  exit 2
fi
if [[ ! "$endpoint_port" =~ ^[0-9]+$ ]] \
    || (( endpoint_port < 1 || endpoint_port > 65535 )); then
  echo "endpoint port must be an integer in [1, 65535], got: $endpoint_port" >&2
  exit 2
fi
for value_name in required_idle_checks max_observations; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ ! "$interval_s" =~ ^[0-9]+$ ]]; then
  echo "interval_s must be a non-negative integer, got: $interval_s" >&2
  exit 2
fi
if (( max_observations < required_idle_checks )); then
  echo "max_observations must be >= required_idle_checks" >&2
  exit 2
fi
if [[ "$allow_collected_scheduler" != 0 && "$allow_collected_scheduler" != 1 ]]; then
  echo "HANDOFF_ALLOW_COLLECTED_SCHEDULER must be 0 or 1, got: $allow_collected_scheduler" >&2
  exit 2
fi

for required_command in systemctl ps ss sleep; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command not found: $required_command" >&2
    exit 2
  fi
done

busy_reasons=()

check_scheduler() {
  local load_state active_state result
  [[ "$scheduler_unit" == "-" ]] && return 0

  if ! load_state=$(systemctl show "$scheduler_unit" \
      -p LoadState --value 2>/dev/null); then
    busy_reasons+=("scheduler state unreadable: $scheduler_unit")
    return 0
  fi
  active_state=$(systemctl show "$scheduler_unit" \
    -p ActiveState --value 2>/dev/null || true)
  result=$(systemctl show "$scheduler_unit" \
    -p Result --value 2>/dev/null || true)

  # A root orchestrator may collect a completed system unit before this final
  # socket/process barrier executes.  That state is safe
  # only when the caller explicitly opts in *after* it has already observed a
  # successful scheduler result.  The default remains fail-closed so a typo
  # or an unreadable unit name can never authorize a handoff.
  if [[ "$load_state" == "not-found" && "$allow_collected_scheduler" == 1 ]]; then
    return 0
  fi
  if [[ "$load_state" != "loaded" ]]; then
    busy_reasons+=("scheduler not loaded: $scheduler_unit ($load_state)")
  elif [[ "$active_state" != "inactive" || "$result" != "success" ]]; then
    busy_reasons+=(
      "scheduler not safely complete: $scheduler_unit active=$active_state result=$result"
    )
  fi
}

check_recovery_units() {
  local unit_output unit load_state active_state sub_state remainder
  [[ "$recovery_unit_prefix" == "-" ]] && return 0

  if ! unit_output=$(systemctl list-units --all --type=service \
      --plain --no-legend "${recovery_unit_prefix}*.service" 2>/dev/null); then
    busy_reasons+=("recovery unit state unreadable: $recovery_unit_prefix")
    return 0
  fi
  while read -r unit load_state active_state sub_state remainder; do
    [[ -z "$unit" ]] && continue
    case "$active_state" in
      active|activating|reloading|deactivating)
        busy_reasons+=("active recovery unit: $unit ($active_state/$sub_state)")
        ;;
    esac
  done <<< "$unit_output"
}

check_eval_processes() {
  local process_output pid command is_eval_process matches_target
  if ! process_output=$(ps -eww -eo pid=,args=); then
    busy_reasons+=("process table unreadable")
    return 0
  fi

  while read -r pid command; do
    [[ -z "$pid" || "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    is_eval_process=0
    case "$command" in
      *"-m src.eval."*|*"run_audit_missing.py"*|*"recompute_math_from_completions.py"*|*"wait_replay_task.py"*)
        is_eval_process=1
        ;;
    esac
    (( is_eval_process == 0 )) && continue

    matches_target=0
    if [[ "$command" == *"--infer-model $model"* \
        || "$command" == *"--infer-models $model"* \
        || "$command" == *":$endpoint_port/v1"* ]]; then
      matches_target=1
    fi
    if (( matches_target == 1 )); then
      busy_reasons+=("matching evaluation process: pid=$pid")
    fi
  done <<< "$process_output"
}

check_endpoint_connections() {
  local socket_output recv_q send_q local_address peer_address process
  if ! socket_output=$(ss -H -tnp state established 2>/dev/null); then
    busy_reasons+=("established socket table unreadable")
    return 0
  fi

  while read -r recv_q send_q local_address peer_address process; do
    [[ -z "$local_address" ]] && continue
    if [[ "$local_address" == *":$endpoint_port" ]]; then
      busy_reasons+=(
        "established target-port connection: $local_address -> $peer_address"
      )
    fi
  done <<< "$socket_output"
}

idle_streak=0
for (( observation = 1; observation <= max_observations; observation += 1 )); do
  busy_reasons=()
  check_scheduler
  check_recovery_units
  check_eval_processes
  check_endpoint_connections

  if (( ${#busy_reasons[@]} == 0 )); then
    (( idle_streak += 1 ))
    echo "handoff idle guard: observation $observation/$max_observations idle ($idle_streak/$required_idle_checks)"
    if (( idle_streak >= required_idle_checks )); then
      echo "handoff idle guard passed for $model on port $endpoint_port"
      exit 0
    fi
  else
    idle_streak=0
    printf 'handoff idle guard: observation %d/%d busy:\n' \
      "$observation" "$max_observations" >&2
    printf '  - %s\n' "${busy_reasons[@]}" >&2
  fi

  if (( observation < max_observations )); then
    sleep "$interval_s"
  fi
done

echo "handoff idle guard did not observe $required_idle_checks consecutive idle checks for $model on port $endpoint_port" >&2
exit 25
