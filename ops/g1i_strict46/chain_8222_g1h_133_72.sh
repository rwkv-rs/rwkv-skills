#!/usr/bin/env bash
set -euo pipefail

ssh_target=rwkv@192.168.0.157
ssh_key=/home/chase/.ssh/id_ed25519
weights=/home/chase/weights/BlinkDL__rwkv7-g1
port=18074
current_scheduler=rwkv-g1i-strict46-133-raw-20260806.service
current_infer=rwkv-g1i-13p3b-gpu2-16k-c640.service

remote() {
  ssh -i "$ssh_key" -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=no -o ControlPath=none "$ssh_target" "$@"
}

wait_remote_unit() {
  local unit=$1 state
  while true; do
    state=$(remote systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)
    [[ -n "$state" && "$state" != not-found ]] && break
    sleep 30
  done
  while remote systemctl --user is-active --quiet "$unit"; do sleep 30; done
  [[ "$(remote systemctl --user show "$unit" -p Result --value 2>/dev/null || true)" == success ]]
}

deploy() {
  local model=$1 unit=$2
  systemctl --user stop "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --property=Restart=on-failure --property=RestartSec=5s \
    --setenv=CUDA_VISIBLE_DEVICES=2 --setenv=VLLM_USE_V2_MODEL_RUNNER=1 \
    --setenv=VLLM_RWKV7_WKV_MODE=fp32io16 --setenv=VLLM_USE_RAPID_SAMPLER=1 \
    --setenv=VLLM_USE_FLASHINFER_SAMPLER=0 --setenv=PYTHONPATH=/home/chase/vllm-rwkv \
    /home/chase/.venv-vllm-56b463bf6/bin/vllm serve \
    "$weights/$model.pth" --host 127.0.0.1 --port "$port" --api-key rwkv-skills \
    --tokenizer-mode rwkv --trust-request-chat-template \
    --enable-auto-tool-choice --tool-call-parser rwkv \
    --max-model-len 10240 --served-model-name "$model" \
    --gpu-memory-utilization 0.98 --max-num-batched-tokens 98304 --max-num-seqs 640 \
    --override-generation-config '{"temperature":1e-5}'
  for _ in $(seq 1 150); do
    curl -fsS --max-time 3 -H 'Authorization: Bearer rwkv-skills' \
      "http://127.0.0.1:$port/v1/models" >/dev/null && return 0
    sleep 3
  done
  return 21
}

run_eval() {
  local model=$1 unit=$2 tag=$3
  remote systemctl --user stop "$unit" 2>/dev/null || true
  remote systemd-run --user --unit="$unit" \
    --working-directory=/home/rwkv/chase/rwkv-skills \
    /home/rwkv/chase/rwkv-skills/ops/g1i_strict46/run_g1h_knowledge_modes.sh \
    "$model" http://127.0.0.1:29574/v1 "$tag"
  wait_remote_unit "$unit"
}

wait_remote_unit "$current_scheduler"
current_model=rwkv7-g1i-13.3b-20260805-ctx16384
if ! remote /home/rwkv/chase/rwkv-skills/ops/g1i_strict46/ensure_model_complete.sh \
  "$current_model" http://127.0.0.1:29574/v1 \
  rwkv-g1i-strict46-133-audit-recovery-20260807 \
  g1i_strict46_133_audit_recovery_20260807 \
  240 60 3; then
  echo "$current_model did not pass the strict 46/46 audit after targeted recovery; keeping the endpoint" >&2
  exit 24
fi
systemctl --user stop "$current_infer"

model133=rwkv7-g1h-13.3b-20260710-ctx10240
infer133=rwkv-g1h-13p3b-gpu2-knowledge-20260806.service
eval133=rwkv-g1h-13p3b-knowledge-modes-20260806.service
deploy "$model133" "$infer133"
run_eval "$model133" "$eval133" g1h_133_knowledge_modes_20260806
systemctl --user stop "$infer133"

model72=rwkv7-g1h-7.2b-20260710-ctx10240
infer72=rwkv-g1h-7p2b-gpu2-knowledge-20260806.service
eval72=rwkv-g1h-7p2b-knowledge-modes-20260806.service
deploy "$model72" "$infer72"
run_eval "$model72" "$eval72" g1h_72_knowledge_modes_20260806
systemctl --user stop "$infer72"
