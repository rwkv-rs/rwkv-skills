#!/usr/bin/env bash
set -euo pipefail

repo=/home/rwkv/chase/rwkv-skills
port=19439
current_scheduler=rwkv-g1i-strict46-15-raw-20260806.service
current_infer=rwkv-g1i-1p5b-gpu3-16k-c640.service

wait_unit() {
  local unit=$1 state
  while true; do
    state=$(systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)
    [[ -n "$state" && "$state" != not-found ]] && break
    sleep 30
  done
  while systemctl --user is-active --quiet "$unit"; do sleep 30; done
  [[ "$(systemctl --user show "$unit" -p Result --value 2>/dev/null || true)" == success ]]
}

deploy() {
  local model=$1 unit=$2
  systemctl --user stop "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --property=Restart=on-failure --property=RestartSec=5s \
    --setenv=CUDA_VISIBLE_DEVICES=3 --setenv=VLLM_USE_V2_MODEL_RUNNER=1 \
    --setenv=VLLM_RWKV7_WKV_MODE=fp32io16 --setenv=VLLM_USE_RAPID_SAMPLER=1 \
    --setenv=VLLM_USE_FLASHINFER_SAMPLER=0 --setenv=PYTHONPATH=/home/rwkv/chase/vllm-rwkv \
    /home/rwkv/chase/.venv-vllm-fcb31d859/bin/vllm serve \
    "$repo/weights/BlinkDL__rwkv7-g1/$model.pth" \
    --host 127.0.0.1 --port "$port" --api-key rwkv-skills \
    --tokenizer-mode rwkv --trust-request-chat-template \
    --enable-auto-tool-choice --tool-call-parser rwkv \
    --max-model-len 10240 --served-model-name "$model" \
    --gpu-memory-utilization 0.97 --max-num-batched-tokens 98304 --max-num-seqs 640 \
    --override-generation-config '{"temperature":1e-5}'
  for _ in $(seq 1 120); do
    curl -fsS --max-time 3 -H 'Authorization: Bearer rwkv-skills' \
      "http://127.0.0.1:$port/v1/models" >/dev/null && return 0
    sleep 3
  done
  return 21
}

run_eval() {
  local model=$1 unit=$2 tag=$3
  systemctl --user stop "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --working-directory="$repo" \
    "$repo/ops/g1i_strict46/run_g1h_knowledge_modes.sh" \
    "$model" "http://127.0.0.1:$port/v1" "$tag"
  wait_unit "$unit"
}

wait_unit "$current_scheduler"
current_model=rwkv7-g1i-1.5b-20260805-ctx16384
if ! "$repo/ops/g1i_strict46/ensure_model_complete.sh" \
  "$current_model" "http://127.0.0.1:$port/v1" \
  rwkv-g1i-strict46-15-audit-recovery-20260807 \
  g1i_strict46_15_audit_recovery_20260807 \
  240 60 3; then
  echo "$current_model did not pass the strict 46/46 audit after targeted recovery; keeping the endpoint" >&2
  exit 24
fi
systemctl --user stop "$current_infer"

model15=rwkv7-g1h-1.5b-20260710-ctx10240
infer15=rwkv-g1h-1p5b-gpu3-knowledge-20260806.service
eval15=rwkv-g1h-1p5b-knowledge-modes-20260806.service
deploy "$model15" "$infer15"
run_eval "$model15" "$eval15" g1h_15_knowledge_modes_20260806
systemctl --user stop "$infer15"

model29=rwkv7-g1h-2.9b-20260710-ctx10240
infer29=rwkv-g1h-2p9b-gpu3-knowledge-20260806.service
eval29=rwkv-g1h-2p9b-knowledge-modes-20260806.service
deploy "$model29" "$infer29"
run_eval "$model29" "$eval29" g1h_29_knowledge_modes_20260806
