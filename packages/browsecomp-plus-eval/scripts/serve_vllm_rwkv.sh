#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${VLLM_PYTHON:-${package_root}/.venv-vllm/bin/python}"
model_path="${MODEL_PATH:?Set MODEL_PATH to an RWKV .pth model}"
host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-18073}"
api_key="${VLLM_API_KEY:?Set VLLM_API_KEY for the local server}"
max_model_len="${VLLM_MAX_MODEL_LEN:-10240}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
max_num_seqs="${VLLM_MAX_NUM_SEQS:-1024}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
served_model_name="${SERVED_MODEL_NAME:-$(basename -- "${model_path}" .pth)}"

if [[ ! -x "${python_bin}" ]]; then
  echo "vLLM Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${model_path}" ]]; then
  echo "RWKV model does not exist: ${model_path}" >&2
  exit 2
fi

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export PYTHONPATH="${package_root}/vendor/vllm-rwkv${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -c \
  'import vllm._custom_ops, runpy; runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")' \
  --model "${model_path}" \
  --host "${host}" \
  --port "${port}" \
  --api-key "${api_key}" \
  --tokenizer-mode rwkv \
  --max-model-len "${max_model_len}" \
  --served-model-name "${served_model_name}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-num-seqs "${max_num_seqs}" \
  "$@"
