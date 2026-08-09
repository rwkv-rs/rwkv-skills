#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${package_root}"

env_file="${ENV_FILE:-${package_root}/.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

python_bin="${EVAL_PYTHON:-${package_root}/.venv/bin/python}"
dataset="${DATASET:-browsecomp_plus}"
infer_base_url="${INFER_BASE_URL:?Set INFER_BASE_URL, including /v1}"
infer_model="${INFER_MODEL:?Set INFER_MODEL to the served model name}"
infer_api_key="${INFER_API_KEY:-}"
judge_mode="${BROWSECOMP_PLUS_JUDGE_MODE:-inline}"
prompt_style="${RWKV_FUNCTION_PROMPT_STYLE:-rwkv_flower_json}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Evaluation Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ "${judge_mode}" == "inline" ]]; then
  : "${JUDGE_MODEL:?Set JUDGE_MODEL for inline judging}"
  : "${JUDGE_API_KEY:?Set JUDGE_API_KEY for inline judging}"
fi

args=(
  -m src.eval.tasks.function_calling.runner
  --dataset "${dataset}"
  --benchmark-kind browsecomp_plus
  --task-desc "${TASK_DESC:-BrowseComp-Plus parallel-candidate evaluation}"
  --run-mode "${RUN_MODE:-fresh}"
  --infer-base-url "${infer_base_url}"
  --infer-model "${infer_model}"
  --infer-api-key "${infer_api_key}"
  --infer-timeout-s "${INFER_TIMEOUT_S:-600}"
  --infer-max-workers "${INFER_MAX_WORKERS:-32}"
  --infer-protocol completions
  --infer-seed-policy "${INFER_SEED_POLICY:-omit}"
  --sample-workers "${SAMPLE_WORKERS:-16}"
  --avg-k 1
  --prompt-style "${prompt_style}"
  --tool-call-io rwkv-json
  --history-max-chars "${HISTORY_MAX_CHARS:-24000}"
  --prompt-max-chars "${PROMPT_MAX_CHARS:-28000}"
  --candidate-router-mode parallel
  --candidate-router-chunk-tools "${CANDIDATE_ROUTER_CHUNK_TOOLS:-3}"
  --candidate-router-batch-size "${CANDIDATE_ROUTER_BATCH_SIZE:-16}"
  --candidate-router-context-chars "${CANDIDATE_ROUTER_CONTEXT_CHARS:-8000}"
  --candidate-router-prompt-max-chars "${CANDIDATE_ROUTER_PROMPT_MAX_CHARS:-12288}"
  --candidate-router-candidate-max-tokens "${CANDIDATE_ROUTER_CANDIDATE_MAX_TOKENS:-192}"
  --candidate-router-aggregate-max-tokens "${CANDIDATE_ROUTER_AGGREGATE_MAX_TOKENS:-192}"
  --candidate-router-max-candidates "${CANDIDATE_ROUTER_MAX_CANDIDATES:-3}"
  --candidate-router-tool-schema-mode "${CANDIDATE_ROUTER_TOOL_SCHEMA_MODE:-compact}"
  --candidate-router-evidence-chars "${CANDIDATE_ROUTER_EVIDENCE_CHARS:-1200}"
  --candidate-router-policy-chars "${CANDIDATE_ROUTER_POLICY_CHARS:-2000}"
  --max-steps "${MAX_STEPS:-100}"
  --browsecomp-plus-judge-mode "${judge_mode}"
)

if [[ -n "${JUDGE_MODEL:-}" ]]; then
  args+=(--judge-model "${JUDGE_MODEL}")
fi
if [[ -n "${JUDGE_API_KEY:-}" ]]; then
  args+=(--judge-api-key "${JUDGE_API_KEY}")
fi
if [[ -n "${JUDGE_BASE_URL:-}" ]]; then
  args+=(--judge-base-url "${JUDGE_BASE_URL}")
fi
if [[ -n "${JUDGE_MAX_WORKERS:-}" ]]; then
  args+=(--judge-max-workers "${JUDGE_MAX_WORKERS}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi

export RWKV_BROWSECOMP_PLUS_RETRIEVER="${RWKV_BROWSECOMP_PLUS_RETRIEVER:-bm25}"
exec "${python_bin}" "${args[@]}" "$@"
