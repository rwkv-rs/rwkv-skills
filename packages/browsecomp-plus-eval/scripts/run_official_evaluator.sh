#!/usr/bin/env bash
set -euo pipefail

official_root="${BROWSECOMP_PLUS_ROOT:?Set BROWSECOMP_PLUS_ROOT to the official checkout}"
input_dir="${OFFICIAL_RUN_DIR:?Set OFFICIAL_RUN_DIR to exported per-query JSON files}"
eval_dir="${OFFICIAL_EVAL_DIR:-./evals}"
judge_model="${OFFICIAL_JUDGE_MODEL:-Qwen/Qwen3-32B}"
python_bin="${OFFICIAL_EVAL_PYTHON:-python3}"

exec "${python_bin}" "${official_root}/scripts_evaluation/evaluate_run.py" \
  --input_dir "${input_dir}" \
  --ground_truth "${official_root}/data/browsecomp_plus_decrypted.jsonl" \
  --qrel_evidence "${official_root}/topics-qrels/qrel_evidence.txt" \
  --eval_dir "${eval_dir}" \
  --model "${judge_model}" \
  --batch_size "${OFFICIAL_JUDGE_BATCH_SIZE:-64}" \
  --tensor_parallel_size "${OFFICIAL_JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
  "$@"
