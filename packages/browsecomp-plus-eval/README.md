# BrowseComp-Plus RWKV evaluation package

This directory is a self-contained source package for reproducing the current
BrowseComp-Plus evaluation path with RWKV-vLLM and the parallel-candidate
router. It is a physical extraction: the evaluator and vendored vLLM sources
are copied into this directory and do not import files from the parent checkout.

The package does not contain credentials, database rows, model weights, the
official 2+ GB corpus, BM25 index, qrels, or generated scores. Creating and
testing the package does not start vLLM, contact an inference endpoint, connect
to PostgreSQL, use a GPU, or mutate any scheduler/runner state.

## Runtime flow

```text
830 official questions
        |
        v
BrowseComp-Plus agent loop -- search(top 5) / get_document / final_answer
        |
        v
3 parallel candidate shards -- evidence/context/tool grounding
        |
        v
candidate aggregation -- one grounded tool call or final answer
        |
        v
OpenAI-compatible /v1/completions -- vendored RWKV-vLLM
        |
        v
PostgreSQL completion/audit rows -- export 830 per-query official run files
        |
        v
internal judge and/or upstream official judge -- accuracy + retrieval Recall
```

Important implementation entry points:

- `src/eval/tasks/function_calling/browsecomp_plus.py`: dataset adapter,
  BM25 tools, prompt budgeting, agent loop, evidence persistence, final-answer
  conversion, judge modes, and score payload.
- `src/eval/experiments/parallel_candidate_router/router.py`: candidate shard
  generation, parsing, grounding, deduplication, and aggregation.
- `src/eval/tasks/function_calling/runner.py`: unified CLI and DB lifecycle.
- `src/infer/backend.py`: OpenAI-compatible completion client.
- `vendor/vllm-rwkv/`: exact vendored inference runtime source snapshot.

## 1. Install

Python 3.12, PostgreSQL, a CUDA environment compatible with the selected vLLM
build, and Java/pyserini for the BM25 index are required.

Create the evaluation environment:

```bash
uv sync --extra dev --extra function-calling-official
cp .env.example .env
```

Keep vLLM in a separate environment because its CUDA/PyTorch dependency set is
usually stricter than the evaluation harness:

```bash
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python -e vendor/vllm-rwkv
```

The launch wrapper also prepends the vendored tree to `PYTHONPATH`, ensuring the
copied implementation is selected. It sets `VLLM_USE_V2_MODEL_RUNNER=1`, which
is required by this RWKV7 runtime path.

## 2. Prepare official assets

Obtain the official `texttron/BrowseComp-Plus` assets and point
`BROWSECOMP_PLUS_ROOT` at them. The expected layout is documented in
`data/README.md`.

Run the offline preflight. Full validation reads the large decrypted JSONL and
requires exactly 830 valid, unique query IDs:

```bash
.venv/bin/python scripts/preflight_browsecomp_plus.py \
  --root "$BROWSECOMP_PLUS_ROOT"
```

The evaluator can auto-prepare `data/browsecomp_plus/test.jsonl` from that
official root when `DATASET=browsecomp_plus`. To prepare it explicitly:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

print(prepare_dataset("browsecomp_plus", Path("data"), "test"))
PY
```

Do not set `RWKV_BROWSECOMP_PLUS_EMBED_DOCS=1` for the BM25 run: documents
should remain in the official index rather than being duplicated into every
manifest row.

## 3. Start the copied RWKV-vLLM runtime

This is an explicit user action; no server is started automatically:

```bash
MODEL_PATH=/path/to/model.pth \
SERVED_MODEL_NAME=rwkv7-model \
VLLM_API_KEY=change-me \
scripts/serve_vllm_rwkv.sh
```

Defaults mirror the evaluated setup: localhost port 18073, RWKV tokenizer,
10,240-token context, 32,768 batched tokens, 1,024 sequences, and 0.97 GPU
memory utilization. Override any `VLLM_*` variable when the hardware differs.

Verify both endpoint discovery and one tiny completion before spending GPU time
on 830 episodes:

```bash
.venv/bin/python scripts/check_endpoint.py \
  --base-url http://127.0.0.1:18073/v1 \
  --model rwkv7-model \
  --api-key change-me
```

## 4. Run all 830 questions

Use a dedicated database and fill `.env`. The DB schema is managed by the
included harness. The run wrapper defaults to a fresh task, BM25 retrieval,
avg@1, 16 concurrent episodes, and no sample limit:

```bash
INFER_BASE_URL=http://127.0.0.1:18073/v1 \
INFER_MODEL=rwkv7-model \
INFER_API_KEY=change-me \
BROWSECOMP_PLUS_ROOT=/path/to/BrowseComp-Plus \
scripts/run_browsecomp_plus.sh
```

The captured full-run profile is:

- history budget 24,000 characters; total prompt budget 28,000;
- 100 agent steps;
- parallel candidate mode with three shards/candidates;
- 16-candidate batch, 8,000 context characters, 12,288 prompt characters;
- 192 requested tokens for each candidate and aggregation pass (the Browse
  adapter may raise the effective research minimum where required);
- OpenAI-compatible raw `/v1/completions`, omitted per-prompt seeds;
- `rwkv_flower_json` prompt serialization and local `rwkv-json` tool-call IO.

Use `MAX_SAMPLES` only for a probe. A formal result must have all 830 unique
attempt keys and sample indices `0..829`.

### Judge modes

- `inline` (default): judge each completion and require `JUDGE_MODEL`,
  `JUDGE_API_KEY`, and normally `JUDGE_BASE_URL`.
- `defer`: finish and persist generation first, leaving judge-pending rows.
- `judge`: score an existing deferred task via the runner's
  `--browsecomp-plus-judge-task-id` option.

A generation-stage score with judge-pending rows is temporary and must not be
reported as the final BrowseComp-Plus result.

## 5. Official export and evaluation

Export persisted completion rows to the upstream one-JSON-file-per-query shape:

```bash
.venv/bin/python scripts/oneoff/export_browsecomp_plus_task_for_official_eval.py \
  --task-id TASK_ID \
  --output-dir official-runs/TASK_ID \
  --expected-count 830
```

The exporter fails on missing/truncated audit fields, duplicate query IDs, or a
count other than 830. Then run the evaluator shipped by the official asset
checkout (it loads its own judge model through vLLM):

```bash
BROWSECOMP_PLUS_ROOT=/path/to/BrowseComp-Plus \
OFFICIAL_RUN_DIR=official-runs/TASK_ID \
OFFICIAL_JUDGE_MODEL=Qwen/Qwen3-32B \
scripts/run_official_evaluator.sh
```

The official assets and evaluator are deliberately consumed from the upstream
checkout rather than silently modified in this package.

## Metrics and audit fields

- `avg@1` / `success_rate`: judged-correct final answers divided by all 830
  questions. Incomplete and unparsable results count as failures.
- `Recall`: for each query with evidence qrels,
  `|retrieved_docids intersect relevant_docids| / |relevant_docids|`, then the
  macro average across eligible queries. It measures retrieval coverage, not
  answer correctness.
- confidence/calibration: the model's final confidence compared with judged
  correctness; it is diagnostic and is not accuracy.
- completion integrity: 830 unique query/attempt keys, no missing sample index,
  and no `judge_pending` rows for a final internal score.

Every completion should retain `browsecomp_plus_run`, retrieved document IDs,
read evidence, agent trace, candidate-router traces, and final candidate
conversion. `scripts/maintenance/restore_browsecomp_plus_audit_fields.py` is a
task-scoped, dry-run-by-default repair helper for historical DB round-trip loss;
it must never be used to synthesize answers or change a score.

## Prompt and conversion format

The endpoint is OpenAI-compatible, but this evaluated path does not rely on
native OpenAI `tool_calls`. The model receives the RWKV flower-formatted
System/User history and emits JSON tool decisions. Parallel candidates see
bounded evidence shards; the aggregator selects a grounded call. A
`final_answer` JSON object is validated against retrieved evidence and converted
to the official output-text structure containing the answer and confidence.

This conversion is part of the measured agent pipeline. It does not inject an
answer, query a hidden source, or bypass retrieval.

## Validation and publication

Run focused validation from this directory:

```bash
.venv/bin/python -m pytest -q tests
bash -n scripts/*.sh
.venv/bin/python -m src.eval.tasks.function_calling.runner --help
```

Before public upload, read `THIRD_PARTY_NOTICES.md`, confirm the harness license,
and check that `.env`, models, datasets, indexes, results, and DB dumps are not
staged. No file in the prepared package should exceed GitHub's 100 MB per-file
limit; the vendored vLLM tree is many small source files.
