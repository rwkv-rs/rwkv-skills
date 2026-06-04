---
name: rwkv-long-doc-evidence-qa
description: Chunk long documents, build evidence QA tasks, run or review brute-force per-chunk answer-or-null extraction, and summarize exact/contain metrics for RWKV-style long-document QA experiments. Use only when explicitly invoked from this file path for long-document QA, context-window workaround experiments, or evidence extraction over chunks; do not use for rwkv-skills benchmark prompt/decoding-parameter iteration, anomaly queue tuning, scheduler jobs, or formal benchmark DB work unless the user explicitly says to combine those workflows.
---

# RWKV Long Doc Evidence QA

Use this skill to test a model on long-document QA without placing the full document in one context window. Keep the first version brute-force over chunks.

## Scope Boundaries

- Require explicit invocation by path, for example `use rwkv-long-doc-evidence-qa/SKILL.md`.
- Do not use this skill for `rwkv-prompt-param-iteration`, `rwkv-result-triage`, formal benchmark promotion, anomaly queue tuning, or scheduler prompt/parameter experiments unless the user explicitly asks to combine workflows.
- Do not turn the first version into embedding RAG, vector search, BM25 retrieval, or reranking infrastructure unless requested after the brute-force baseline is recorded.
- Do not modify unrelated benchmark code, checker code, datasets, `.env`, scheduler config, or prompt/parameter tuning surfaces.
- Do not commit copyrighted source text into git. Chunk JSONL files contain source text; keep them local/untracked unless the user confirms the text is safe to commit.

## Inputs

Expected inputs:

- Source text path for the long document.
- Task-definition JSONL path with one task per line.
- Output directory or explicit output paths.
- Chunk settings: `max_chars` and `overlap_lines`.
- Optional model command or existing per-chunk model output JSONL.

Task-definition JSONL rows must include:

```json
{"id":"task_id","question":"...","answer":"...","answer_format":"scalar_string","positive_rule":{"all":["term1","term2"]}}
```

Allowed `answer_format` values:

- `scalar_string`
- `scalar_number_string`

Allowed `positive_rule` forms:

- `{"all": ["term1", "term2"]}`
- `{"any": ["term1", "term2"]}`
- `{"not": {"all": ["term1"]}}`

## Workflow

1. Prepare the run directory and record paths in `commands.sh` or notes.
2. Normalize newlines in the source text: replace `\r\n` and `\r` with `\n`.
3. Split by original text lines with line numbers preserved.
4. Greedily accumulate lines until adding the next line would exceed `max_chars`.
5. From chunk 1 onward, prepend the previous base chunk's last `overlap_lines` lines.
6. Write chunk JSONL rows with:

```json
{"chunk_id":0,"char_count":1000,"line_start":1,"line_end":20,"overlap_lines":0,"text":"..."}
```

7. Build structured task JSONL by reading the task definitions and recomputing `positive_chunks` from each chunk's `text`.
8. Run oracle validation before any model inference:
   - every task must have at least one positive chunk
   - each gold `answer` string must appear in at least one positive chunk
   - unsupported `positive_rule` forms must fail before model inference
9. Run the optional model stage over every task/chunk pair. Use deterministic decoding: temperature `0`, short max tokens.
10. Ask the model for answer-or-null extraction using only the current chunk:

```text
User: 根据材料回答问题。只有答案被材料明确支持时，输出 JSON {"answer":"..."}。否则输出 JSON {"answer":"null"}。
材料:
<chunk text>
问题: <question>
Assistant:
```

11. Parse model outputs as JSON where possible. Treat invalid JSON, empty output, and missing `answer` as null or as parse failures that are counted separately.
12. Collect non-null candidates per task in deterministic order, usually task order then chunk order.
13. Compute metrics:
   - per-chunk label accuracy: predicted non-null iff the chunk is in `positive_chunks`
   - first-candidate exact: first non-null candidate exactly equals the gold answer after simple normalization
   - first-candidate contain: first non-null candidate contains the gold answer, or the gold answer contains the candidate
   - task-level no-candidate count
   - invalid JSON / parse-failure count

## Outputs

Keep outputs JSONL and summaries JSON. A normal run directory should contain:

- `chunks.jsonl`
- `tasks.jsonl`
- `oracle_summary.json`
- `model_outputs.jsonl` when model inference is run
- `summary.json`
- `commands.sh`
- `notes.md` when decisions or caveats need to be recorded

Do not paste raw copyrighted chunk text into final replies. Summarize file paths, counts, metrics, and failure categories instead.

## Local Three Body Experiment

The files in this directory are corpus-specific reference experiments. Treat them as examples, not as reusable skill content:

- Do not copy the raw source text into skill instructions.
- Do not hard-code the Three Body task list into future generic skill logic unless the user asks for that exact corpus experiment.
- Generalize the procedure to caller-provided source text and task JSONL.
- Keep generated chunk files local and uncommitted unless the user says the source text is commit-safe.
