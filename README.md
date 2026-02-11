# RWKV Skills

English | [中文](README.zh-CN.md)

An inference & evaluation scaffold for RWKV7, including a continuous-batching inference engine, dataset preppers for common benchmarks, and a GPU scheduler skeleton.

## Quick tour
- `src/infer`: RWKV model loading, sampling strategies, and a continuous-batching generation engine.
- `src/infer/rwkv7`: The upstream RWKV7 reference implementation (CUDA extension, vocab) is vendored in; no extra submodule is required.
- `src/eval/datasets`: Data structures, JSONL loaders, and dataset preparation scripts.
- `src/eval/evaluators`: Evaluation pipelines for multiple-choice / free-response / instruction-following / code generation (HumanEval, MBPP).
- `src/eval/scheduler`: A CLI for queueing evaluation jobs, GPU detection, and dispatching (with entry scripts for multi-choice / free-response / instruction-following / human-eval / mbpp).
- `weights`, `data`, `results` (optional): Default locations for model weights, datasets, and evaluation artifacts.

## Requirements
- Python 3.12+. `uv` is recommended for dependency management.
- NVIDIA/AMD GPU (via dependencies like `triton` + bundled rapid-sampling kernels), with CUDA/ROCm matching your chosen PyTorch build.

## Installation
```bash
# Install dependencies (example: CUDA 12.9, matching the torch-cu129 extra in pyproject)
uv sync --extra torch-cu129

# Editable install to expose CLI entry points
uv pip install -e .
```
For other CUDA/CPU builds, use `--extra torch-cu126` / `--extra torch-cpu`, etc.

## Download model weights
`rwkv-download-weights` enumerates and downloads `.pth` weights concurrently from a Hugging Face mirror:
```bash
rwkv-download-weights /path/to/weights
# or add extra repos:
rwkv-download-weights --repo BlinkDL/rwkv7-g1 --repo your/repo
```
You can override the default endpoint/token via environment variables (`HF_ENDPOINT`, `HF_TOKEN`).

## Dataset preparation
Datasets are stored under `data/` by default. You can call the prepper to generate JSONL files:
```bash
python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

prepare_dataset("mmlu", Path("data"))  # writes data/mmlu/<split>.jsonl
PY
```
To see supported dataset aliases, check the `available_*_datasets()` family of functions.

## Evaluation & inference example
At the moment, the recommended entry point is to call pipeline classes directly:
```python
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig

pipeline = MultipleChoicePipeline(ModelLoadConfig(weights_path="weights/rwkv7-*.pth", device="cuda:0"))
result = pipeline.run_direct(
    dataset_path="data/mmlu/test.jsonl",
    output_path="results/completions/mmlu_direct.jsonl",
    sample_limit=50
)
print(result)
```
Free-response and instruction-following work similarly via `FreeResponsePipeline` and `InstructionFollowingPipeline`.

## Scheduler CLI
`rwkv-skills-scheduler` provides commands for queue preview, dispatch, status, stop, and log rotation:
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --run-log-dir results/logs
```
`queue` is a dry-run of `dispatch` and accepts the same filtering/dispatch flags (including `--overwrite`) to preview what would be scheduled.
By default, jobs that already have scores are skipped.
To force a rerun, pass `--overwrite` on dispatch. This creates a new run/version without deleting historical completion / score / eval records.
By default the evaluator scripts run the LLM wrong-answer checker when configured; to skip it, pass `--disable-checker` on dispatch.

You can re-run only specific benchmarks with `--only-datasets aime24 aime25` (names only; no `_test` suffix), or exclude sets with `--skip-datasets mmlu`. To run only a subset of models, you can filter filenames via `--model-regex '^rwkv7-.*7\\.2b$'` while keeping the default weight glob.

The default model glob is configured in `src/eval/scheduler/config.py` (it only points to `weights/rwkv7-*.pth` within the repo; override as needed). Scheduler entry scripts are provided:
`src/bin/eval_multi_choice.py`, `eval_multi_choice_cot.py`, `eval_free_response.py`, `eval_free_response_judge.py`, `eval_instruction_following.py`, `eval_code_human_eval.py`, `eval_code_mbpp.py`, `eval_code_livecodebench.py`.

Free-response sets that require LLM judging (e.g. `gsm8k_test` / `math_500_test` / `answer_judge_test` / `gaokao2023en_test`) are automatically dispatched to `eval_free_response_judge.py`; other free-response tasks still use `eval_free_response.py`'s exact-match logic.

Sampling-parameter grid search is handled via the param-search workflow:
- Runner jobs write *all* trial artifacts under `results/param_search/{completions,eval,scores}/{model}/{benchmark}/trial_*.{jsonl,json}`.
- The selector job aggregates `results/param_search/scores/...` across `gsm8k_test` + `hendrycks_math_test` (alias: `math`) and promotes one best shared grid point into the unsuffixed `{benchmark}` paths.

When evaluating the latest 2.9B model, the scheduler automatically runs param-search on `gsm8k_test` + `hendrycks_math_test`.

## HumanEval code generation evaluation
- Dataset prep: `prepare_dataset("human_eval", Path("data"))` downloads the official `HumanEval.jsonl.gz` and writes `data/human_eval/test.jsonl`.
- Run via CLI:
  ```bash
  python -m src.bin.eval_code_human_eval \
    --model-path weights/rwkv7-*.pth \
    --dataset data/human_eval/test.jsonl \
    --batch-size 128 \
    --pass-k 1 --pass-k 2 --pass-k 4 --pass-k 8 --pass-k 16 \
    --eval-timeout 3
  ```
  Samples are written under `results/completions`, and the official unit tests are executed automatically to produce pass@k (number of generations equals the maximum k).

## MBPP code generation evaluation
- Dataset prep: `prepare_dataset("mbpp", Path("data"))` uses the EvalPlus variant MBPP+ and converts 4-space indentation in prompts into tabs.
- Run via CLI:
  ```bash
  python -m src.bin.eval_code_mbpp \
    --model-path weights/rwkv7-*.pth \
    --dataset data/mbpp/test.jsonl \
    --batch-size 128 \
    --pass-k 1 --pass-k 2 --pass-k 4 --pass-k 8 --pass-k 16 \
    --eval-timeout 3
  ```
  Multiple samples are generated and executed against EvalPlus test cases to output pass@k (number of generations equals the maximum k).

## LiveCodeBench code generation evaluation
- Dataset prep: `prepare_dataset("livecodebench", Path("data"))` downloads the LiveCodeBench release_v6 (lite) split and writes `data/livecodebench/test.jsonl` (override with `RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG`).
- Run via CLI:
  ```bash
  python -m src.bin.eval_code_livecodebench \
    --model-path weights/rwkv7-*.pth \
    --dataset data/livecodebench/test.jsonl \
    --batch-size 64 \
    --pass-k 1 --pass-k 5 \
    --eval-timeout 6 \
    --eval-workers 12
  ```
  LiveCodeBench uses the extracted code blocks for execution and reports pass@k (number of generations equals the maximum k).

## Known gaps / TODO
- Other code benchmarks (BigCodeBench, etc.) are not supported yet.

Contributions are welcome—please implement missing pieces and update the docs accordingly.

## Migrating historical results
If you have JSON summaries from older versions (`rwkv-mmlu` / `rwkv-skills`) under `results_old/`, you can migrate them into the current `results/scores/` layout to avoid rerunning everything:

```bash
python -m src.bin.migrate_old_results --source results_old
# use --dry-run to preview outputs; use --overwrite to replace existing results
```

## Scheduler-only workflow (DB)
### C.1 One-time setup
1. Prepare PostgreSQL and ensure connectivity (set `.env` / env vars).
   `.env` reference: `.env.example`
2. Prepare model weights:
   `/home/jay/workspace/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1a-0.1b-20250728-ctx4096.pth`
3. Prepare dataset directory: `/home/jay/workspace/rwkv-skills/data` (contains task JSONL files).

### C.2 Queue preview
Purpose: verify all 8 entries are schedulable and dataset paths resolve.
```bash
uv run rwkv-skills-scheduler queue \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot
```

### C.3 Dispatch
Purpose: run all 8 entries and write into the database.
```bash
uv run rwkv-skills-scheduler dispatch \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot \
  --skip-missing-dataset
```

### C.4 Monitor/stop
```bash
rwkv-skills-scheduler status
rwkv-skills-scheduler logs
rwkv-skills-scheduler stop --all
```

### Multi-model resume logic
For each model+dataset(+cot) combination, existing scores are skipped by default. When no score exists, the scheduler resumes the latest unfinished task. If you pass `--overwrite`, the scheduler forces a fresh rerun and writes a new task/version without deleting prior records.
