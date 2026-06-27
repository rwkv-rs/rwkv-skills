# Benchmark Prompt/Parser Alignment Audit - 2026-06-18

## Scope

- Current checkout: `/home/chase/GitHub/rwkv-skills`.
- 2333 project checkout: `/home/caizus/Projects/MachineLearning/chase-rwkv-skills`.
- Current active DB: `chase_rwkv_skills`, `task`/`completions`/`eval` schema.
- Old formal reference DB: `rwkv-skills` on 2333, used only for reference scores and historical prompt/context inspection.
- This audit does not launch remaining formal benchmarks. It stops after alignment checks, data sync readiness, and regression tests.

## Immediate Result

The biggest remaining mismatch class is not a single parser bug. It is a baseline identity problem:

1. IFEval no-cot must be plain `Assistant:` and must enforce `ban_tokens=[295]` in the inference backend. The current code and post-fix smoke path satisfy this.
2. HumanEval family is now intentionally using the requested prompt prefix:

   ````text
   User: You are a top-level code master. Complete the following code without any additional text or explanation:
   {HumanEval prompt}

   Assistant: <think>
   </think>
   ```python
   ````

   The old formal DB did not always use this exact form, so old HumanEval-family scores are reference numbers, not same-prompt validation numbers.
3. `human_eval_cn_nocot`, `human_eval_fix_nocot`, and `human_eval_plus_nocot` are now covered by scheduler/runner regression tests. Local data exists for all three; 2333 had only `human_eval/test.jsonl` before sync.
4. MBPP/MBPP+ and LiveCodeBench source prompt forms match the local old-project implementation. The old formal DB confirms MBPP/MBPP+ no-cot and LCB two-stage behavior.
5. LongBench must parse format artifacts, not loosen scoring. The fenced JSON helper now strips role prefixes before and after fence handling.

## Old Formal Score Reference

Latest old formal 1.5B reference rows inspected from the old DB:

| benchmark | model | task | metric | completions/evals |
| --- | --- | ---: | --- | ---: |
| IFEval | g1f 1.5B | 9703 | avg@8=0.4087338262, instruction_accuracy=0.5184352518 | 4328/4328 |
| IFEval | g1g 1.5B | 16576 | avg@8=0.4269870610, instruction_accuracy=0.5391187050 | 4328/4328 |
| HumanEval | g1f 1.5B | 10570 | avg@32=0.3134527439 | 5248/5248 |
| HumanEval | g1g 1.5B | 16565 | avg@32=0.3601371951 | 5248/5248 |
| HumanEval CN | g1f 1.5B | 10636 | avg@32=0.3115472561 | 5248/5248 |
| HumanEval CN | g1g 1.5B | 16798 | avg@32=0.3578506098 | 5248/5248 |
| HumanEval Fix | g1f 1.5B | 10023 | avg@32=0.2963033537 | 5248/5248 |
| HumanEval Fix | g1g 1.5B | 16806 | avg@32=0.3957698171 | 5248/5248 |
| HumanEval Plus | g1f 1.5B | 10043 | avg@32=0.2427591463 | 5248/5248 |
| HumanEval Plus | g1g 1.5B | 16814 | avg@32=0.3325076220 | 5248/5248 |
| MBPP | g1f 1.5B | 10592 | avg@16=0.5358796296 | 6048/6048 |
| MBPP | g1g 1.5B | 16571 | avg@16=0.5338955026 | 6048/6048 |
| MBPP Plus | g1f 1.5B | 8918 | avg@16=0.4604828042 | 6048/6048 |
| MBPP Plus | g1g 1.5B | 16892 | avg@16=0.4550264550 | 6048/6048 |
| LiveCodeBench | g1f 1.5B | 10615 | avg@4=0.0500000000 | 4220/4220 |
| LiveCodeBench | g1g 1.5B | 16596 | avg@4=0.0632701422 | 4220/4220 |
| LongCodeQA | g1f 1.5B | 19071 | avg@1=0.2844243792 | 443/443 |
| LongCodeQA | g1g 1.5B | 19076 | avg@1=0.3611738149 | 443/443 |

These rows are useful for sanity checks, but prompt identity must be checked before treating them as a target score.

## Prompt/Parser Differences Found

| area | old formal DB evidence | current implementation | status |
| --- | --- | --- | --- |
| IFEval no-cot | `User: ...\n\nAssistant:`; sampling includes `ban_tokens=[295]`; old completions did not start with `<think>` | `build_instruction_following_prompt(..., enable_think=False)` returns `User: ...\n\nAssistant:`; remote backend now forwards and enforces `ban_tokens` | Aligned after backend ban-token fix |
| HumanEval | old formal g1f prompt uses `User:You...` and `A: <think>\n</think>\n```python` | current prompt uses `User: You...` and `Assistant: <think>\n</think>\n```python` | Intentional new baseline per requested prompt; old score not same-prompt |
| HumanEval CN | old formal DB matches HumanEval's `A:` style | current prompt uses the same new `Assistant:` style as HumanEval | Intentional new baseline; included in scheduler and runner tests |
| HumanEval Fix | old formal DB uses `Assistant:<think></think>\n```python` with no space after role | current prompt uses `Assistant: <think>\n</think>\n```python` | Prompt differs; included in scheduler and runner tests |
| HumanEval Plus | old formal DB shows `Assistant:` followed by echoed prompt/code, then completion is only the suffix | current prompt uses new HumanEval-style fenced Python prefix | Prompt differs materially; treat old score as reference only |
| MBPP | old formal DB uses `Assistant:<think></think>\n```python` | current no-cot uses `Assistant: <think></think>\n```python` plus evaluator code extraction | Semantically aligned; whitespace differs only at role prefix |
| MBPP Plus | old formal DB matches MBPP no-cot shape | current no-cot uses MBPP shape with signature support where present | Aligned to current old-project source |
| LiveCodeBench | old formal DB uses two stages: stage 1 starts from `Assistant: <think`, stage 2 appends `</think>\n```python\n` | current pipeline uses the same two-stage structure | Aligned |
| LongBench / LongBench QA | previous bug could leave an opening fenced JSON marker when completion echoed `Assistant:` before JSON | `_strip_longbench_json_fence()` strips role prefix before and after fence handling | Fixed by parser, not by relaxing scoring |

## Coding Variant Coverage

Current scheduler/registry coverage:

- `human_eval_test`
- `human_eval_cn_test`
- `human_eval_fix_test`
- `human_eval_plus_test`

All four map to `code_human_eval`, use `configs/human_eval*.toml`, and default to `avg_k=[32]`.

Regression coverage added:

- `tests/test_runner_registry.py::test_human_eval_job_includes_all_nocot_variants`
- `tests/test_coding_runner.py::test_coding_runner_treats_human_eval_variants_as_human_eval`

Local data counts:

| dataset | rows |
| --- | ---: |
| `data/human_eval/test.jsonl` | 164 |
| `data/human_eval_cn/test.jsonl` | 164 |
| `data/human_eval_fix/test.jsonl` | 164 |
| `data/human_eval_plus/test.jsonl` | 164 |
| `data/mbpp/test.jsonl` | 427 |
| `data/mbpp_plus/test.jsonl` | 378 |
| `data/livecodebench/test.jsonl` | 1055 |

2333 initially only had `data/human_eval/test.jsonl` for the HumanEval family. This audit synced CN/Fix/Plus data to 2333 and verified all four HumanEval-family files have 164 rows.

## Current Invalid Task State

The current active DB has the earlier wrong rerun tasks marked `Failed` and labeled invalid:

- `INVALID_PROMPT_CONFLICT_20260618`
- `INVALID_SUITE_MISMATCH_20260618`
- smoke task `smoke_ifeval_banfix_20260618 ... NOT_FORMAL_DO_NOT_REPORT`

Relevant invalid completed examples:

- IFEval task 226/227/231/232: 4328 completions each, invalid due prior backend/prompt conflict.
- HumanEval task 234/235/238/239: 5248 completions each, invalid due prompt-suite mismatch.
- LongBench QA balanced task 236/237: 1750 completions each, invalid due suite mismatch.

There are two stale DB `Running` rows from an older 8222 MMLU-Pro speed run:

- task 78: tmp, `mmlu_pro`, g1g 2.9B, 0 completions.
- task 79: non-tmp, `mmlu_pro`, g1f 1.5B, 0 completions.

No matching benchmark runner process is active on 2333. These should be cleaned or explicitly ignored before a fresh scheduler run, but this audit did not mutate those rows.

## Verification

Local focused tests:

```text
96 passed, 3 warnings
```

2333 targeted tests after rsync:

```text
15 passed, 1 warning
```

2333 data verification:

```text
164 data/human_eval/test.jsonl
164 data/human_eval_cn/test.jsonl
164 data/human_eval_fix/test.jsonl
164 data/human_eval_plus/test.jsonl
```

Covered files:

- `tests/test_runner_registry.py`
- `tests/test_coding_runner.py`
- `tests/test_benchmark_registry.py`
- `tests/test_benchmark_dataset_utils.py`
- `tests/test_longbench_runner.py`
- `tests/test_code_generation_evaluate.py`
- `tests/test_prompt_builders.py`
- `tests/test_instruction_following_runner.py`
- `tests/test_infer_split.py`

## Stop Point

Do not start the remaining formal benchmark suite from this state. Next step is to update or tune the inference engine performance, then rerun only after:

1. 2333 keeps the synced HumanEval CN/Fix/Plus data files in place.
2. stale `Running` task rows are cleaned or ignored.
3. launch scripts rely on benchmark config files for `avg_k` and benchmark behavior, with scripts limited to performance/runtime settings.
4. invalid task IDs above remain excluded from reporting.
