# G1h Codex Handoff (2026-07-19 23:30 CST)

This is the short authoritative entrypoint for the next Codex window. Read it first,
then read the detailed documents linked below. Do not infer current state from old chat.

## Read First

1. `docs/runbooks/g1h-codex-handoff-2026-07-19.md` (this file)
2. `docs/runbooks/g1h-benchmark-anomalies-and-retests-2026-07-19.md`
3. `docs/runbooks/g1h-eval-handoff-2026-07-14.md`
4. `configs/g1h/README.md`
5. `configs/scheduler/g1h-normal-7gpu.toml`

The anomaly document is the authoritative retest queue. The older handoff contains DB,
BrowseComp-Plus, score, output-format, inference-engine, and historical runtime evidence.

## Hard Requirements

- Evaluate all four G1h models: 1.5B, 2.9B, 7.2B, and 13.3B.
- Use vLLM endpoints for inference. `normal` and `native` are evaluation-side modes;
  prioritize only `normal`. Do not create a separate "normal/vLLM" mode.
- Preserve every benchmark's TOML prompt, split, sample count, scorer, judge policy,
  generation budget, and registered `avg@k`/`pass@k`. Never reduce `k` for speed.
- No checker. Keep a judge only where the registered benchmark requires it, and raise
  judge concurrency when the provider remains healthy.
- Non-FC priority order: math first, then all Knowledge (including all newly added
  datasets and every historical invalid/abnormal CoT result), then coding and instruction
  following. FC is last for all four models, not only 7.2B.
- Use one physical inference endpoint per active runner. Different sampling stages or
  benchmarks must not share one endpoint concurrently; the f476 rapid sampler crashes on
  mixed scalar sampling parameters.
- Inspect completions, stop reasons, truncation, score curves, and model-size monotonicity.
  A completed task is not accepted merely because a score row exists.
- Report in person every 30 minutes: Completed/Failed/Running, score total, new scores,
  remaining combinations, active tasks, endpoint health, GPU memory/utilization/power,
  suspicious results, and fixes. A monitoring script is not the report.
- Preserve dirty BrowseComp-Plus work in both local and 157 checkouts. Do not broadly
  clean `/home/rwkv/chase`, `.venv`, weights, data, logs, or unrelated user processes.

## Current Code and Validation

Local workspace: `/home/chase/GitHub/rwkv-skills`, branch `benchmark/mmlu-prox`, base
commit `7418395`. The worktree is intentionally very dirty, especially vendored vLLM and
Browse/FC files. Do not reset or replace it.

157 formal checkout: `/home/rwkv/chase/rwkv-skills-g1h-normal-20260719`, also based on
`7418395`. It has the tested G1h configs, Knowledge cascade, streaming completion stop,
and endpoint-affinity changes synced from local. It uses the shared main checkout's
`.venv`, data, weights, references, and logs through symlinks.

Relevant modified implementation:

- `src/infer/backend.py`: per-prompt SSE completion streaming and text answer detector.
- `src/eval/tasks/knowledge/pipeline.py`: Knowledge strategy A/B cascade.
- `src/eval/tasks/knowledge/runner.py`: grouped A/B scoring and formal cumulative B.
- `src/eval/metrics/multi_choice.py`: earliest formal answer after first `</think>`.
- `src/eval/metrics/at_k.py`: fractional GPQA score aggregation.
- `src/eval/benchmark_config.py`: Knowledge strategy and missing-score config.
- `src/eval/scheduler/{action_dispatch,launch_config,remote_slots}.py`: endpoint affinity.
- `configs/g1h/`: complete G1h TOMLs; all 116 Knowledge/model resolutions audited.

Validation completed:

- Local focused+scheduler tests: `85 passed, 2 warnings`.
- 157 synced focused tests: `71 passed, 2 warnings`.
- All 29 Knowledge benchmarks x four models resolve `cascade_a_b`, official
  `Bot✿<think`, and `stop_tokens=(0,)`.
- No-DB 7.2B GPQA-Diamond probe, 16 questions x four rollouts: strategy A exact/adjusted
  `25.00%/39.84%`; cumulative B `48.44%/58.20%`; 48 rerouted and 15 rescued; final valid
  answer rate `95.31%`.

Math keeps its separate A/B/C policy. NoCoT remains a direct answer and must not be forced
to emit reasoning.

## Dataset State and Current Matrix

The 18 previously missing new Knowledge datasets were materialized with their pinned repo
preppers and stored under shared `/home/rwkv/chase/rwkv-skills/data`:

`agieval_mcq`, `arabicmmlu`, `arc_challenge`, `arc_easy`, `bbh_mcq`, `commonsense_qa`,
`hellaswag`, `kmmlu`, `medmcqa`, `medqa`, `mmlu_cf`, `mmlu_prox`, all three `mmlu_sr`
variants, `openbookqa`, `truthfulqa_mc1`, and `winogrande`.

Validation: 18 JSONL files, 475,482 rows, 510,991,346 bytes, exact expected row counts,
valid JSON, and canonical non-empty question/answer fields. Local temporary copies were
deleted after rsync.

Current normal non-FC runnable matrix per model:

- math: 37 CoT
- Knowledge: 29 NoCoT + 29 CoT = 58
- coding: 6 NoCoT + 1 CoT = 7
- instruction following: 2 NoCoT
- total: 104 per model, 416 across four G1h models

`hle` and `hy_math` still lack source data. Seven SWE-bench variants require their official
harness and are not included in 416. The final audit must either integrate these registered
items or record a specific source/harness blocker; never silently count them as completed.

The new post-repair full rerun is `0/416`. The historical format-final matrix had
`141/272` acceptable scores and 131 missing/failed combinations, but those scores are only
comparison evidence. They do not count as completion of the newly required full rerun.
Tasks 26440-26446 are seven failed AIME launch attempts from the old mixed-sampling setup;
they wrote no scores and must be rerun fresh.

## Inference Endpoints

All endpoints passed authenticated `/v1/models` and `/v1/completions` from 157.
API key: `rwkv-skills`. Model context: 10,240.

| Use | Model | 157 endpoint | Physical placement |
| --- | --- | --- | --- |
| formal | G1h-1.5B | `http://127.0.0.1:19315/v1` | 157 GPU0 |
| formal | G1h-1.5B | `http://127.0.0.1:19316/v1` | 157 GPU1 |
| formal | G1h-2.9B | `http://127.0.0.1:19329/v1` | 157 GPU2 |
| formal | G1h-2.9B | `http://127.0.0.1:19330/v1` | 157 GPU3 |
| formal | G1h-7.2B | `http://127.0.0.1:29572/v1` | 8222 GPU0, port 18072 |
| formal | G1h-13.3B | `http://127.0.0.1:29533/v1` | 8222 GPU1, port 18133 |
| formal | G1h-13.3B | `http://127.0.0.1:29534/v1` | 8222 GPU2, port 18134 |
| Browse experiment | G1h-7.2B | `http://127.0.0.1:29573/v1` | 8222 GPU3, port 18073 |

The 8222 `topology_latent_transition_probe.py` process ended before cleanup; GPU3 was then
assigned to the 7.2B experiment. Current idle model memory is about 12.3/12.3/17.8/19.9 GB
on the four 157 GPUs and 53.1/60.0/60.0/41.1 GB on the four 8222 GPUs. Low idle power is
expected; report loaded utilization and power after runners start.

vLLM source used by the live endpoints:

- 157: `/home/rwkv/chase/vllm-rwkv-f476ab83f`
- 8222: `/home/chase/vllm-rwkv-f476ab83f`
- reported version: `0.23.1rc1.dev1227+gf476ab83f`
- required env: `VLLM_USE_V2_MODEL_RUNNER=1`

Do not redeploy these endpoints unless health or completion probes fail.

## Required Execution Order

1. Re-read this file and the anomaly queue; verify the eight endpoints and DB snapshot.
2. Audit the scheduler's broad benchmark alias lookup before trusting skip decisions.
3. Build the explicit rerun queue: all 416 normal non-FC combinations, with math and all
   Knowledge forced fresh, plus every anomaly and interrupted task in the anomaly document.
4. Launch seven endpoint-affine formal lanes. Keep each physical endpoint bound to one
   runner and preserve benchmark-specific TOMLs. Monitor failures and scores continuously.
5. Once seven formal lanes are stable, resume BrowseComp-Plus only on experimental
   `29573`, using G1h-7.2B. Preserve the official 100-step upper bound, no checker, official
   judge, parallel-candidate conversion, and document/chunk evidence work described in the
   detailed handoff. Inspect completions; do not accept a low score without root cause.
6. When BrowseComp-Plus produces a truthful full score and passes trace/evidence audit,
   stop treating GPU3 as experimental and use it as an eighth formal lane where useful.
7. Finish all normal non-FC combinations, then rerun every registered FC benchmark for all
   four G1h models. FC uses vLLM inference but evaluation-side tool prompts, JSON conversion,
   parallel-candidate logic, judges, and benchmark-specific `k` remain authoritative.
8. Re-run failures and suspicious curves until every runnable combination has a checked
   score. Compare against old G1h and contract-matched G1g rows; inspect completions before
   attributing regressions to model capability.
9. Finalize DB/task/score counts, task IDs, score IDs, deltas, truncation diagnostics,
   endpoint configuration, and remaining source-required blockers in both detailed docs.
10. After the user accepts the final audit, stop runner/judge processes and release the
    eight G1h vLLM endpoints and tunnels. Record final `screen`, ports, and `nvidia-smi`
    before and after release. Do not release them before acceptance.

## Reporting Format

Use this exact shape every 30 minutes:

```text
Half-hour report:

Current state:
- Completed N
- Failed N
- Running N
- score_total = N
- Formal matrix progress = N/416 (recompute if registered runnable scope changes)
- Root disk usage and free space
- Runner screens and endpoint screens alive

New scores:
- model benchmark: metric=value; old comparable value/delta; completion audit result

Problems/actions:
- new failures, suspicious scores, truncation/format issues, exact fixes

Active tasks:
- model: benchmark/mode/task_id/completions progress

GPU:
- per GPU memory, utilization, power draw/limit; explain idle or under-utilized cards

Next:
- concrete work before the next report
```

## Live score-audit protocol (2026-07-20)

The seven formal runners remain active while every new score is audited from DB rows,
raw completions, scorer decisions, finish/truncation fields, and sampled correct/wrong
answers. The procedure, report format, BrowseComp-Plus trace checks, and the gate for
turning 29573 into an eighth formal lane are recorded in
`docs/runbooks/g1h-live-monitoring-and-score-audit-2026-07-20.md`.

At 00:41 CST, formal tasks 26447--26453 and Browse task 26454 were running; Browse had
25/830 completions and no trusted score. GPU3 was active through 29573, so it was not
released or left idle.

At 01:05 CST, the fresh formal count is 2/416: scores 10004 (AIME24, G1h-1.5B)
and 10005 (AIME25, G1h-1.5B) have completion-level audits recorded in the live
audit document. The five remaining initial formal Math lanes plus the active
Knowledge lanes remain running. The old-code Browse diagnostic 26454 is excluded;
r26 task 26461 is running on 29573 and has not yet committed its first batch.
