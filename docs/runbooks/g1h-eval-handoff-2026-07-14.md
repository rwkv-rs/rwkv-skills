# g1h Eval Handoff - 2026-07-15

Purpose: compact handoff for continuing the active g1h evaluation work. Do not print secrets from `.env`.

## Hard Boundaries

- Only use 157 through `rwkv-8222`; do not use 2333 for this work.
- 157 repo: `/home/rwkv/chase/rwkv-skills`
- 157 DB is read from repo `.env`: `PG_HOST=127.0.0.1`, `PG_PORT=5432`, `PG_USER=postgres`, `PG_DBNAME=chase_rwkv_skills`.
- Do not stop other users' processes. Only stop screens/processes launched for this g1h run.
- BrowseComp-Plus must be full 830 with official judge; do not replace it with random 500.
- Local shell commands should be prefixed with `rtk`.

## Current Runs

Refreshed: `2026-07-15 14:03 CST`.

### 09:35 CST Current Truth

The earlier `29572` / `29533` / `19329` queues were not truly healthy.

- `127.0.0.1:29572` (`g1h-7.2b`) and `127.0.0.1:29533` (`g1h-13.3b`) reset connections. 157 has no visible local 7.2/13.3 GPU process.
- `127.0.0.1:19329` and `127.0.0.1:19315` vLLM small-model servers shut down with EngineCore errors. Do not schedule new evals on these ports.
- Healthy active endpoints are the lightning proxies:
  - `g1h-1.5b`: `http://127.0.0.1:29315/v1`
  - `g1h-2.9b`: `http://127.0.0.1:29329/v1`
  - older `g1h-2.9b`: `http://127.0.0.1:29298/v1` is still serving the long `polymath` task.
- NAS files for `g1h-7.2b` and `g1h-13.3b` under `/mnt/nas/rwkv-weights/BlinkDL__rwkv7-g1/` are sparse incomplete aria2 files (`du` only about `189M` / `212M`). Do not use them as model weights until resumed and verified.

Stopped to prevent fake progress / 0-completion failure spam:

- `g1h72_nonfc_fill2_20260714_2122_primaryonly_0313`
- `g1h72_nonfc_fill2_20260714_2122_coding_after_0815`
- `g1h72_low_fc_after_current_20260714_2132`
- `g1h_72_133_core_all_20260712_195641_133_math`
- `g1h_other_domains_noceval_29_133_20260713_151234_133`
- `g1h_other_domains_noceval_29_133_20260713_151234_29`

DB rows manually changed from orphan `Running` to `Failed` because the owning process was gone and they had `0` completions / `0` scores:

- `25438` `g1h-13.3b mmlu_redux CoT`
- `25447` `g1h-2.9b supergpqa CoT`
- `25458` `g1h-13.3b minerva_math`
- `25459` `g1h-7.2b hmmt_feb25`
- `25460` `g1h-7.2b human_eval NoCoT`

Latest confirmed completions after 09:00:

- `25427` `g1h-1.5b livecodebench`: restored from eval rows, score_id `9393`, `avg@4=0.07890995260663507`; then child PID `1678384` was TERM'd so the matrix parent could finish.
- `25453` `g1h-2.9b superchem`: completed normally, score_id `9392`, `avg@1=0.07662835249042145`.

Current true live rows at 09:34 CST:

| model | task | progress |
| --- | --- | --- |
| `g1h-1.5b` | `25329 polymath` | `24384/72000`, log active on `29315` |
| `g1h-2.9b` | `25422 polymath` | `9092/72000`, log active on `29298` |
| `g1h-2.9b` | `25463 svamp` | started on healthy `29329` |

### 09:47 CST Endpoint Recovery

Recovered the `g1h-7.2b` / `g1h-13.3b` external infer endpoints.

- 8222 host: `rwkv-260304`, user `chase`.
- Existing tunnel screens on 8222 were still alive:
  - `tunnel_157_g1h72_29572`: `-R 127.0.0.1:29572:127.0.0.1:18072`
  - `tunnel_157_g1h133_29533`: `-R 127.0.0.1:29533:127.0.0.1:18133`
- Full weights exist on 8222:
  - `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1h-7.2b-20260710-ctx10240.pth` (`14G`)
  - `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1h-13.3b-20260710-ctx10240.pth` (`25G`)
- Root cause of the 09:20-09:24 EngineCore deaths was:
  `RuntimeError: rapid-sampling does not support greedy requests. Set VLLM_USE_RAPID_SAMPLER=0 to use the native greedy path.`
- Restarted both services on 8222 with `VLLM_USE_RAPID_SAMPLER=0`:
  - `881133.g1h72_18072_restart_native_sampler_20260715_0945`
  - `881136.g1h133_18133_restart_native_sampler_20260715_0945`
- Verified from 157:
  - `http://127.0.0.1:29572/v1/completions` returns HTTP 200 for `temperature=0` and `0.001`.
  - `http://127.0.0.1:29533/v1/completions` returns HTTP 200 for `temperature=0` and `0.001`.

Relaunched recovery queues:

- `2787141.g1h72_nonfc_recover_20260715_0947`
  - fields: `maths`, `knowledge`, `coding`
  - endpoint: `http://127.0.0.1:29572`
  - env: `RWKV_MATH_PRIMARY_ONLY=1`, `RWKV_MATH_FAST_INTEGER_MATCH=1`, `RWKV_SKILLS_DISABLE_CHECKER=1`
  - `25439 gsm_plus` resumed at `8640/9204` and completed at 09:52, score_id `9394`, `avg@4=0.7088222511951325`.
  - current next task at 09:53: `25457 hendrycks_math`.
- `2787146.g1h133_nonfc_recover_20260715_0947`
  - fields: `maths`, `knowledge`, `coding`
  - endpoint: `http://127.0.0.1:29533/v1`
  - env: `RWKV_MATH_PRIMARY_ONLY=1`, `RWKV_MATH_FAST_INTEGER_MATCH=1`, `RWKV_SKILLS_DISABLE_CHECKER=1`
  - current first task: `25088 beyond_aime`, resumed at `5568/6400`.
- `2801465.g1h72_low_fc_after_recover_20260715_0949`
  - waits for `g1h72_nonfc_recover_20260715_0947` screen to disappear, then reruns the low-score FC list on 7.2 with `--run-mode fresh --sample-cap 0`.

### 10:37 CST Live Audit

Use the active recovery screens above, not the stopped `g1h72_nonfc_fill2_*` / `g1h72_low_fc_after_current_*` screens.

Current true live rows at `2026-07-15 14:02 CST`:

| model | task | progress |
| --- | --- | --- |
| `g1h-13.3b` | `25200 cl_bench` | generation full `1899/1899`; now `LLM judging`, log reached about `332/1801` |
| `g1h-1.5b` | `25467 polymath` | `avg@1`, `3360/9000`, log active on `29315` |
| `g1h-2.9b` | `25468 polymath` | `avg@1`, `3200/9000`, log active on `29298` |
| `g1h-2.9b` | `25167 ceval` | CoT log reached `9856/12342`; DB completions not yet flushed |
| `g1h-7.2b` | `25474 math_odyssey` | `avg@16`, `1806/6192`, log active on `29572` |

Core registry dry-run at `2026-07-15 13:31 CST`; no new score by `2026-07-15 14:02 CST`, so remaining is unchanged:

| model | scored | running | missing | remaining |
| --- | ---: | ---: | ---: | ---: |
| `g1h-1.5b` | 65 | 1 | 0 | 1 |
| `g1h-2.9b` | 54 | 2 | 10 | 12 |
| `g1h-7.2b` | 25 | 1 | 40 | 41 |
| `g1h-13.3b` | 35 | 1 | 30 | 31 |

The active 2.9 / 7.2 / 13.3 recovery screens all started `run_true_g1h_core_matrix.py` with `runs=66`, so they cover maths, knowledge, and coding. They will skip already-scored/running pairs and proceed through the missing items above. The 7.2 low-FC retest screen is intentionally waiting until `g1h72_nonfc_recover_20260715_0947` exits.

Polymath correction at `2026-07-15 11:39 CST`: old tasks `25329` and `25422` were wrong because `configs/polymath.toml` still used `avg_k=[8]`, creating `9000 * 8 = 72000` effective attempts. Those two tasks were marked `Failed` after their screens were stopped. `configs/polymath.toml` now uses `avg_k=[1]` / `report_avg_k=[1]`, and replacement tasks are:

| model | task | expected |
| --- | --- | --- |
| `g1h-1.5b` | `25467 polymath` | `avg@1`, `9000` attempts |
| `g1h-2.9b` | `25468 polymath` | `avg@1`, `9000` attempts |

### 7.2 Recovery Scores

- Note: `comp_math_24_25` uses `free_response_judge`, so primary-only still requires the primary LLM judge. A `Timeout during comparison` appeared in the generation log, but DB completions reached full `4096/4096` and score `9375` was written successfully; do not stop future free-response-judge tasks for that timeout alone while they continue progressing.

Completed in this recovery line:

| task_id | benchmark | score_id | primary metric |
| --- | --- | --- | --- |
| 25089 | beyond_aime | 9355 | `avg@64=0.05875` |
| 25392 | brumo25 | 9357 | `avg@64=0.18072916666666666` |
| 25398 | cl_bench | 9363 | `avg@1=0.08162190626645603` |
| 25410 | cl_bench_life | 9364 | `avg@1=0.12345679012345678` |
| 25413 | cmt_benchmark | 9365 | `avg@1=0.22` |
| 25416 | college_math | 9370 | `avg@2=0.4673527324343506` |
| 25425 | comp_math_24_25 | 9375 | `avg@16=0.095947265625` |
| 25429 | frontierscience_olympiad | 9376 | `avg@1=0.03` |
| 25430 | frontierscience_research | 9377 | `avg@1=0.05` |
| 25431 | gaokao2023en | 9380 | `avg@16=0.6918831168831169` |
| 25436 | gsm8k | 9383 | `avg@4=0.8633434420015162` |
| 25439 | gsm_plus | 9394 | `avg@4=0.7088222511951325` |
| 25457 | hendrycks_math | 9396 | `avg@1=0.7246` |
| 25459 | hmmt_feb25 | 9399 | `avg@64=0.058854166666666666` |
| 25471 | horizonmath | 9400 | `avg@1=0.0` |
| 25472 | imoanswerbench | 9401 | `avg@1=0.06` |
| 25473 | math_500 | 9403 | `avg@8=0.7265` |

Important fix: `src/eval/tasks/maths/runner.py` now makes the optional wrong-answer checker opt-in via `--run-checker` or `RWKV_MATH_RUN_CHECKER=1`. This prevents exact/LLM scoring from hanging before `record_score`. Validated locally and on 157 with `tests/test_maths_runner.py` (`5 passed`).

Additional scoring fix: `src/eval/tasks/maths/runner.py` now supports `--primary-only` / `RWKV_MATH_PRIMARY_ONLY=1`. Use this for g1h fill runs so math tasks write the primary benchmark score without waiting for slow B/C strategy diagnostics. Validated locally and on 157 with `tests/test_maths_runner.py` (`5 passed`). `college_math` hit full `5636/5636` completions, then spent too long in all-strategy math comparison; its screen was stopped and the score was restored with:

```bash
RWKV_MATH_FAST_INTEGER_MATCH=1 \
.venv/bin/python scripts/maintenance/restore_math_scores_from_completions.py \
  --task-id 25416 --primary-only --execute
```

If a math task reaches full completions/evals but no score and the log stops, first verify whether it is the same recoverable score-tail case, then use:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /home/rwkv/chase/rwkv-skills &&
   .venv/bin/python scripts/maintenance/restore_math_scores_from_completions.py \
     --task-id <task_id> --primary-only --execute'
```

### All g1h Model Monitoring

All current release g1h models are being monitored, not only `g1h_7.2`.

Current live Running rows at `2026-07-15 14:02 CST`:

| model | task | progress |
| --- | --- | --- |
| `g1h-13.3b` | `25200 cl_bench` | generation full `1899/1899`; now `LLM judging`, log reached about `332/1801` |
| `g1h-1.5b` | `25467 polymath` | `avg@1`, `3360/9000`, log active |
| `g1h-2.9b` | `25468 polymath` | `avg@1`, `3200/9000`, log active |
| `g1h-2.9b` | `25167 ceval` | CoT log reached `9856/12342`; DB completions not yet flushed |
| `g1h-7.2b` | `25474 math_odyssey` | `avg@16`, `1806/6192`, log active |

Recently completed/recovered:

| task_id | model | benchmark | score_id | metric |
| --- | --- | --- | --- | --- |
| 25338 | `g1h-2.9b` | `mmmlu` | 9384 | `avg@0.2=0.5872523335961544` |
| 25426 | `g1h-1.5b` | `gpqa_extended` | 9385 | `avg@16=0.28136446886446886` |
| 25437 | `g1h-1.5b` | `mbpp_plus` | 9386 | `avg@16=0.46908068783068785` |
| 25434 | `g1h-13.3b` | `hmmt_feb25` | 9387 | `avg@64=0.15625` |
| 25443 | `g1h-13.3b` | `horizonmath` | 9388 | `avg@1=0.0` |
| 25440 | `g1h-2.9b` | `supergpqa no_cot` | 9389 | `avg@1=0.21998567605262165` |
| 25446 | `g1h-13.3b` | `imoanswerbench` | 9390 | `avg@1=0.13` |
| 25424 | `g1h-2.9b` | `simpleqa` | 9391 | `avg@8=0.019` |
| 25453 | `g1h-2.9b` | `superchem` | 9392 | `avg@1=0.07662835249042145` |
| 25427 | `g1h-1.5b` | `livecodebench` | 9393 | `avg@4=0.07890995260663507` |
| 25439 | `g1h-7.2b` | `gsm_plus` | 9394 | `avg@4=0.7088222511951325` |
| 25088 | `g1h-13.3b` | `beyond_aime` | 9395 | `avg@64=0.1546875` |
| 25457 | `g1h-7.2b` | `hendrycks_math` | 9396 | `avg@1=0.7246` |
| 25463 | `g1h-2.9b` | `svamp` | 9397 | `avg@4=0.84925` |
| 25466 | `g1h-2.9b` | `usamo_2026` | 9398 | `avg@1=0.0` |
| 25459 | `g1h-7.2b` | `hmmt_feb25` | 9399 | `avg@64=0.058854166666666666` |
| 25471 | `g1h-7.2b` | `horizonmath` | 9400 | `avg@1=0.0` |
| 25472 | `g1h-7.2b` | `imoanswerbench` | 9401 | `avg@1=0.06` |
| 25199 | `g1h-13.3b` | `brumo25` | 9402 | `avg@64=0.37447916666666664` |
| 25473 | `g1h-7.2b` | `math_500` | 9403 | `avg@8=0.7265` |

`25338`, `25426`, `25437`, and `25440` had full eval rows but no score because the runner was blocked before `record_score` in optional checker handling. They were restored from existing eval rows with:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /home/rwkv/chase/rwkv-skills &&
   .venv/bin/python scripts/maintenance/restore_nonmath_scores_from_eval_rows.py \
     --execute --task-id <task_id>'
```

After DB score confirmation, only the corresponding stuck child runner PIDs were terminated with `TERM` so their matrix parents could continue. Parent logs showed `rc=-15` and continued to the next benchmark.

### BrowseComp-Plus Full830

- generation screen: finished; original screen was `1151544.browsecomp_plus_g1h72_parallel_full830_20260714_2126_dbctxfix_2147`
- official judge screen: finished; successful screen was `2818762.browsecomp_plus_official_qwen_task25391_tp4_eager_snapshot_0446`
- script: `scripts/oneoff/run_true_g1h_function_calling_matrix.py`
- DB task: `25391`
- benchmark: `browsecomp_plus`
- status/completions/scores: `Completed / 830 of 830 / 1`; the runner originally ended as deferred-judge `Failed`, then official results were backfilled to DB score `9373`.
- DB context: `830 / 830` completions have root `browsecomp_plus_run`; truncated retrieved docids: `0`.
- route: normal search steps use `parallel_candidate`; final-answer step stays direct.
- judge mode: `RWKV_BROWSECOMP_PLUS_JUDGE_MODE=defer`, so no DB score is expected before official export/judge.
- log: `results/logs/true_g1h_function_calling_browsecomp_plus_g1h72_parallel_full830_20260714_2126_dbctxfix_2147/rwkv7-g1h-7.2b-20260710-ctx10240__browsecomp_plus.log`
- exported official input: `/tmp/browsecomp_plus_official_runs_task25391_full830` (`830` files; exporter summary `completed=559`, `incomplete=271`).
- official judge log: `/tmp/browsecomp_plus_official_qwen_task25391_tp4_eager_snapshot.log`
- official judge outputs:
  - `/tmp/browsecomp_plus_official_runs_task25391_full830/evaluation_summary.json`
  - `/tmp/browsecomp_plus_official_runs_task25391_full830/detailed_judge_results.csv`
- official result / DB score `9373`: `Accuracy=0.84%` (`score=0.0084`), `Recall=4.33%`, `Calibration Error=82.91%`, `830` evaluated responses.
- official judge command uses local Qwen3-32B snapshot:
  `/tmp/hf_8222_hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
  with `/tmp/evaluate_run_vllm_fit_tp4_eager.py`, `tensor_parallel_size=4`, `gpu_memory_utilization=0.46`, `enforce_eager=True`.

Launch contract:

```bash
RWKV_BROWSECOMP_PLUS_RETRIEVER=bm25
RWKV_BROWSECOMP_PLUS_JUDGE_MODE=defer
.venv/bin/python scripts/oneoff/run_true_g1h_function_calling_matrix.py \
  --benchmark browsecomp_plus \
  --model name=rwkv7-g1h-7.2b-20260710-ctx10240,base_url=http://127.0.0.1:29572,batch=160,workers=160,jobs=1,sample_workers=16 \
  --sample-cap 0 \
  --run-mode fresh
```

After task `25391` reaches 830 completions, export:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /home/rwkv/chase/rwkv-skills &&
   rm -rf /tmp/browsecomp_plus_official_runs_task25391_full830 &&
   .venv/bin/python scripts/oneoff/export_browsecomp_plus_task_for_official_eval.py \
     --task-id 25391 \
     --expected-count 830 \
     --output-dir /tmp/browsecomp_plus_official_runs_task25391_full830'
```

Then run the official judge after choosing a free GPU:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /tmp/rwkv-official-refs/BrowseComp-Plus &&
   CUDA_VISIBLE_DEVICES=<free_gpu> /home/rwkv/chase/rwkv-skills/.venv/bin/python scripts_evaluation/evaluate_run.py \
     --input_dir /tmp/browsecomp_plus_official_runs_task25391_full830 \
     --ground_truth /tmp/rwkv-official-refs/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
     --eval_dir /tmp/browsecomp_plus_official_runs_task25391_full830 \
     --qrel_evidence /tmp/rwkv-official-refs/BrowseComp-Plus/topics-qrels/qrel_evidence.txt \
     --batch_size 64 \
     --tensor_parallel_size 1 \
     --force'
```

Failed/obsolete BrowseComp-Plus tasks to ignore:

- `25286`: full830 plus official judge, but not the desired `parallel_candidate` route. Baseline only: `Accuracy 2.05`, `Recall 3.75`.
- `25388`: stopped at 54 because DB context did not preserve usable root `browsecomp_plus_run`.
- `25282`: failed after 143 completions.

### Low-Score FC Queue

- screen: `902912.g1h72_low_fc_after_current_20260714_2132`
- state: waiting until both current BrowseComp-Plus and non-FC screens disappear.
- log: `results/logs/g1h72_low_fc_after_current_20260714_2132_screen.log`
- model endpoint: `http://127.0.0.1:29572`
- run mode: `--run-mode fresh`
- sample cap: `--sample-cap 0`

Queued benchmarks:

```text
agentbench_db
bfcl_v3
tau2_bench_retail
tau2_bench_telecom
tau3_bench_banking_knowledge
tau3_bench_mock
tau3_bench_retail
tau3_bench_telecom
tau_bench_retail
tau_bench_telecom
complexfuncbench_official
complexfuncbench_subset
longbench
longbench_qa
longbench_qa_balanced
tau_bench_airline
mcp_bench
mcp_bench_multi_3server
mcp_bench_multi_2server
mcp_bench_single
tau2_bench_airline
tau3_bench_airline
```

## Code Changes To Preserve

- `src/eval/tasks/function_calling/browsecomp_plus.py`: routes BrowseComp-Plus normal steps through `parallel_candidate`; records `decision_io="parallel_candidate"` and router metadata.
- `src/db/eval_db_service.py`: preserves root `browsecomp_plus_run` and non-truncated `retrieved_docids` for official export.
- `scripts/oneoff/export_browsecomp_plus_task_for_official_eval.py`: exports DB completions to official `<query_id>.json` files and fails fast on missing/truncated run context.
- `scripts/oneoff/run_true_g1h_function_calling_matrix.py`: BrowseComp-Plus spec uses full 830 and candidate router config.
- `scripts/oneoff/run_true_g1h_core_matrix.py`: benchmark-name alias matching avoids duplicate/rerun mismatches.
- `scripts/maintenance/restore_math_scores_from_completions.py`: supports g1h math parent score recovery with `--primary-only`.
- `scripts/maintenance/restore_nonmath_scores_from_eval_rows.py`: restores knowledge/coding scores from existing eval rows when optional checker blocks before `record_score`.
- `src/eval/tasks/knowledge/runner.py`: optional wrong-answer checker is opt-in via `--run-checker` or `RWKV_KNOWLEDGE_RUN_CHECKER=1`; disable env still wins.
- `src/eval/tasks/coding/runner.py`: optional wrong-answer checker is opt-in via `--run-checker` or `RWKV_CODING_RUN_CHECKER=1`; disable env still wins.
- `src/eval/tasks/maths/runner.py`: optional wrong-answer checker is opt-in; primary-only scoring can be enabled with `--primary-only` or `RWKV_MATH_PRIMARY_ONLY=1`.
- Tests touched: `tests/test_eval_db_service_sanitize.py`, `tests/test_function_calling_common.py`, `tests/test_maths_runner.py`, `tests/test_knowledge_runner.py`, `tests/test_coding_runner.py`.

Focused validation already run:

```text
local tests/test_eval_db_service_sanitize.py tests/test_function_calling_common.py: 53 passed, 1 warning
157 tests/test_eval_db_service_sanitize.py tests/test_function_calling_common.py: 53 passed, 1 warning
local tests/test_maths_runner.py: 5 passed
157 tests/test_maths_runner.py: 5 passed
local tests/test_knowledge_runner.py tests/test_coding_runner.py: 10 passed
157 tests/test_knowledge_runner.py tests/test_coding_runner.py: 10 passed
157 default probe: run_checker False
```

Do not revert unrelated dirty files.

## Monitoring Commands

Screens:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'screen -ls | grep -E "g1h72_nonfc_fill2_20260714_2122_primaryonly_0313|browsecomp_plus_g1h72_parallel_full830_20260714_2126|g1h72_low_fc" || true'
```

DB progress:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /home/rwkv/chase/rwkv-skills &&
   set -a && . ./.env && set +a &&
   PGPASSWORD="$PG_PASSWORD" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DBNAME" -At -F "|" \
     -c "select now();
         select t.task_id,t.status,b.benchmark_name,
                (select count(*) from completions c where c.task_id=t.task_id) comps,
                (select count(*) from eval e join completions c on c.completions_id=e.completions_id where c.task_id=t.task_id) evals,
                (select count(*) from scores s where s.task_id=t.task_id) scores
         from task t join benchmark b on b.benchmark_id=t.benchmark_id
         where t.task_id in (25391,25416,25425)
         order by t.task_id;
         select count(*) filter (where c.context ? '\''browsecomp_plus_run'\'') root_run,
                count(*) total
         from completions c
         where c.task_id=25391;"'
```

Logs:

```bash
rtk ssh -o ControlMaster=no -o ControlPath=none -J rwkv-8222 rwkv@192.168.0.157 \
  'cd /home/rwkv/chase/rwkv-skills &&
   for f in \
     results/logs/true_g1h_core_g1h72_nonfc_fill2_20260714_2122_primaryonly_0313/rwkv7-g1h-7.2b-20260710-ctx10240__maths__comp_math_24_25__test__cot.log \
     results/logs/true_g1h_function_calling_browsecomp_plus_g1h72_parallel_full830_20260714_2126_dbctxfix_2147/rwkv7-g1h-7.2b-20260710-ctx10240__browsecomp_plus.log \
     results/logs/g1h72_low_fc_after_current_20260714_2132_screen.log
   do
     echo "===$f===";
     stat -c "%s|%y" "$f" 2>/dev/null || true;
     tail -n 30 "$f" 2>/dev/null || true;
   done'
```

## Local Cleanup

Done on local checkout `/home/chase/GitHub/rwkv-skills`:

- Removed stale ignored runtime records under `tmp/`, `results/performance/`, `results/pids*`, `results/scores/`, `results/space/`, and `results/swebench_predictions/`.
- Removed local `.pytest_cache` and repo-level `__pycache__` directories.
- Kept `results/logs/agent_gpu_progress_30min_2333_8222_only_public_agent_fresh_20260709_143222.log` and `results/logs/local_web_search_proxy_18902.log` because local processes still had them open.
- Kept tracked docs. `docs/agent_loop.md` and `docs/benchmark_taxonomy.md` are still referenced by code/tests.

## 2026-07-18 Context-Budget Rerun Queue

This section records future reruns discovered during the `formatfinal_20260718_1529`
monitoring line. Do not restart the seven active core runners to apply these changes.
Their command lines were created before the change and remain on the old 4096-token
math budget. Apply this queue only after the current formal run finishes.

### Configuration Change On 157

The model endpoints still have a hard 10240-token context window. The following
changes increase the Stage 1 output budget for short-input math problems; they do
not increase the model's actual context window:

- `configs/g1h/aime24.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/aime25.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/amc23.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/arxivmath.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/beyond_aime.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/brumo25.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/cl_bench.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/cl_bench_life.toml`: `[cot].max_generate_tokens = 8192`
- `configs/g1h/cmt_benchmark.toml`: `[cot].max_generate_tokens = 8192`
- `scripts/oneoff/run_true_g1h_core_matrix.py`: removed the global math
  `--max-tokens 4096` override so benchmark TOML controls the budget.

Verified on 157: the nine targets resolve to 8192, while `algebra222` still
resolves to 4096. The launcher compiles. Global templates and all non-target
benchmarks are unchanged.

### Must Rerun With The Same 8192 Configuration

Rerun all four g1h sizes (`1.5B`, `2.9B`, `7.2B`, `13.3B`) for:

| benchmark | reason |
| --- | --- |
| `arxivmath` | Stage 1 truncation was about 99.0% / 98.1% / 97.1% for 1.5B / 2.9B / 7.2B. Across those tasks, truncated samples were 0/303 correct while non-truncated samples were 4/6 correct. Existing raw scores are budget-limited, not clean capability measurements. |
| `beyond_aime` | Active 1.5B / 2.9B runs showed about 98.9% / 97.5% Stage 1 truncation under 4096. Any result produced by the current runners remains an old-budget result. |
| `aime24` | Existing runs used the old 4096 cap and showed high stop/truncation pressure. Rerun every size so model comparisons use identical sampling and output budgets. |
| `aime25` | Same comparability requirement as `aime24`; current formal tasks were launched before the new config. |
| `amc23` | Existing small/medium-model runs had substantial Stage 1 truncation. Rerun every size with the same config rather than mixing 4096 and 8192 scores. |
| `brumo25` | The new 1.5B task `26202` truncated 1810/1920 Stage 1 outputs (94.27%) and scored `avg@64=0.008333`. Its value is close to an older 1.5B run, but remains strongly budget-limited. Rerun every size for comparable 8192-budget scores. |
| `cl_bench` | 1.5B task `26208` truncated 1193/1899 Stage 1 outputs (62.82%); its old config explicitly allowed only 1024 output tokens. The `avg@1=0.078989` result remains a short-budget baseline. |
| `cl_bench_life` | 1.5B task `26215` truncated 320/405 Stage 1 outputs (79.01%) and reported `stop_rate=0.85185`; its old config also explicitly allowed only 1024 tokens. |
| `cmt_benchmark` | 1.5B task `26222` truncated 49/50 Stage 1 outputs (98%) and scored `avg@1=0.02`. Treat the current value as budget-limited. |

Do not change each benchmark's declared `avg@k`, temperature, prompt format, or
judge mode for these reruns. Only the target Stage 1 budget changes.

### Blocked Until Data And Long-Context Handling Are Fixed

`aa_lcr` must be rerun for all four g1h sizes, but it is not ready now:

- The source metadata reports about 94,494 input tokens per sample.
- The materialized 100-row JSONL has question/answer metadata but no `context`,
  `document`, `passage`, or message body.
- Recorded prompts were only about 892 characters and the model explicitly said
  that no documents were provided. Existing 0%-3% scores are ungrounded guesses.
- Raising Stage 1 generation from 4096 to 8192 cannot repair missing 94k input.

Before rerunning `aa_lcr`, rematerialize official document text, invalidate the
old manifest/cache, and implement a benchmark-appropriate multi-stage
chunk/compaction path. Acceptance requires every sample to carry document
content and the completion trace to prove that the answer was grounded in that
content.

`gpqa_extended` also remains blocked on long-context handling rather than an
output-budget increase. During the active formal line, 13.3B NoCoT task `26217`
failed with a prompt of at least 10,240 input tokens plus the minimum one output
token. This cannot be recovered by reducing `max_tokens` or by the 8192 math
configuration. Keep the current failed row, skip the benchmark for now, and
rerun affected g1h sizes/modes only after the dedicated GPQA-Extended
chunk/compaction path is ready. The current matrix continued normally after the
failure.

### BrowseComp-Plus Experiment

Task `26187` is the completed g1h-7.2B full-830 experiment. It initially stopped
at `303/830` because a batched prompt reached at least 10,100 input tokens and
requested 141 output tokens against the 10,240-token endpoint limit. The backend
reduced `max_tokens`, but its four context retries were exhausted while different
long prompts in the batch became the next offender.

The evaluation-side backend now allows up to 16 context-length retries instead
of four, with a regression test that exposes five progressively longer prompts
before succeeding at 124 output tokens. A second format-edge fix makes the
min-think guard recognize prompts ending in incomplete `<think` as well as
`<think>` / `<think>\n`, canonicalizes the request to `<think>`, and keeps `</`
banned for the guarded prefix. Validation on 157:
`tests/test_infer_backend.py` reported `14 passed`.

Task `26187` was resumed in place at 303 completions with screen
`browsecomp_official_adapter_g1h72_resume_20260718_2255`; no new task was
created and no vLLM redeploy was required. It then completed `830/830`
completions and evaluations. The inline judge reported `10/830 = 1.2048%`, mean
`4.9` agent turns, mean `3.9` searches, and `96.75%` final-answer-tool usage.

The official BrowseComp-Plus judge ran all 830 responses on 8222 with
`Qwen/Qwen3-32B` and produced:

- official Accuracy: `1.57%`
- official Recall: `3.45%`
- Calibration Error: `91.64%`
- completed/incomplete exported agent responses: `803/27`
- official mean search calls: `3.9`

The official files were copied back to 157:

- `/tmp/browsecomp_plus_official_runs_task26187_full830/evaluation_summary.json`
- `/tmp/browsecomp_plus_official_runs_task26187_full830/detailed_judge_results.csv`
- `/tmp/browsecomp_official_qwen_task26187_gpu1_vllm019.log`

`score_id=9921` was updated transactionally in place. Existing inline diagnostic
metrics were preserved, the old value was copied to `inline_avg@1`, and the
top-level formal `avg@1` / `success_rate` became the official `0.0157`. The row
also records the official judge model, recall, calibration error, response
counts, artifact paths, and `manual_backfill_source=official_browsecomp_plus_judge`.
No duplicate score row was created.

This run is now a valid full-830 official score, but the low value is still a
real agent-quality problem: average search usage was only 3.9 calls versus the
official example's roughly 12.61. The remaining BrowseComp work is prompt/router
and evidence-use improvement, followed by a fresh task; do not overwrite task
`26187` with another experiment.

### 2026-07-19 10:00 Monitoring Snapshot

For `formatfinal_20260718_1529`, the DB reported `Completed=223`, `Failed=9`,
`Running=7`, and `formal score_total=107`. This is 38 new formal scores since
the `2026-07-19 01:00` baseline of 69. All seven runner screens and all seven
normal-model endpoints remained present; none was restarted or retuned.

The ninth failure is 13.3B GPQA-Extended CoT task `26219`. It failed with the
same known input-context condition as task `26217`: at least 10,240 input tokens
plus one requested output token. It therefore belongs to the blocked
GPQA-Extended chunk/compaction queue and is not a new inference-engine failure.

New-score audit notes:

- Healthy/improved examples include 13.3B AMC23 `avg@64=0.62148`, 2.9B
  MMLU-Pro CoT `avg@1=0.46385`, 2.9B MMLU-Redux CoT `avg@1=0.74056`, 13.3B
  ASDiv `avg@2=0.91432`, and 7.2B College-Math `avg@2=0.49556`.
- 13.3B ArxivMath `avg@1=0`, 7.2B CMT `avg@1=0.02`, and the CL-Bench /
  CL-Bench-Life values remain old-budget measurements. They stay in the 8192
  rerun queue and must not be treated as final capability scores.
- 1.5B HumanEval, HumanEval-CN, HumanEval-Fix, and HumanEval-Plus rose from old
  roughly 1% values to `0.4091 / 0.4042 / 0.3798 / 0.3801`. The gain was audited
  and is valid: each task has 5,248 independent execution rows, with
  `2147/2121/1993/1995` passes and no long completion duplicated across distinct
  samples. Old HumanEval task `25655` had 2,036 max-length outputs and 2,931
  answers missing the requested function; new task `26296` has only 32 and 38,
  respectively. The improvement is explained by repaired inference-output
  completeness, not an execution-score backfill or cross-sample contamination.

At the snapshot, 8222 GPUs 0/2/3 ran at `94%/95%/100%` utilization and the
300 W power limit. GPU 1 was free after the official Qwen judge exited. On 157,
GPUs 0/2/3 were active at `41%/44%/54%`; GPU 1 held the 1.5B endpoint but was
temporarily idle while its lane was in CPU-side LiveCodeBench work. Root storage
was 82% used with about 339 GiB free.

### 2026-07-19 10:35 Remaining Work And Experiment Endpoint

The complete four-model non-FC normal matrix contains `4 * (37 math + 31
nonmath) = 272` score pairs. At `formal score_total=109`, 163 pairs still lack a
score. The seven active launchers account for 133 of these gaps: seven Running,
eight failed within their launch sets, and 118 not yet created. The separate
7.2B nonmath set accounts for the other 30 gaps; only one of its 31 pairs has a
score and one has failed.

After first-pass coverage, 40 additional validity reruns remain: nine
budget-limited benchmarks across four models (36 tasks) plus AA-LCR across four
models after data and chunking repair. Therefore the current lower bound is 163
first-pass score gaps and 203 future task executions for a fully usable non-FC
matrix. FC is excluded from these counts; completed BrowseComp-Plus task `26187`
is not counted again.

The dedicated g1h-7.2B experiment endpoint was restored on 8222 GPU 1 without
touching the seven normal endpoints:

- 157 URL: `http://127.0.0.1:29573/v1`
- 8222 URL: `http://127.0.0.1:18073/v1`
- model: `rwkv7-g1h-7.2b-20260710-ctx10240`
- API key: `rwkv-skills`
- screen: `vllm_g1h72_gpu1_18073_experiment_20260719`
- hard context: 10,240 tokens

Both `/v1/models` and `/v1/completions` were verified through the 157 reverse
forward; a deterministic `2+2` probe returned `4`. Use `29573` for experiments,
not formal endpoint `29572`. Direct NoCoT calls should seed
`Assistant: <think></think>`. CoT calls need the evaluation backend's
`min_think_tokens=16` prefix guard; a raw `<think>` suffix alone is not the
validated CoT path.

The BrowseComp-Plus score remains low for measured agent reasons, not official
judge misalignment:

- official correctness is 13/830 (1.57%); 803 responses completed and 27 did not
- mean search count is 3.9; only nine samples reached at least ten searches
- all 830 traces have zero `get_document` and zero `get_document_chunks` calls
- official Recall is 3.45%, and only one official row contains a citation
- 2,037/4,064 router decisions had aggregate errors; 1,253 had no valid candidate
- 6,737/16,252 candidate shards failed parsing, and 2,071 decisions needed the
  final-answer format-recovery path
- the 100-iteration value is only an upper bound; no step was force-finalized,
  and most samples voluntarily answered after three to five searches
- every completed export currently says `Confidence: 100%`, producing 91.64%
  calibration error

With `chunk_tools=1`, the get-document candidate shards frequently emit a
global `search` or `final_answer` call instead of their shard tool, so they do
not survive validation. The next fresh experiment must expose/re-group all four
tools coherently, gate `final_answer` on evidence, generate short rare-entity
BM25 queries, open retrieved documents, and export model-derived confidence.
Increasing `max_iterations` alone cannot repair this run.

### Results That Do Not Need A Context Rerun

- `algebra222` remains on 4096. The observed 18k value was characters, not
  tokens; it was an outlier near the 4096-token Stage 1 cap. Current 1.5B / 2.9B
  / 7.2B scores and declining truncation rates are internally plausible.
- Current NoCoT multi-choice output is healthy: 43,128 inspected outputs were
  two-character answers such as ` A`, with no hidden reasoning. Do not rerun
  NoCoT because it lacks a chain of thought.
- Current `formatfinal_20260718_1529` CoT samples overwhelmingly contain
  non-empty reasoning. GPQA task `26165` and MMLU task `26186` had zero immediate
  closes. Include task `26195` had 2/4528 prompts ending in incomplete
  `Assistant: <think` that immediately closed before writing the explanation
  outside the tag. The incomplete-tag guard is now fixed for future child
  tasks; keep the 2/4528 rate in the score audit. Older pre-format g1h CoT rows
  with immediate `<think></think>` remain superseded and must not be used as the
  new formal baseline.

### Rerun Acceptance Checks

1. Resolve the nine target TOMLs to 8192 and a control such as `algebra222` to
   4096 before launch.
2. Use the existing benchmark `avg@k`, prompt, stop tokens, judge mode, and
   checker policy unchanged.
3. Confirm non-empty CoT and report Stage 1 truncation rate with every new score.
4. Use fresh formal task rows; do not overwrite the 4096-budget rows.
5. Record old task IDs, new task IDs, config root, model endpoint, score IDs,
   and score deltas in this handoff after completion.

### 2026-07-19 Knowledge CoT Official-Format Repair

Knowledge CoT now uses `Bot✿<think`, `stop_tokens=[0]`, and a two-strategy cascade.
Strategy A extracts the earliest formal answer after `</think>` from the same streamed
completion. Only A failures receive a fresh strategy-B CoT plus final-choice generation;
B inherits A scores and adds rescues. The parent task records cumulative B while separate
A/B strategy tasks preserve both curves. GPQA missing A/B predictions score `0.25` as in
the official script. NoCoT and the existing Math A/B/C pipeline are unchanged.

The remote completion backend now streams only requests carrying a text-answer detector.
It closes the SSE request when an answer is seen, while vLLM performs continuous batching
across those independent requests. This avoids both post-answer overwrite and the old
multi-prompt pollution path. The answer converter accepts Markdown-decorated choices and
always selects the first answer after the first closing think tag.

A no-DB 7.2B GPQA-Diamond probe (16 questions x 4 rollouts, official sampling, 8192 cap)
produced A exact/adjusted scores of `25.00%`/`39.84%`; cumulative B reached
`48.44%`/`58.20%`, with `15/48` rerouted attempts rescued and `95.31%` final valid
answers. All 116 resolved combinations (29 Knowledge benchmarks x four G1h models) were
audited to `cascade_a_b`, stop tokens `(0,)`, and official prompt suffix. All historical
G1h Knowledge CoT scores remain queued for full normal-mode rerun.

### 2026-07-19 15:08 BrowseComp Repair, 6k Probe, And Live State

BrowseComp-Plus keeps the official top-5 BM25 search, optional
`get_document`, a 100-step upper bound, no fixed search/read quota, no checker,
and deferred official judge. Task
`26378` is not active: it was stopped at 32 completions after an early audit
found premature final answers. Task `26379` was stopped after one sample because
the unbounded fallback ran away. Task `26380` (r17) completed 5/8 final outputs,
retrieved qrel evidence for 6/8, averaged 9.125 searches and 14.875 document
reads, and scored 0/8 exact.

The chunk bug is now isolated and repaired. The question remains a permanent
message; each read document stores four clue-selected chunks; every fitted
chunk preserves its opening entity window plus its strongest query window; the
generic second lexical compactor is bypassed. Candidate memory keeps the eight
documents covering the most distinct question clues rather than rewarding
repeated generic words. A real reconstruction of q804 keeps doc `94235` and
`King Jaja of Opobo` in the final roughly 9.7k-character prompt. The DB prompt
column is only a truncated trace preview and is not evidence that the runtime
prompt lost the passage.

Task `26395` (r18) completed 6/8 final outputs, retrieved qrel evidence for 5/8,
averaged 9.875 searches and 17.125 reads, and scored 0/8 exact. It showed that
retrieval/evidence ranking, not chunk absence, is now the main quality problem:
q804 saw King Jaja but selected the nearby generic title `From Slavery to
Freedom`. It also exposed current g1h top-level `annotations` / `citations`
metadata. The parallel-candidate conversion layer now ignores those two fields
without re-enabling legacy `tool`, `tool_name`, or `parameters` aliases. Local
and 157 validation both report `73 passed`; Ruff passes.

Task `26399` (r19) completed 6/8 final outputs, averaged 10.25 searches and
16.75 document reads, and scored 0/8 exact. A ground-truth audit separated the
evidence path into three layers:

- at least one qrel was retrieved for 6/8 samples;
- at least one qrel was actually read for only 5/8;
- q843 kept a qrel only as a short search snippet, while q778 and q856 had no
  qrel at retrieval time.

The frozen-evidence CoT probe generated non-empty reasoning, but a single
2048/3072-token stage looped over the noisy document list and truncated. A
512-token reasoning stage followed by a 128-token NoCoT decision stage produced
stable output syntax, but still selected nearby titles when the complete qrel
chain was absent. Do not add that two-stage synthesis to the formal loop until
retrieval-to-read recall improves.

Tasks `26403`, `26404`, and `26405` finished the bounded r20-r22 probe sequence.
All three scored `0/8` exact, so no r23 variant or full-830 run was started.

- r20 used evidence-specialized candidate shards and averaged `16.5` searches
  and `22.875` reads. Its qrel layers were `retrieved 6/8`, `read 6/8`, and
  `final evidence 5/8`; only five samples produced final answers.
- r21 added entity-bridge guidance, a twelve-document final memory, and bounded
  convergence. It averaged `10.875` searches and `15.875` reads, with qrel
  layers `6/8`, `5/8`, and `5/8`; five samples produced final answers.
- r22 forced a read of the best unread result after consecutive searches. It
  averaged `10.25` searches and `17.875` reads, with qrel layers `5/8`, `5/8`,
  and `5/8`; six samples produced final answers.

Increasing the number of read documents and final-memory chunks did not improve
exact accuracy. Three r22 samples never retrieved a qrel. The other five read a
qrel but selected a nearby entity or generic title, for example `From Slavery
to Freedom` instead of `Jaja of Opobo: The slave who became a king`, `Road
Runner` instead of `Sleepwalker`, and `Ken Shamrock` instead of `Chris Jericho`.
The remaining blocker is joint clue resolution and answer synthesis, not lost
chunks. More chunks now add distractors; resume only after a materially
different retrieval/reranking or synthesis implementation is available.

Beyond-AIME has a separate 6k diagnostic config at
`configs/g1h/experiments/beyond_aime_6k/`. It preserves `avg@64`, all sampling
parameters, and a ten-problem subset while changing only Stage 1 to 6144 tokens.
Task `26389` produced zero completions: it was mistakenly run concurrently with
the formal 13.3B math lane on endpoint 29533. Different final-stage sampling
parameters entered one rapid-sampler batch, which requires uniform scalar
temperature/top-k/top-p/penalties. The resulting `EngineDeadError` killed the
8222 GPU2 vLLM process at 14:43. Do not run the 6k probe concurrently with a
formal task on the same endpoint.

The 13.3B service was restored with the original code and launch parameters as
screen `vllm_g1h133_gpu2_18133_recover_20260719`. Both 18133 and forwarded 29533
are healthy. The crash window invalidated these formal 13.3B math tasks, which
need fresh reruns after the current first pass: `brumo25`, `cl_bench`,
`cl_bench_life`, `cmt_benchmark`, `college_math`, `comp_math_24_25`,
`frontierscience_olympiad`, and `frontierscience_research`.

At 15:27 the non-FC line was `Completed=128`, `Failed=18`, `Running=6`, and
`score_total=128`, leaving 144 of 272 score pairs without a score. The new score
was 7.2B GSM8K CoT `avg@4=0.873010`, with Stage 1 truncation `7.56%` and
strategy-c `0.924564`; its reasoning format is healthy. Beyond-AIME 6k must
use an exclusive 13.3B endpoint. At 16:07, the completed BrowseComp experiment
endpoint on 8222 GPU 1 was replaced by an isolated 13.3B service on port 18073,
forwarded to 157 port 29573. It uses the same stable vLLM launch parameters as
the formal 13.3B endpoints but is not shared by a formal runner. Keep monitoring
the six formal runners without changing their benchmark configuration.

### 2026-07-19 16:35 Beyond-AIME 6k And Browse Full Run

The 6k experiment initially used `target_samples=10`. That field is consumed by
the function-calling planner, while the shared field runner limits math records
with `max_samples`. Task `26406` therefore began the full 100-problem dataset and
was stopped after 16 completions. The isolated experiment TOML now uses
`max_samples=10`; local and 157 resolution both report ten sample indices,
`repeat_count=64`, and `effective_sample_count=640`.

Corrected task `26410` was verified with that 10-by-64 plan on the exclusive
13.3B endpoint. It was stopped after eight completions when the experiment GPU
was reprioritized for a full BrowseComp-Plus run. All eight Stage 1 generations
ended at the 6144-token `max_length` limit. This partial row is a truncation
diagnostic only, not a 6k `avg@64` score. Resume the complete 6k experiment only
after the Browse run releases the endpoint.

The experiment endpoint was restored to g1h-7.2B and verified through both
8222 port 18073 and 157 port 29573. BrowseComp-Plus task `26411` started at
16:34 with all 830 samples, official top-5 BM25, the 100-step cap, coherent
parallel candidates, no checker, and deferred judge. Runtime identifiers:

- screen: `browsecomp_plus_g1h72_full830_r22_20260719`
- log: `/tmp/browsecomp-plus-g1h72-full830-r22-20260719.log`
- result directory:
  `results/logs/browsecomp_plus_g1h72_full830_r22_20260719/`
- vLLM screen: `vllm_g1h72_gpu1_18073_browse_full830_20260719`

At launch, 8222 GPU 1 was at 100% utilization and 300 W with 72 active
requests. Treat this as a requested comprehensive measurement of the current
r22 adapter, not evidence that the 0/8 probe regression was resolved. Keep the
old official task `26187` and score 1.57% unchanged; task `26411` must receive a
fresh official judge result before any new formal score is written.

### 2026-07-19 17:00 Browse Candidate Parser Fix And R23

Task `26411` was stopped after 27 completions and marked `Failed`; it is a
diagnostic run and must not be judged or backfilled. An audit of its first 20
rows found `0/20` exact, mean `10.0` searches, mean `16.95` document reads, and
unique read docids for every executed read. Qrel evidence was retrieved for
4/20 rows, read for 3/20, and retained in the final prompt for 2/20.

Eight of those 20 rows ended without a final answer because G1h sometimes
emitted a candidate like this:

```json
{"name":"final_answer","arguments":"{\"answer\":\"...\"}","explanation":"...","confidence":0.95}
```

The candidate parser decoded `answer` but treated the outer `explanation` and
`confidence` as router metadata, so schema validation rejected the otherwise
complete final call. `parse_candidate_tool_call()` now folds those two fields
into `final_answer.arguments` only when they are missing there. It does not
restore legacy tool aliases or alter non-final candidates. Local and 157
validation both report `79 passed`; Ruff passes.

An eight-row random probe was rejected before task creation because that sample
had zero BM25/qrel overlap and the small-sample preflight correctly failed.
Full-dataset preflight still passes. The replacement all-830 run started at
17:00 with the same retrieval, prompt, concurrency, and judge settings:

- screen: `browsecomp_plus_g1h72_full830_r23_parserfix_20260719`
- log: `/tmp/browsecomp-plus-g1h72-full830-r23-parserfix-20260719.log`
- result directory:
  `results/logs/true_g1h_function_calling_browsecomp_plus_g1h72_full830_r23_parserfix_20260719/`
- endpoint: 157 `29573` -> 8222 `18073`, g1h-7.2B

Audit the first approximately 20 r23 rows before treating the conversion issue
as resolved. The old official task `26187` and its 1.57% score remain unchanged
until r23 completes and receives a fresh official judge result.

r23 created task `26420` but was stopped with zero completions after replaying
the exact malformed DB text exposed two additional current G1h final-only
shapes: a flat `answer` object and an answer value emitted in `name`. The
conversion remains restricted to a final-only shard. It now accepts these
current outputs, ignores only harmless outer metadata on a final answer, and
still rejects legacy normal-tool aliases. Replay of task `26411` recovered a
schema-valid final call for seven of its eight previously empty rows; the last
row contained no answer anywhere and remains correctly invalid. Local and 157
validation both report `84 passed`; Ruff passes.

The replacement r24 full-830 run started at 17:12 with no other configuration
change:

- task: created after preflight; query by the r24 stamp below
- screen: `browsecomp_plus_g1h72_full830_r24_conversionfix_20260719`
- log: `/tmp/browsecomp-plus-g1h72-full830-r24-conversionfix-20260719.log`
- result directory:
  `results/logs/true_g1h_function_calling_browsecomp_plus_g1h72_full830_r24_conversionfix_20260719/`

Tasks `26411` and `26420` are failed diagnostics. Use only r24 for the new full
measurement and official judge.

r24 task `26423` was stopped and marked `Failed` after 18 diagnostic
completions. Final-answer conversion improved, but its first 13 rows still had
zero qrel retrieval. The cause was visible in the persisted search trace: 126
of 195 search calls stringified a list of several queries and sent the whole
representation to the BM25 string interface. This violated the official
one-search-call/one-query contract and produced long mixed-clue retrieval
expressions.

`_browsecomp_plus_call_from_candidate()` now treats a sequence-valued
`search.query` as candidate queries and executes the first non-empty query not
already used. Scalar strings and duplicate-query rejection are unchanged. The
focused test suite passes locally and on 157 with `86 passed`; Ruff passes.

The replacement r25 full-830 run started at 17:33 with the same model,
concurrency, BM25 top-k, 100-step maximum, no-checker, and deferred-judge
settings:

- screen: `browsecomp_plus_g1h72_full830_r25_queryfix_20260719`
- log: `/tmp/browsecomp-plus-g1h72-full830-r25-queryfix-20260719.log`
- result directory:
  `results/logs/true_g1h_function_calling_browsecomp_plus_g1h72_full830_r25_queryfix_20260719/`

Treat r25 as the only active full measurement. Tasks `26411`, `26420`, and
`26423` are diagnostics and must not be officially judged or backfilled.

At 17:52 a process audit found that exiting the r23/r24 matrix screens had left
their `src.main --config ...browsecomp...` children orphaned under PPID 1. They
were still sharing endpoint 29573 with r25. Only the two obsolete child PIDs
were terminated; all formal runners and r25 were preserved. Tasks `26420` and
`26423` were marked `Failed` again. Future Browse cancellation must verify both
the screen and the config-specific `src.main` child rather than assuming
`screen -X quit` stopped the workload.

After cleanup, r25 task `26424` began writing normally. Its first 23 rows had
zero stringified query lists out of 233 searches, qrel retrieval in 4/23 rows,
qrel reads in 2/23, and final qrel evidence in 2/23. Exact was still 0/23, so
the active blockers are retrieval coverage and cross-clue answer synthesis;
the query type mismatch itself is resolved. Keep r25 running.

### 2026-07-19 18:39 R25 100-Row Audit

Task `26424` remains the only active BrowseComp-Plus measurement and reached
100/830 completions without vLLM errors. It produced 83 valid final answers,
17 rows without a final answer, and 20 rows with an agent error. Mean behavior
was 28.96 turns, 10.02 searches, and 17.57 document reads. Only one row used at
most five searches, so the old two-search early-stop behavior has not returned.
No search query was a stringified list.

Qrel coverage remains the primary blocker: 19/100 rows retrieved a qrel and
15/100 read one. Strict normalized exact match was 1/100. At least one
additional answer was semantically exact but wrapped in a sentence (`The
magazine is The Dial.` versus `The Dial`), so do not convert this diagnostic
exact count into a score. The official Qwen judge must run after all 830 rows
finish. The currently visible 1-2% band is close to the old official 1.57%
score, but most qrel-reading rows still selected a nearby entity instead of the
requested central answer.

At 18:30 the formal non-FC line was `Completed=138`, formal `Failed=17`,
`Running=5`, and `score_total=138`, leaving 134 of the 272 expected scores.
The only new score since 18:00 was 2.9B Hendrycks Math: Stage 1 `avg@1=0.5378`
and strategy-c `0.6514`; the old strategy-c value was `0.6522`, so the repaired
final-answer path is aligned. Stage 1 still had 39.8% truncation and remains a
long-budget retest candidate.

Thirty-second GPU sampling showed that all four 8222 cards remained near their
300 W power limit. On 157, GPU 0 averaged roughly 42% utilization and GPU 2
roughly 60%, while GPUs 1 and 3 stayed at 0% because their non-math matrix
runners had already exited and only the vLLM services remained. Do not change
the active formal runners during monitoring. A later throughput pass can shard
disjoint remaining benchmark lists onto those two idle endpoints without
changing benchmark repeat counts.
