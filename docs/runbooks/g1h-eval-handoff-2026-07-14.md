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
