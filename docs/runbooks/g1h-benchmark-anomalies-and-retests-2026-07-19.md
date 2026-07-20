# G1h Benchmark Anomalies and Retests (2026-07-19)

This is the authoritative queue for invalid, suspicious, interrupted, and not-yet-run
G1h non-FC benchmark results. A row remains here until a contract-matched rerun has
completed and its completions and score have been checked.

## Scope and invariants

- Models: G1h 1.5B, 2.9B, 7.2B, and 13.3B.
- Formal inference endpoints use vLLM. Normal and native differ only in evaluation-side
  prompt/sampling configuration; the current priority is normal.
- Preserve each benchmark's registered `avg_k`, CoT mode, scorer, split, and sample count.
  Do not normalize benchmarks to `avg@1` or otherwise change `k` for convenience.
- No checker. Keep the benchmark's judge only where its registered scorer requires one.
- G1g/G1h comparisons are valid only when prompt contract, `avg_k`, sample count, scorer,
  split, and decoding budget match.
- BrowseComp-Plus task 26424 and endpoint `157:29573 -> 8222:18073` are outside this
  stop/update/redeploy operation and must not be interrupted.

## Frozen state before update

Snapshot: 2026-07-19 19:22 CST, old registry and `formatfinal_20260718_1529` run.

- Runnable registry combinations: 68 per model, 272 total.
- Formal scores written: 141.
- Formal failed combinations after the intentional stop: 22.
- Remaining under the old registry: 131 = 22 failed reruns + 109 not started.
- All formal runners are stopped. BrowseComp-Plus remains Running and continued from
  170 to 177 completions during stop verification.
- The final remaining five formal tasks were intentionally interrupted before the code
  and inference update. Their DB descriptions record that they must be rerun.

The total must be recomputed after `git pull`, because the remote adds benchmarks. Count
benchmark/model/mode combinations from the registry and resolved datasets, not screen or
runner processes.

## Interrupted tasks

| Task | Model | Benchmark | Reason | Required action |
| --- | --- | --- | --- | --- |
| 26385 | 13.3B | mmlu_pro CoT | Update-time stop, 0 completions | Full rerun |
| 26398 | 13.3B | gaokao2023en CoT | Update-time stop at 4288 completions | Fresh full rerun |
| 26427 | 1.5B | minerva_math CoT | Update-time stop at 4107 completions | Fresh full rerun |
| 26433 | 7.2B | hendrycks_math CoT | Update-time stop at 384 completions | Fresh full rerun |
| 26439 | 2.9B | imoanswerbench CoT | Update-time stop at 128 completions | Fresh full rerun |

## Failed combinations in the frozen matrix

The interrupted five above are included in this list.

- 1.5B: `minerva_math CoT`, `gpqa_extended NoCoT`, `gpqa_extended CoT`.
- 2.9B: `imoanswerbench CoT`, `cmmlu CoT`, `gpqa_extended NoCoT`,
  `gpqa_extended CoT`.
- 7.2B: `hendrycks_math CoT`, `ceval CoT`.
- 13.3B: `brumo25 CoT`, `cl_bench CoT`, `cl_bench_life CoT`,
  `cmt_benchmark CoT`, `college_math CoT`, `comp_math_24_25 CoT`,
  `frontierscience_olympiad CoT`, `frontierscience_research CoT`,
  `gaokao2023en CoT`, `ceval CoT`, `gpqa_extended NoCoT`,
  `gpqa_extended CoT`, `mmlu_pro CoT`.

`gpqa_extended` is the known long-context failure class. Do not accept it as a model
failure. Rerun only after its long-context strategy is explicitly selected and validated.

## Suspicious scored math tasks

These scores are diagnostic, not final capability claims. Inspect the completion tail,
stop reason, answer extraction, and strategy-C result before replacing them.

| Score/task | Model/benchmark | Symptom | Required retest |
| --- | --- | --- | --- |
| 9986/26287 | 13.3B Beyond-AIME | Stage 1 0.066875, stop 94.47%; strategy-C 0.1175 versus old 0.154688 | Validated longer-budget run; 6K probe failed before completion |
| 10000/26409 | 2.9B Hendrycks Math | Stage 1 0.5378, stop 39.8%; strategy-C 0.6514 matches old 0.6522 | Rerun after budget policy is fixed; strategy-C is sanity reference |
| 10002/26430 | 2.9B HMMT Feb 25 | Stage 1 0.008333, stop 98.07%; strategy-C 0.016146 | Longer-budget full rerun |
| 10003/26436 | 2.9B HorizonMath | Stage 1 0.017857, stop 94.64%; strategy-C 0.071429 | Longer-budget full rerun |
| 9976/26332 | 1.5B Hendrycks Math | Stage 1 0.346, stop 54.82%; strategy-C 0.4702 | Rerun after budget policy is fixed |
| pending | 1.5B MAWPS | Stage 1 0.72397, stop 37.22%; strategy-C 0.92421 | Rerun after budget policy is fixed |
| pending | all ArxivMath | About 1%-2% with about 97%-99% Stage 1 truncation | Longer-budget full rerun |
| pending | all AA-LCR | About 0%-3%, inconsistent with expected math curve | Completion/extractor audit and full rerun |
| pending | MathArena Apex 1.5B | Score 0, Stage 1 stop 100% | Full rerun; do not publish zero as capability |

The configured weight context is 10,240 tokens. A generation budget must also leave room
for the input prompt. Do not infer that an 18K-character completion necessarily exceeded
10K tokens; measure tokenizer lengths and finish reasons. Chunking is not a default math
strategy and must not silently alter ordinary math prompts.

## Prompt and output-format validity

- Historical CoT tasks that immediately emitted `<think></think>` are invalid and must be
  rerun. The current G1h CoT contract starts with `<think>` and enforces a minimum reasoning
  prefix before accepting a closing tag.
- NoCoT is expected to have no reasoning chain. Do not apply the CoT minimum-think guard to
  NoCoT.
- Before the full relaunch, probe all four model sizes for stable CoT and NoCoT output and
  run a multi-prompt contamination test against every newly deployed endpoint.
- Scores produced before the final prompt-format repair remain retest candidates even if
  their numeric value looks plausible.

### Knowledge CoT A/B contract (2026-07-19)

All four G1h sizes and all 29 registered Knowledge benchmarks now resolve the same
normal-mode CoT contract:

- Prompt ends with the official `Bot✿<think` prefix. Knowledge CoT stop tokens are only
  `[0]`; token `10060` (`✿`) must not terminate reasoning.
- Strategy A is one completion. Generation ends when the first formal answer after the
  first `</think>` is observed, on EOS, or at the configured length limit.
- Only A failures or missing answers enter strategy B. B starts a fresh CoT rollout and
  then generates the final choice. The cumulative B score inherits every A result and
  adds only B rescues.
- A and cumulative B are stored as separate strategy tasks. The formal parent score uses
  cumulative B. GPQA missing predictions retain the official `0.25` score; this does not
  alter NoCoT scoring.
- The extractor keeps the earliest formal answer after `</think>`, including Markdown
  forms such as `The correct answer is **B. ...**`. Later continuation cannot overwrite it.
- Completion-style remote generation uses individual SSE requests when this answer
  detector is enabled. vLLM still continuous-batches them internally, while the client
  closes each sequence as soon as an answer is available.

The no-DB G1h-7.2B GPQA-Diamond probe used 16 questions, four rollouts, the official
sampling (`temperature=0.96`, `top_p=0.76`, `top_k=32`, presence `1.0`, frequency
`0.1`, decay `0.988`), and an 8192-token local cap. Results:

| Stage | Exact | GPQA score | Valid | Other |
| --- | ---: | ---: | ---: | --- |
| A | 25.00% | 39.84% | 40.63% | 26 answer stops, 38 length stops |
| cumulative B | 48.44% | 58.20% | 95.31% | 48 rerouted, 15 rescued |

The probe wrote no DB task or score. Historical Knowledge CoT results generated with
`stop_tokens=[0,10060]`, immediate empty-think prompts, or unconditional old two-stage
answering are invalid and all Knowledge CoT combinations must be rerun for every G1h
size. Math keeps its separate A/B/C policy.

## Scheduling and registry anomalies

- The old core matrix resolves 37 math, 22 knowledge-mode, 7 non-SWE coding, and 2
  instruction-following combinations per model.
- `hle` and `hy_math` were registered but had no resolved local dataset.
- Seven SWE-bench variants require their official harness and were skipped by the normal
  core matrix.
- `arena_hard_v2`, `flores200`, and `wmt24pp` were placeholders without runnable scheduler
  jobs in the frozen registry. Re-audit after pull.
- The matrix DB-state lookup uses broad benchmark/dataset aliases. The frozen audit showed
  nine missing combinations could be counted as scored through alias collisions. Fix or
  prove this lookup before relaunch so new benchmarks are not silently skipped.
- Count and report all runnable benchmark/model/mode combinations after pull, including
  newly added instruction-following benchmarks. Separate source-required placeholders from
  runnable tasks.

## G1g/G1h parity audit

Before choosing coding or instruction-following reruns, compare exact model-size pairs:

- G1g: `rwkv7-g1g-{1.5,2.9}b-20260526-ctx8192` and
  `rwkv7-g1g-{7.2,13.3}b-20260523-ctx8192`.
- G1h: `rwkv7-g1h-{1.5,2.9,7.2,13.3}b-20260710-ctx10240`.

Flag and rerun when any of these differ: `avg_k`, metric key, evaluator, benchmark split,
sample count, CoT mode, prompt profile, max tokens, judge mode, or extraction/scoring
implementation. Also flag large G1h-over-G1g coding deltas and non-monotonic size curves;
inspect completions before attributing either to model capability.

### Historical avg-k drift found on 2026-07-19

The latest scored normal-evaluator rows exposed 36 G1g/G1h model-size and CoT-mode
pairs whose recorded `sampling_config.avg_k` did not match. The affected benchmark
families and formal G1g values are:

| Benchmark | G1g normal avg-k | Previous G1h avg-k | G1h rerun avg-k |
| --- | ---: | ---: | ---: |
| `beyond_aime` | 32 | 64 | 32 |
| `gaokao2023en` | 8 | 16 | 8 |
| `ifbench` | 16 | 1 | 16 |
| `math_odyssey` | 8 | 16 | 8 |
| `mbpp` / `mbpp_plus` | 8 | 16 | 8 |
| `simpleqa` | 4 | 8 | 4 |
| `supergpqa` | 0.2 | 1 | 0.2 |

The G1h-only files under `configs/g1h/` override only `avg_k` and
`report_avg_k`. The loader still inherits each public benchmark's prompt template,
sampling parameters, split, evaluator, and scoring implementation. Existing G1h
scores with the previous k values remain historical rows and must not be compared
directly to G1g. Legacy `*_naive` G1g rows are also excluded from normal-mode parity
comparisons because their evaluator differs.

### Unified scheduler backpressure note

The 2026-07-19 router can forward authenticated inference requests, but its vLLM
metrics probe does not send the backend API key. `/v1/backpressure` consequently
reported every backend as HTTP 401 with `ok_route_count=0`, which blocked dispatch
before any task row was created. `configs/scheduler/g1h-normal-7gpu.toml` disables
that unavailable signal and uses the profile's static, parameter-size concurrency
budgets. This changes launch control only; benchmark prompts, k values, and scoring
are unaffected.

## 2026-07-20 live audit addition

New scores must be checked at completion level, including passing and failing answers,
answer/reference normalization, format, finish reason, truncation, scorer decision,
and benchmark-specific Math/Knowledge routing. BrowseComp-Plus is audited separately
for search-query shape, retrieved/read/evidence traces, candidate/chunk conversion,
final-answer fields, agent errors, and official-judge completeness. See
`docs/runbooks/g1h-live-monitoring-and-score-audit-2026-07-20.md` for the active
30-minute report contract and the 830-row promotion gate for 29573.

## Relaunch gate

1. Pull the updated project without discarding the dirty BrowseComp-Plus work.
2. Recompute the full runnable registry and remaining count.
3. Upload and install the optimized local `~/GitHub/vllm-rwkv` on both servers.
4. Redeploy the seven formal endpoints; leave the Browse endpoint untouched.
5. Pass endpoint health, format, multi-prompt contamination, memory, utilization, and power
   checks.
6. Launch by missing/failed/anomalous benchmark combinations, with math first. Runner count
   is only an execution detail.
7. Record each replacement task and score here, then remove the anomaly only after a
   completion-level audit.

## 2026-07-19 23:30 Handoff Snapshot

All 18 previously missing new Knowledge datasets are now materialized in the shared 157
data directory and validated against fixed expected row counts: 18 files, 475,482 rows,
510,991,346 bytes, no invalid JSON/canonical question-answer rows. The normal non-FC
runnable matrix is now 104 combinations per model and 416 total: 37 math, 58 Knowledge,
7 coding, and 2 instruction-following per model. The new post-repair full rerun remains
0/416; old scores are comparison evidence only.

Seven formal vLLM endpoints and one G1h-7.2B Browse experiment endpoint are deployed and
passed authenticated model-list and completion probes. Exact endpoints, physical GPU
placement, startup constraints, execution order, and the next-window prompt are recorded
in `docs/runbooks/g1h-codex-handoff-2026-07-19.md`.

## 2026-07-20 live retest entries

- Browse task `26454` is invalid diagnostic evidence: its remote process loaded the
  pre-r26 implementation and was stopped after the code audit. It has 79 completions,
  no eval/score, and is marked `Failed`; it must not count toward Browse completion or
  the formal matrix. The replacement r26/full-830 task is `26461` on `29573`.
- Fresh Math tasks `26447` and `26450` produced scores `10004` and `10005`. Their
  sampled wrong answers were ordinary `math_verify_false` cases, not parser/checker
  failures, while sampled correct answers matched references. Both have high stage-1
  `max_length` rates (69.90% and 76.09%), but the latest comparable historical rates
  were even higher; retain this as a model/output optimization anomaly rather than
  silently accepting the low aggregate scores as capability evidence.
- The seven formal runners remain active. Any future retest must preserve the Math
  TOML's 8192-token budget and A/B/C strategy contract; no checker is authorized.
- By 01:08 CST, new scores `10006` (2.9B AIME24), `10007` (2.9B AIME25), and
  `10008` (1.5B HMMT-Feb25) were also written and sampled. The AIME25 2.9B
  malformed extraction `9b - kb = -7k` is a concrete format/extraction anomaly;
  HMMT's 75.10% stage-1 truncation is a high-truncation anomaly. Both remain
  documented for narrow repair/retest and are not promoted as clean capability
  evidence solely from their numeric scores.
- Browse r26 task `26461` early audit (16 rows) found 12 completed traces with
  scalar BM25 queries, document reads, and final-answer fields, plus 4 forced-final
  failures caused by malformed candidate formats (`answer` missing and legacy/extra
  fields). Official judging remains deferred until the full 830-row run and trace
  audit are complete.
- Score `10009` (1.5B Brumo25) is 0.3125% versus the comparable 0.83333% historical
  value, with 53.33% stage-1 truncation. Sampled wrong rows were math-verifier
  false cases rather than output-parser exceptions; retain it for the Knowledge
  cascade/truncation retest queue.

## 2026-07-20 01:22 CST Browse format decision

Browse task `26461` had 35/830 rows, with 30 completed and 5 incomplete. The
incomplete rows reached the final stage after normal BM25 search and document
reads, then produced invalid candidate tool-call formats (missing `answer`,
legacy `function_call`, unsupported metadata/fields, or unterminated JSON).
The runner must not recover an answer from an explanation, search snippet, or
document: that would change the benchmark answer rather than repair transport
format. Keep the strict explicit-answer requirement, record the natural invalid
rate, and only consider a future prompt/schema prefill change after this full
run's trace rate is known. No official Browse score exists yet.

## 2026-07-20 01:36 CST HMMT 2.9B anomaly

Fresh task `26466` / score `10012` (G1h-2.9B HMMT-Feb25 CoT) is
`avg@64=0.0005208333` versus historical task `26430` at `0.0083333`
(delta `-0.78125` percentage points). Only 1/1920 rows passed. The sampled
pass was `20/20`; sampled wrongs were `5/103`, `0/103`, and `10/103`, all
`math_verify_false`. Stage-1 truncation was 841/1920 (43.80%). This is a
model/output strategy anomaly, not a justification for answer modification;
preserve the HMMT TOML and verifier contract for any later retest.

## 2026-07-20 01:27 CST AIME24 7.2B anomaly

Fresh task `26449` / score `10010` (G1h-7.2B AIME24 CoT) is
`avg@64=0.0005208333` versus the latest same-model/same-benchmark historical
task `26092` at `0.0973958` (delta `-9.6875` percentage points). Only 1/1920
rows passed; the sampled pass was `25/25`, while a sampled wrong was
`437 + 23/73` with `math_verify_false`. Stage-1 truncation was 410/1920 and
stage-2 max-length was 1908/1920. Preserve the dedicated TOML and no-checker
contract; inspect completions and Math A/B/C output/extraction before any
narrow strategy retest. Do not fabricate or repair an answer from reasoning.

## 2026-07-20 02:00--02:39 CST missed reports backfilled

The server-side watch was active, but no user-facing reports were emitted. The
missing checkpoints are now recorded in the live audit. Browse task `26461`
advanced from 88/830 at 02:00 to 132/830 at 02:30 and 147/830 at 02:39; the
latest nested result was 124 completed and 23 incomplete. All 23 incomplete
rows had normal search/read activity and failed only at the final candidate
format (`no valid candidate tool calls`).

Fresh primary scores added during this interval were `10013` Algebra222,
`10014` Brumo25 2.9B, `10015` Beyond-AIME, `10017` Math-Odyssey, `10018`
AIME25 13.3B, `10019` AIME24 13.3B, `10020` AIME25 7.2B, and `10021` SimpleQA.
Completion samples, references, fail reasons, truncation counts, and historical
comparisons are recorded in the live audit. Primary fresh progress is 16/416;
`10011` and `10016` are auxiliary `answer_judge` rows and do not count.

## 2026-07-20 07:16 CST monitoring outage

At 07:16 CST the 157 server became unreachable through both the jump host
(`47.115.88.183:8222` refused) and direct private SSH (`192.168.0.157:22`
timed out). The last trusted checkpoint is 02:44 CST: Browse `154/830`
(`130 completed / 24 incomplete`) and formal primary fresh `16/416`. This is
an observability outage; no runner or endpoint is to be restarted or released
until connectivity returns and DB/process/GPU state is revalidated.
