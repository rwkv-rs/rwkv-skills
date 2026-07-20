# G1h live monitoring and score audit (2026-07-20)

This is the live operating record for the fresh G1h normal rerun. It supplements
the 2026-07-19 handoff and anomaly notes; it does not replace benchmark TOMLs,
the scheduler profile, or the existing BrowseComp-Plus implementation.

## Scope and non-negotiable rules

- Seven formal endpoint-affine runners stay active. Do not pause them for a
  suspicious individual result unless the issue is severe enough to threaten
  data integrity, the endpoint, or the host.
- The formal normal non-FC target is 416 combinations. Historical
  `formatfinal`/completed rows never count toward this fresh total.
- BrowseComp-Plus runs concurrently on `29573` as the G1h-7.2B experiment. It
  becomes the eighth formal lane only after the promotion gate below passes.
- Inference is vLLM only, normal mode only, no checker. Keep every benchmark's
  own TOML, split, k, sample count, scorer, judge, and generation budget.

## Live snapshot at handoff

At 2026-07-20 00:41 CST, tasks `26447`--`26453` were all `Running` on the
seven formal endpoints. They were the first AIME24/AIME25 Math tasks; each had
completed completions but no new score yet. Browse task `26454` was also
`Running` on `29573`, with 25/830 completed completions and no official eval or
score. The eight authenticated `/v1/models` probes were HTTP 200. GPU3 on 157
was not idle: the Browse endpoint was using it through the 8222 tunnel.

This snapshot is evidence only; every report must query the live DB and host
before stating progress.

## Every 30-minute report

Record one report in the conversation and, when there is a material anomaly,
in this file or the anomaly log:

```text
Half-hour report:
Current state: Completed / Failed / Running; fresh formal N/416; score_total;
disk free and usage; live runner/task screens.
New scores: model + benchmark + mode + task id; new metric; comparable old
value and delta; completion audit result.
Problems/actions: failures, truncation, malformed format, suspicious curve,
exact evidence read, code/config change, and whether a retest was queued.
Active tasks: task id, model, benchmark/mode, completed/total completions,
eval progress, score state.
GPU: each card's memory, utilization, and power; explain any idle card.
Next: concrete checks before the next report.
```

## Formal score audit: required for every new score

1. Identify the new task and score rows from the authoritative 157 DB. Confirm
   that the task is `fresh`, normal, uses the expected benchmark identity and
   split, and points at the benchmark's dedicated TOML. Confirm the endpoint
   and model are the intended affine pair.
2. Recompute completion/eval counts and inspect `finish_reason`, token counts,
   truncation flags, errors, and stored format fields. A score is not accepted
   merely because the scorer returned a number.
3. Sample both passing and failing rows. For each sample compare the model
   answer with the reference and inspect the scorer decision, normalized
   answer, failure reason, and any extracted final-answer field. For Math,
   separately check A/B/C strategy behavior and that direct NoCoT answers were
   not turned into hidden CoT. For Knowledge, check the A/B cascade: B may
   rescue A failures, but the reported result must retain the stage and reason.
4. Check output shape and protocol: no checker calls, no accidental tool calls
   in non-FC tasks, no prompt contamination, no empty/fabricated final answer,
   no benchmark-wide format fallback, and no silent scorer exception. High
   truncation or malformed output is an anomaly even when the aggregate score
   looks plausible.
5. Compare only contract-matched old G1g/old G1h values. Record the new value,
   old value, delta, sampled correct/wrong examples, diagnosis, and action.
   A low score is not fixed by backfilling an old `formatfinal` row; queue a
   fresh retest after the cause is repaired.
6. If the issue is localized, leave the seven runners moving and patch the
   narrow code/config path, then queue a replacement task. Pause only for
   endpoint corruption, cross-task contamination, widespread wrong format,
   data loss, or another severe integrity/host failure.

## BrowseComp-Plus audit protocol

Do not judge or publish a partial Browse score. During the run, query task
`26454` and sample completions at each report. Required trace checks are:

- `browsecomp_plus_run` exists and retains the root question/run metadata;
- search calls contain scalar queries, not a Python/list representation;
- retrieved document IDs, `get_document` reads, chunks/passages, qrels, and
  final evidence are present and internally consistent;
- the agent reaches a final answer without an agent error, malformed candidate,
  premature stop, or unexplained max-step exhaustion;
- candidate/chunk conversion and `parallel_candidate` routing preserve the
  final answer, explanation, confidence, and evidence fields;
- counts for search/read/final-evidence, duplicate queries, missing documents,
  incomplete/truncated outputs, and answer fields are recorded; after judging,
  inspect both correct and wrong examples, including false citations and
  unsupported answers.

The full-run promotion gate is all of the following: 830/830 completions;
complete eval/export inputs with no unexplained truncation; official Browse
judge output for all rows; DB score backfilled with its source and task id; and
the trace/evidence audit passes on both correct and incorrect samples. Only
then may GPU3/29573 be assigned the next formal normal queue as lane eight.

## What changed in the Browse experiments

The earlier probes established the repair order and are not trusted formal
scores:

- r17--r22 isolated retrieval, document reading, and final-evidence failures;
  the blocker was evidence/reasoning quality, not merely missing chunks.
- r23/r24 repaired final-candidate parser shapes.
- r25 repaired sequence-valued queries being stringified before BM25. Its
  100-row diagnostic had 1 strict exact match, 19 qrel retrievals, 15 reads,
  83 valid finals, 17 missing finals, and 20 agent errors; it was useful for
  diagnosis but not a publishable score.
- The current r26/full-830 implementation keeps top-5 BM25, up to 100 steps,
  `parallel_candidate`, chunk conversion, explicit final answer/explanation/
  confidence parsing, retrieved-doc/evidence traces, and deferred official
  judging. The live run is task `26454` on `29573`.

At every stage, the decision is based on DB rows plus raw completions and trace
fields, not on aggregate accuracy alone. A repair is recorded only with the
observed failure pattern, the narrow code/config change, a replacement run,
and a post-fix sample audit.

## Local evidence record

Append each material event below with CST timestamp, task id, DB counts, sampled
row IDs, diagnosis, action, and verification result. Keep this file local until
the current run is stable; then copy the final handoff state to the shared
formal checkout without overwriting unrelated dirty work.

### 2026-07-20 00:41 CST

- Seven formal tasks `26447`--`26453`: all Running; no new scores.
- Browse task `26454`: Running, 25/830 completions, no official judge/score.
- 157 GPUs 0--3 were loaded and busy; GPU3 was serving Browse through 29573.
- No scheduler `failed`, `error`, `traceback`, engine-dead, or OOM marker found.

### 2026-07-20 00:54--01:00 CST

- The first Browse task `26454` was the server's old Browse implementation, not
  r26. It produced 79 diagnostic completions and no eval/score. Its completion
  traces showed BM25/search/read markers, but the old code did not persist the
  newer document-read/search/evidence fields. After the exact old process was
  stopped, the stale DB task was marked `Failed`; it is excluded from all
  progress and score totals.
- The local r26 Browse files and their two required prompt/contract dependencies
  were copied narrowly to the formal checkout. Remote py_compile passed and the
  focused function-calling suite passed 67 tests. The first r26 start exposed
  the missing dependency before task creation; after syncing those dependencies,
  task `26461` was created and is `Running` on `29573`.
- New formal score `10004`, task `26447`, G1h-1.5B AIME24 CoT:
  `avg@64=0.0005208333` (0.0520833%). Latest contract-matched historical
  comparable task `26086` was 0.0015625 (0.15625%), delta -0.10417 percentage
  points. Sampled pass `65828870` had answer/reference `540/540`; sampled wrong
  `65825082` had `900/73`, `math_verify_false`.
- New formal score `10005`, task `26450`, G1h-1.5B AIME25 CoT:
  `avg@64=0.0067708333` (0.6770833%). Latest comparable task `26099` was
  0.0145833 (1.45833%), delta -0.78125 percentage points. Sampled pass
  `65825155` had `60/60`; sampled wrong `65825146` had `7/70`,
  `math_verify_false`.
- Fresh Math stage-1 `max_length` rates were 1342/1920 (69.90%) for task
  26447 and 1461/1920 (76.09%) for task 26450. Historical comparable rates
  were 1878/1920 and 1836/1920, so this is not a new truncation regression;
  it remains a high-priority model/output optimization anomaly. The benchmark
  TOML budget stays unchanged at 8192, and no checker was used.
- At the same check, formal tasks 26447/26450 were complete and the other five
  formal lanes remained active. GPU3 was busy generating r26 on 29573.

### 2026-07-20 01:08 CST score batch

- Score `10006`, task `26451`, G1h-2.9B AIME24 CoT: `avg@64=0.0005208333`
  (0.0520833%), versus historical task `26089` at 0.0270833% (delta
  -2.65625 percentage points). Sampled pass `65831988` was `25/25`; sampled
  wrongs `65825210` and `65825211` were `4/73` and `1/699`, both
  `math_verify_false`. Stage-1 truncation was 865/1920 (45.05%).
- Score `10007`, task `26453`, G1h-2.9B AIME25 CoT: `avg@64=0.009375`
  (0.9375%), versus task `26102` at 0.046875% (delta -3.75 percentage
  points). Sampled pass `65825288` was `60/60`; sampled wrong
  `65825250` produced malformed/incomplete extracted answer `9b - kb = -7k`
  versus `70`, and `65825251` produced `100/117`; both were
  `math_verify_false`. Stage-1 truncation was 859/1920 (44.74%). The
  `9b - kb = -7k` case is recorded as an answer-format/extraction anomaly,
  not silently treated as a model-capability miss.
- Score `10008`, task `26458`, G1h-1.5B HMMT-Feb25 CoT:
  `avg@64=0.0005208333` (0.0520833%), equal to historical task `26349`
  (delta 0). Sampled pass `65841851` was `20/20`; sampled wrongs
  `65841148` and `65841149` were `7/103` and `8/\\frac{1}{576}` with
  `math_verify_false`. Stage-1 truncation was 1442/1920 (75.10%).
- All five new scores currently have stop rates at or above 0.9989 and no
  checker/judge row. The low Math scores are therefore not being accepted as
  clean capability curves: wrong outputs and high truncation remain linked
  anomalies for later narrow prompt/extraction optimization and contract-matched
  retest.

### Browse r26 early trace audit at 01:08 CST

- Task `26461` had 16 completed completion rows: 12 `completed` runs and 4
  `incomplete` runs. Completed runs carried 8--12 scalar BM25 searches,
  14--20 `get_document` reads, 15--46 retrieved docids, one `final_answer`,
  and non-empty explanation/final-answer fields. No list-valued query marker
  appeared in the sampled traces.
- The four incomplete runs failed only at the forced final stage with
  `no valid candidate tool calls`. The candidate-shard errors were mostly
  `final_answer` missing `answer` (7/8 shard errors), plus unsupported legacy or
  extra fields such as `function_call`, `birth_date/death_date`, and
  `tool_call_id`. One raw candidate used `arguments` containing an explanation
  but no actual answer key. This is a real final-format failure to track, but
  the run continues to gather the full 830-row rate before deciding on a repair.
- The official judge remains deferred; none of these 16 rows is a trusted Browse
  score. GPU3/29573 remains active while this trace audit continues.

### 2026-07-20 01:15 CST score batch

- Score `10009`, task `26455`, G1h-1.5B Brumo25 CoT:
  `avg@64=0.003125` (0.3125%), versus historical task `26202` at 0.0083333%
  (delta -0.520833 percentage points). The task had 6/1920 passing rows,
  stop rate 0.97448, and stage-1 truncation 1024/1920 (53.33%). Sampled
  passing rows `65837030` and `65844281` were `12/12` and `7/7`; sampled
  wrong rows `65836987`, `65836988`, and `65836989` were `5/\\frac{1}{9}`,
  `2/2^{99}`, and `81/\\frac{1}{9}`, all `math_verify_false`. No checker or
  silent scorer error was observed; retain the low score/truncation as an
  anomaly for the Knowledge A/B audit and later retest.

### 2026-07-20 01:22 CST Browse format audit

- Browse task `26461` reached 35/830 completion rows: 30 `completed` and 5
  `incomplete` (`sample_index` 0, 1, 8, 11, 28). All five incomplete traces
  completed research first: 19--34 retrieved docids, 16--20 document reads,
  and 6--12 scalar search queries. The failure is isolated to the forced final
  stage, not retrieval or evidence persistence.
- The raw candidate-shard errors are output-format failures: missing required
  `answer`, unsupported legacy `function_call`, unsupported `tool_call_id` or
  `birth_date/death_date`, and one unterminated JSON string. Several malformed
  explanations mention a possible answer, but that text is deliberately not
  converted into `answer`; no answer is inferred from explanations, snippets,
  or documents.
- The strict parser therefore remains correct for this sample: only the
  model's explicit `answer` field may enter the official final-answer path.
  Do not relax the parser by guessing or by extracting a missing answer from
  explanation text. Continue the full 830 run to measure the natural invalid
  output rate; any later format change must be a schema/prompt fix followed by
  a fresh run.
- Seven formal lanes remain active. The six fresh formal score rows are still
  `10004`--`10009`; Browse has no score and is not counted in the 416 formal
  normal non-FC combinations.

### 2026-07-20 01:27 CST score batch

- Formal score `10010`, task `26449`, G1h-7.2B AIME24 CoT:
  `avg@64=0.0005208333` (0.0520833%), `stop_rate=0.9942708`. The latest
  same-model/same-benchmark historical row found was task `26092` at
  `0.0973958` (9.73958%), delta `-9.6875` percentage points. This is a
  severe capability/output anomaly, not a scorer acceptance.
- Task `26449` has 1/1920 passing eval rows. Sampled pass `65843376` is
  `25/25`; sampled wrong `65825418` is `437 + 23/73`, with
  `math_verify_false`. Stage-1 truncation is 410/1920 (21.35%), while the
  stage-2 extraction is 1908/1920 max-length (the dedicated final-answer
  budget); no checker was used. The result is queued for completion-level
  inspection and narrow Math strategy/output retest, without changing the
  benchmark TOML or inferring answers.
- Auxiliary score `10011` belongs to `answer_judge` task `26471`; it is not a
  fresh normal benchmark combination and is excluded from the `N/416` count.
- Formal primary fresh progress is now 7/416. Browse task `26461` remains on
  29573 with no official score; its earlier observed state was 35/830 and the
  live process is still running.

### 2026-07-20 01:34 CST live resource check

- The same Browse task advanced to 48/830: 42 `completed` and 6
  `incomplete`; no judge or score row exists. GPU3/29573 is actively generating.
- The scheduler log shows only normal remote-slot backpressure and continues to
  launch missing fresh tasks. No `failed`, `error`, `traceback`, OOM, or engine
  death marker was found. DB task status at this check is `Completed=24`,
  `Failed=1`, `Running=8`; the sole failure is the excluded stale Browse task
  `26454`.
- All four physical GPUs on 157 are at approximately 97--98% utilization and
  250W, with 335G free on the 1.9T evaluation volume (82% used).
- Re-probed all eight local vLLM listeners with the authenticated benchmark key:
  ports `19315`, `19316`, `19329`, `19330`, `29572`, `29573`, `29533`, and
  `29534` all returned HTTP 200 from `/v1/models`. The earlier unauthenticated
  401 was only a probe-key mismatch, not endpoint failure.

### 2026-07-20 01:36 CST score batch

- Formal score `10012`, task `26466`, G1h-2.9B HMMT-Feb25 CoT:
  `avg@64=0.0005208333` (0.0520833%), `stop_rate=0.9994792`. The latest
  same-model/same-benchmark historical row was task `26430` at `0.0083333`
  (0.83333%), delta `-0.78125` percentage points. Only 1/1920 eval rows
  passed; sampled pass `65867286` was `20/20`, while sampled wrong rows
  `65858261`, `65858304`, and `65858321` were `5/103`, `0/103`, and `10/103`,
  all `math_verify_false`.
- Stage-1 truncation was 841/1920 (43.80%). No checker or scorer anomaly was
  found. This score stays in the narrow Math output/strategy retest queue;
  the HMMT TOML, k, split, and verifier contract remain unchanged.
- Formal primary fresh progress is now 8/416. The 10011 answer_judge row and
  strategy A/B/C auxiliary tasks are excluded from this count.

### 2026-07-20 01:37 CST live checkpoint

- Browse task `26461` reached 53/830 completion rows. The DB completion rows
  are persisted as `Completed` by the runner; the nested Browse run result is
  the authoritative quality status and remains split between completed and
  incomplete traces. The 29573 process is still alive and GPU3 remains loaded.
- The read-only watch was started in screen `codex_g1h_watch_20260720`; it
  records DB task/completion counts, GPU utilization/power, and scheduler error
  markers every 60 seconds. Manual DB/completion/trace audits remain the source
  of decisions.

At 01:38 CST, task `26461` was 54/830 with 47 nested `completed` and 7
`incomplete` Browse runs. Formal primary fresh progress remained 8/416; DB task
status remained `Completed=27`, `Failed=1`, `Running=8`, with no new scheduler
error marker.

### 2026-07-20 02:00 CST backfilled report

- The watch log recorded Browse task `26461` at 88/830 completion rows and 33
  nested `completed` runs. DB task status was `Completed=33`, `Failed=1`,
  `Running=8`; the only failure remained excluded task `26454`.
- The seven formal runners and the Browse process remained alive. GPU0--3 were
  loaded; transient 0% readings on a lane were waiting/phase transitions, not
  endpoint loss. No scheduler error, traceback, OOM, or engine-death marker was
  present.

### 2026-07-20 02:30 CST backfilled report

- Browse task `26461` reached 132/830 rows and 45 nested `completed` runs.
  DB task status was `Completed=45`, `Failed=1`, `Running=8`.
- Fresh formal score rows continued to arrive. The two auxiliary
  `answer_judge` rows were kept out of normal progress; no old formatfinal row
  was counted.
- The 157 volume remained at 82% used with roughly 334--335G free. GPU power
  stayed near 250W when lanes were generating; no severe incident justified
  pausing a formal runner.

### 2026-07-20 02:39 CST backfilled score/trace report

- Browse task `26461` was 147/830 rows: 124 nested `completed`, 23
  `incomplete`. Every incomplete row had search/read activity (19--43
  retrieved docids, 10--20 document reads, 5--12 scalar queries). The failure
  split was `no valid candidate tool calls` (15) and no parse marker (8), all
  with final `no valid candidate tool calls`; no retrieval chain failure was
  found.
- New primary score audit:
  - `10013` task `26479`, 1.5B Algebra222: `avg@16=0.2559122` (25.5912%),
    909/3552 passed, stage-1 truncation 994/3552. Sampled pass `-21/-21`;
    sampled wrong `5.5/9.57`, `math_verify_false`. Historical task `26108`
    was 63.5698% (delta -37.9786 percentage points).
  - `10014` task `26462`, 2.9B Brumo25: `avg@64=0.0104167` (1.04167%),
    20/1920 passed, truncation 769/1920. Sampled pass `20/20`; wrong
    `2/2^99`, `math_verify_false`. Historical task `26228` was 6.14583%
    (delta -5.10417 points).
  - `10015` task `26463`, 1.5B Beyond-AIME: `avg@32=0.003125` (0.3125%),
    10/3200 passed, truncation 2385/3200. Sampled pass was the valid product
    expression for reference `12`; wrong `3031/3`, `math_verify_false`. No
    same-k historical row was available.
  - `10017` task `26483`, 1.5B Math-Odyssey: `avg@8=0.1114341` (11.1434%),
    345/3096 passed, truncation 2041/3096. Sampled pass `101/101`; wrong
    `1/16`, `math_verify_false`. No same-k historical row was available.
  - `10018` task `26452`, 13.3B AIME25: `avg@64=0.009375` (0.9375%),
    18/1920 passed, truncation 356/1920. Sampled pass `16/16`; wrong
    `-7/70`, `math_verify_false`. Historical task `26130` was 24.4271%
    (delta -23.4896 points).
  - `10019` task `26448`, 13.3B AIME24: `avg@64=0.00260417` (0.260417%),
    5/1920 passed, truncation 371/1920. Sampled pass `25/25`; wrong `4/73`,
    `math_verify_false`. Historical task `26096` was 23.8021% (delta
    -23.5417 points).
  - `10020` task `26474`, 7.2B AIME25: `avg@64=0.0125` (1.25%), 24/1920
    passed, truncation 451/1920. Sampled pass `60/60`; wrong `7/70`,
    `math_verify_false`. Historical task `26105` was 12.2917% (delta
    -11.0417 points).
  - `10021` task `26489`, 1.5B SimpleQA: `avg@4=0.00725` (0.725%), 29/4000
    passed, truncation 2564/4000. Sampled pass `Kanaloa/Kanaloa`; wrong
    `2019/120,000 euros`, `math_verify_false`. The dedicated TOML routes this
    free-response benchmark through the existing math verifier; no checker was
    used and no scorer override was made.
- Formal primary fresh progress is 16/416. Scores `10011` and `10016` are
  `answer_judge` auxiliary rows and remain excluded. The low Math results are
  recorded as output/truncation/strategy anomalies; no answer was corrected or
  inferred.

### 2026-07-20 07:16 CST access-outage report

- Current local time is 07:16 CST. The 157 host is not currently observable:
  `rwkv-8222` at `47.115.88.183:8222` returns `Connection refused`, and a
  direct SSH attempt to `192.168.0.157:22` times out. This is a network/SSH
  availability outage, not a DB query result.
- The last trustworthy server checkpoint was 02:44 CST: Browse task `26461`
  had 154 rows with nested status `130 completed / 24 incomplete`; formal
  primary fresh progress was 16/416; DB task status was `Completed=54`,
  `Failed=1` (excluded old task `26454`), `Running=8`.
- No current GPU, DB, scheduler, or score claim is made while the server is
  unreachable. Do not restart, kill, or release any runner/endpoint based only
  on this outage. Resume the DB/completion audit immediately after SSH returns.
- The watch continued writing its local remote log until the last reachable
  checkpoint, but it cannot create user-facing messages; this distinction is
  now explicit in the report procedure.

### 2026-07-20 07:47 CST public frontend/completions audit

- The public dashboard `https://shp6000.rwkvos.com/` and its JSON routes were
  reachable even though 157 SSH/tunnel was not. `/api/meta` returned HTTP 200
  with 1,649 score-index entries and the four G1h model choices; the default
  `/api/leaderboard` returned HTTP 200 with the expected Knowledge/Math/Coding/
  Instruction domains. This is a frontend/API observation, not a current DB
  or runner checkpoint.
- The eval table's `model_output` column is semantically wrong: the frontend
  binds it to `eval.answer`, which is the evaluator-extracted answer. The raw
  model output is only available in the lazy `/api/eval-context` response under
  `stages[].completion`. The `context` button itself shows only the first-stage
  prompt preview (SQL `LEFT(..., 240)`), not a completion. This explains why a
  correct single-stage sample can look as if no direct extraction happened.
- Live examples confirm the distinction without changing scores: task `26078`
  (1.5B ceval NoCoT) sample 1 has one stage with completion ` B`, evaluator
  answer `B`, reference `B`, and `is_passed=true`; task `26081` (1.5B ceval
  CoT) sample 0 has stage 1 reasoning (`max_length`) and stage 2 completion
  ` A`, with evaluator answer `A`. These are consistent raw/output pairs; the
  misleading labels and unlabeled prompt/completion blocks are presentation
  defects. Task `26317` returned no context for the sampled row and is kept as
  a missing-context observation, not inferred as a parser failure.
- The public response also showed G1h flower-template sentinels such as
  `User✿...✿Bot✿`. They are expected generation delimiters in the G1h prompt
  contract, but the dashboard display cleaner did not format them, so they
  looked like concatenated/corrupt messages. When `stages` existed, the modal
  additionally discarded `strategy_a`, `browsecomp_plus_run`, `agent_trace`,
  `events`, and other auxiliary context from the visible view.
- A narrow local display fix is now prepared: label `eval.answer` as
  `抽取答案（evaluator）`, label each prompt/completion separately, show the
  final non-empty completion beside the extracted/reference answers, preserve
  auxiliary context sections, and format flower sentinels only in the copied
  display payload. Evaluator logic, scorer behavior, raw DB context, and
  benchmark data are unchanged. Python dashboard tests pass (`4 passed`) and
  the client TypeScript check passes. The public site still serves the old
  bundle until the 157 deployment path is reachable; no deployment was
  attempted during the tunnel outage.
- A genuine output/extraction anomaly is visible in public task `26552`
  (`arxivmath_cot`): `eval.answer` is the beginning of the full reasoning
  (`>We need to compute...`) while `ref_answer` is
  `\\frac{r(2d-r-1)}{2}`. Its context has a 27,106-character stage 1 with
  `stop_reason=max_length` and a 420-character stage 2 also ending in
  `max_length`, with no final answer. This must be treated as a narrow
  completion/strategy retest candidate, not repaired in the UI or scorer.

### 2026-07-20 08:35 CST public score checkpoint

- The public frontend and JSON API still returned HTTP 200. Compared with the
  previous public snapshot, the visible dataset did not advance: `entry_count`
  remains `1649`, the visible fresh task set after task `26400` remains `51`,
  and the highest visible task remains `26570`. This is a frontend delta only;
  it does not prove the inaccessible DB or runner state is unchanged.
- Of those 51 fresh visible task rows, three are `answer_judge` auxiliary rows
  (`26471`, `26486`, `26546`) and must be excluded. The frontend therefore
  exposes 48 fresh formal benchmark rows: 44 Math, 2 Coding, and 2 Instruction
  Following; new Knowledge and Function Call rows are still 0 in this public
  index. The formal distribution is 6 rows at or above 50%, 19 below 1%, and
  2 exactly zero; scores use different benchmarks/k/metrics, so their simple
  mean is not a capability aggregate.
- Formal fresh rows by model are currently: 1.5B 21, 2.9B 18, 7.2B 5, and
  13.3B 4. Representative visible scores include 1.5B `mawps_cot` 52.9%,
  `asdiv_cot` 46.0%, and `horizonmath_cot` 0.0%; 2.9B `mawps_cot` 50.1%,
  `svamp_cot` 46.6%, and `math_odyssey_cot` 4.3%; 7.2B `brumo25_cot` 1.7%
  and `hmmt_feb25_cot` 0.2%; 13.3B `brumo25_cot` 6.6% and `aime24_cot`
  0.3%. These remain output/truncation and extraction-audit candidates, not
  clean capability conclusions.
- The visible BrowseComp rows are old G1g tasks (`470`, `471`, `474`, `475`,
  etc.); no G1h BrowseComp-Plus score is present in the public index. The
  frontend has therefore not yet published a trusted G1h BrowseComp-Plus
  result.

### 2026-07-20 09:16 CST public no-change checkpoint

- The public `/api/meta`, `/api/leaderboard`, and
  `/api/score-history/options` endpoints all returned HTTP 200. The response
  dates were 09:16 CST, so the frontend/API process itself is responding.
- Comparing the fresh leaderboard response with the 08:35 snapshot found no
  added task, removed task, or changed visible score. `entry_count` remains
  1,649, the visible task set remains unchanged, and the highest visible task
  remains `26570`.
- This proves that no new result has reached the public score index during
  this interval. It does not distinguish a stopped runner, a DB write stall,
  or a score-index refresh stall; that distinction requires DB task status,
  completion timestamps, or runner logs, which remain unavailable through the
  failed SSH/tunnel path.

### 2026-07-20 10:26 CST 2333 fallback resource audit

- The 2333 SSH path is reachable via `47.115.88.183:2333`, but it is not an
  available evaluation host. All eight RTX PRO 6000 GPUs are occupied at
  approximately 61--62 GiB / 97.9 GiB, 99% utilization, and 400 W / 400 W.
  `nvidia-smi pmon` shows Ray worker processes plus eight `VLLM::EngineCore`
  processes on the cards.
- The active workload belongs to
  `/home/caizus/Projects/MachineLearning/helicopter/workspaces/feat-maxrl`
  and was launched by `scripts/strict_train_run.py`; it is unrelated to this
  G1h evaluation. No process was stopped and no GPU was taken over.
- The requested `~/chase-rwkv-skills` and `~/GitHub/vllm-rwkv` paths are not
  present on 2333. No remote deployment or forwarding was attempted because
  doing so would interfere with the active workload.
- The existing local `rwkv-eval` database is not suitable as the fallback
  authority: it contains older mixed-project data (657 tasks, 451 scores,
  1,140,933 completions, and 790,519 eval rows; the newest task is from
  2026-06-29 and includes G1d/G1f/G1g). It was not modified.
- A separate database `rwkv-g1h-fallback-20260720` was created on the local
  PostgreSQL instance at `127.0.0.1:5433`, initialized from
  `scripts/schema.sql`, and verified by `rwkv-skills-scheduler bootstrap-db`.
  The local `.env` now points to this isolated database; it is empty and ready
  for the G1h fallback queue. No 157 data was overwritten.

### 2026-07-20 10:39 CST fallback preparation checkpoint

- Read-only probes of all eight configured local-forward ports (`19083`,
  `19315`, `19316`, `19329`, `19330`, `29572`, `29533`, `29534`) failed to
  connect. No runner was started against a dead inference endpoint.
- The current worktree initially had only 68 runnable non-FC specs per model
  because the 18 Knowledge JSONL files documented as present on 157 were
  absent locally. The existing fixed-revision repository preppers materialized
  all 18 locally. Exact validation now reports 18 files, 475,482 JSONL rows,
  510,991,346 bytes, and zero invalid/non-empty question-answer failures.
  The selector now resolves the complete 104 specs per model: 37 Math, 58
  Knowledge, 7 Coding, and 2 Instruction Following, or 416 fresh combinations.
- The local fallback database remains empty and schema-valid. The 2333 GPU
  recheck is unchanged: all eight cards are at roughly 61--62 GiB used,
  99--100% GPU utilization, and about 400 W, owned by the unrelated Ray/
  Helicopter workload. It remains unsafe and unauthorized to take over.

### 2026-07-20 10:41 CST queue preflight

- With the isolated DB empty and all local data present, the four-model matrix
  generated exactly 416 fresh runner commands (104 per model). All 416 use the
  vLLM `completions` protocol and none contains a checker flag. No task rows
  were inserted while the inference ports were unreachable, so the DB remains
  an honest `0/416` rather than a queue of doomed tasks.

- The public frontend still responds (`/api/meta` and `/api/leaderboard` HTTP
  200; `entry_count=1649`). Its displayed historical/latest G1h rows are not
  evidence for this isolated fresh run and no trusted new G1h BrowseComp-Plus
  score was added. The fallback DB remains the only authoritative source for
  any future local run.

### 2026-07-20 12:41 CST endpoint recovery and formal start

- SSH connectivity to both `rwkv-157` and `rwkv-8222` recovered. The existing
  157 services were preserved: 19083 infer-router plus 19315/19316 (1.5B) and
  19329/19330 (2.9B). Local forwarding was restored for those ports and for
  29572/29573/29533/29534 on 8222; a separate 19083 forward was then verified.
- All eight local `/v1/models` probes returned HTTP 200 with the expected model
  identity. One real `/v1/completions` probe per endpoint also succeeded. The
  2.9B and 7.2B probes can continue past the short prompt and report
  `finish_reason=length`; this is normal endpoint health behavior, not a
  scheduler failure.
- The four missing 8222 services were first found to fail during vLLM V2
  startup because the custom `get_cuda_view_from_cpu_tensor` operator had not
  been imported. The failed screens were stopped narrowly, and the services
  were relaunched with an import-only preload of `vllm._custom_ops`; no vLLM
  source, weights, or unrelated process was changed. All four ports now listen
  and the 8222 GPU memory footprint matches the loaded 7.2B/13.3B models.
- The scheduler profile doctor reports a valid isolated DB and seven explicit
  endpoint slots. The fresh queue remains exactly 416. At 12:41 it launched
  seven Math CoT tasks, one per physical endpoint, with `completions`, checker
  disabled, and the configured Math judge settings. DB status was `Running=7`;
  the Browse-only GPU3/29573 lane remained intentionally idle pending formal
  runner stability.

### 2026-07-20 12:57 CST Browse recovery on existing 157 chain

- The first local Browse launch was rejected by its own preflight because the
  official root is present on 157, not in the local fallback filesystem. It
  created no local Browse task and was not retried with `--skip-runtime-preflight`.
  A temporary local copy was stopped; the partial BM25 directory is not used.
- The authoritative 157 preflight was run against its existing
  `/tmp/rwkv-official-refs/BrowseComp-Plus`: 830 manifest records, official
  evaluator present, corpus present, BM25 index present, and `preflight ok=true`.
  Existing failed historical Browse tasks 26411/26420/26423/26424/26454/26461
  remain historical and are not counted as this run.
- A reverse SSH forward exposes local 29573 to 157 as `127.0.0.1:29573`, which
  reaches 8222:18073 without changing the formal 157 router or pausing formal
  runners. New task `26588` is running on 157 with G1h-7.2B, `bm25`,
  `rwkv_flower_json`, `parallel_candidate`, chunk conversion, 100-step cap,
  checker disabled, and judge concurrency 64.
- Browse is intentionally full 830, as required by the handoff; the earlier
  local 500-sample probe was never inserted into the DB and must not be used as
  a score. GPU3/29573 is now active while GPU0--2 continue formal evaluation.

### 2026-07-20 13:08 CST recovery without redoing completed work

- The isolated DB was checked before recovery. Tasks 1 and 2 have complete
  completion/eval sets (1,920 each) and scores; tasks 9/10/13/14 are the
  expected complete Math strategy-B/C temporary tasks. They were retained and
  not relaunched. Tasks 3--8 and 12 had partial completions but no score and
  their runner processes had been interrupted, so they were the only formal
  tasks resumed. Task 11 was confirmed as a duplicate AIME24 launch after task
  1, was stopped narrowly, and remains Failed with 448 partial completions; it
  is not counted as a completed result.
- The duplicate was traced to the scheduler's fresh-run reconciliation: an
  empty first DB poll was indistinguishable from an initialized score snapshot,
  so the first scores appearing later were not added to the current-session
  completion set. `DispatcherState` now tracks snapshot initialization
  explicitly. The dispatcher was restarted in `auto` recovery mode; it launched
  exactly the seven incomplete endpoint-affine tasks and did not relaunch task
  1 or 2.
- At this checkpoint the local DB is `Completed=6, Failed=1, Running=7`.
  All seven vLLM formal ports return HTTP 200 with the configured API key.
  The separate Browse task 26588 is still alive on 157 against the existing
  official 830-record root; GPU3 is active for Browse and GPUs 0--2 remain
  active for formal evaluation.

### 2026-07-20 13:11 CST first Browse completion audit

- Browse task 26588 has begun inserting completions: 11/830 are present and
  the task remains Running; no score row exists yet. The existing official
  corpus/index path is being used, and the 29573 model probe is HTTP 200.
- The first 11 samples are currently all judged incorrect by the task's inline
  result payload. Nine have completed research traces with extracted
  `Exact Answer` text but wrong answers; two ended incomplete with
  `no valid candidate tool calls`. This is a semantic retrieval/decision
  quality signal, not a formatfinal or score backfill, and no partial score is
  counted.
- The stored traces include the expected parallel-candidate stages, tool names,
  search/read counts, final output, and agent result. The sampled final-output
  extraction is structurally present; the two incomplete cases are recorded as
  parser/valid-tool-call failures for later completion inspection. No strategy
  change was made from this small sample, and no checker was enabled.

### 2026-07-20 13:23 CST full fresh restart with the G1h config root

- The earlier recovery attempt was stopped after verification showed that its
  old profile resolved the global/root Math TOMLs (the 4,096-token default),
  and the subsequent corrected AIME-only queue was also stopped because it was
  a narrow repair rather than the requested full matrix. Those partial rows
  remain audit evidence only; none is included in the formal fresh count.
- Six clearly legacy G1e/G1f TOMLs under `configs/g1h/` were removed narrowly:
  the four obsolete G1e/G1f `amc23.toml` copies and the two G1f performance
  service TOMLs. Formal G1h TOMLs, the separate 6,144-token `beyond_aime_6k`
  diagnostic experiment, and G1g run files were retained. No weights, data,
  `.venv`, unrelated configs, or runtime processes were broadly cleaned.
- The scheduler was dry-run and then started with
  `RWKV_BENCHMARK_CONFIG_ROOT=/home/chase/GitHub/rwkv-skills/configs/g1h`,
  the isolated local PostgreSQL database, and `--run-mode fresh`. Both checks
  report `待调度任务=416`; the real run occupies all seven formal endpoint
  slots and launched the first four-model AIME24/AIME25 wave.
- The seven launch commands use the endpoint-affine ports 19315, 19316,
  19329, 19330, 29572, 29533, and 29534. New DB tasks 22--28 all point to
  `configs/g1h` paths and their stored AIME stage-1 `max_new_tokens` is 8,192
  (stage 2 remains 128), confirming that the old global 4,096 setting is not
  being used. The config audit also confirms benchmark-specific G1h values:
  4,096 for the ordinary Math/Knowledge/Instruction TOMLs, 1,024 for the
  HumanEval/MBPP family, and 8,192 for the explicitly long-budget benchmarks;
  the 6,144 `beyond_aime_6k` file is not part of the 416 formal tasks.
- All eight local `/v1/models` endpoints, including 29573, returned HTTP 200
  with the required API key. Browse task 26588 remains running on the
  existing official 830-record 157 chain and is not counted until its complete
  evaluator/judge result is present.

### 2026-07-20 13:30--13:35 CST rapid-sampler incident and full restart

- A one-request diagnostic was sent to the busy 19315 chat endpoint with
  `top_k=40` while its formal AIME24 batch was already running with the
  endpoint's default private sampling values. The custom vLLM rapid sampler
  mixed these non-uniform request parameters and raised:
  `rapid-sampling with penalties only supports uniform scalar
  temperature/top_k/top_p/presence_penalty/repetition_penalty/penalty_decay`.
  Its EngineCore exited, 19315 returned 500/connection-reset, and the seven
  active first-wave tasks became Failed with partial rows. This was an
  operational probe failure, not a model-weight or context-length failure.
- The crashed 19315 service was recovered with its original model, CUDA,
  context, batching, and port arguments. It returned HTTP 200 after reload;
  the other six formal vLLM services were not restarted. Partial task rows
  from this incident are retained for diagnosis and excluded from all formal
  Completed counts.
- The inference client was corrected narrowly for the RWKV-specific vLLM
  deployment: `protocol=vllm` now forwards per-request `top_k`, repetition
  penalty, penalty decay, and token stop ids for ordinary and tool-call
  generation. The targeted backend tests pass (10/10). This is safe only with
  the endpoint-affine rule: one physical endpoint must finish one benchmark
  sampling configuration before the next one is assigned; no mixed-parameter
  probe is allowed.
- At 13:34 the same profile was restarted with `--run-mode fresh`, the G1h
  config root, and the isolated DB. It again reports `Pending=416` and has
  seven Running tasks (DB task ids 29--35). The first formal requests are
  succeeding after the 19315 reload; no score is counted yet.

### 2026-07-20 13:41--13:55 CST evaluation pause and sampling fix

- Per the user's pause instruction, the local formal scheduler and the
  authoritative Browse runner were stopped. The seven local tasks created by
  the last fresh wave (29--35) had partial completions but no scores; they
  were changed narrowly from stale `Running` to `Failed`, not deleted or
  counted. The isolated local DB is now `Completed=6, Failed=29, Running=0`.
  Browse task 26588 on 157 is also `Failed` after its runner was interrupted;
  its partial 830-record trace is retained and has no score.
- All eight vLLM services and their tunnels were left in place. No runner,
  scheduler, model weight, data, or database reset was performed.
- Root cause was confirmed in the local `~/GitHub/vllm-rwkv` source. The
  rapid CUDA kernel accepts scalar sampling arguments, but the worker can
  batch requests from different benchmark stages. With `penalty_decay` active,
  the worker skipped its earlier mixed-parameter check and passed vectors to
  `rapid_sample`, which raised inside EngineCore instead of isolating the
  rows.
- The local vLLM patch in
  `vllm/v1/sample/ops/topk_topp_sampler.py` groups mixed rows by the complete
  `(top_k, top_p, temperature, presence_penalty, repetition_penalty,
  penalty_decay)` signature and calls the scalar rapid kernel per group. This
  preserves the penalty buffer and decay updates; it does not silently use
  the native path, which lacks decay-state updates. The worker's obsolete
  forced rejection of mixed top-k/top-p/temperature vectors was removed from
  `vllm/v1/worker/gpu/sample/sampler.py`.
- The evaluation client now sends exact vLLM sampling semantics: zero
  presence/decay remain zero, `alpha_frequency=0` becomes vLLM's no-penalty
  `repetition_penalty=1.0`, configured nonzero values remain unchanged, and
  FC vLLM calls no longer send the same value as an unrelated
  `frequency_penalty`. `tests/test_infer_backend.py` passes 10/10 and both
  local code patches pass compile and whitespace checks. The vLLM pytest
  environment currently has no installed Torch; its attempted dependency
  download was stopped, so CUDA execution validation remains pending until a
  Torch-equipped vLLM environment is available.
- Evaluation must not resume until the patched vLLM source is built/deployed
  to the eight serving processes and a one-request mixed-parameter smoke test
  proves that the endpoint remains alive and each response records the
  intended sampling parameters. No score from the interrupted rows is a new
  formal score.
