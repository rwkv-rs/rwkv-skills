# Math / Free Response Score Alignment Status

范围：

- 当前项目：`/home/chase/GitHub/rwkv-skills`
- 旧项目：`/home/chase/rwkv-skills`
- 领域：Math / Free Response，包括 `free_response` 和 `free_response_judge`
- 对齐标准：同一权重、同一 benchmark、同一数据集 artifact、同一 prompt、同一采样参数、同一 judge 配置下，分数应保持不变或只出现很小波动。

## 结论

当前 Math 对齐基准以旧项目为准：

1. 只比较第一阶段 `completion1`，不把当前 GitHub 分支的 `completion2` / final-answer stage 纳入正式 Math 分数对齐。
2. 测评逻辑按旧项目：`math_verify.parse()` + `verify(strict=False)`，并保留 strategy A/B/C 的 repair / grouped metrics 逻辑。
3. `free_response_judge` 只把 math verify 未通过的样本送 LLM judge；math verify 已通过的样本不再交给 judge。
4. 当前 GitHub 分支默认两阶段生成是分数漂移风险，不是本轮对齐目标。
5. 当前分支与旧项目的正式 Math prompt/config 已经不完全一样，尤其是 `_templates.toml`、`aime24.toml`、`amc23.toml`、`gaokao2023en.toml`、`olympiadbench.toml`、model-specific AMC23。分数对齐前必须选旧项目配置作为 source of truth。

## Score-Sensitive Differences

### 1. 生成阶段：当前两阶段，旧项目单阶段

当前：

- `src/eval/maths/runner.py` 使用统一 runner，`--judge-mode exact|llm` 分别对应 `free_response` / `free_response_judge`。
- runner 总是解析 `cot` 和 `final` 两套配置，并传入 `FreeResponsePipeline.run(... final_answer_template=..., final_sampling=...)`。
- `src/eval/maths/pipeline.py` 在有 final stage 时会生成 `completion1` + `completion2`。
- `src/eval/metrics/free_response.py` 遇到 `completion2` 会用 `build_context_from_completions()` 拼回两阶段上下文后评分，并补齐未闭合的 `\boxed{...}`。

旧项目：

- `src/bin/eval_free_response.py` 和 `src/bin/eval_free_response_judge.py` 只传 `generation_sampling` 给旧 `FreeResponsePipeline`。
- `src/eval/evaluators/free_response.py` 只生成 `completion1`。
- 旧 `src/eval/metrics/free_response.py` 只从 `completion1` 取评分文本。

影响：

- 这是最高风险差异。哪怕同一权重、同一第一阶段 prompt、同一第一阶段采样，当前分支第二阶段 final answer 会改变最终评分输入，分数可能显著变化。
- 本轮对齐要求当前 runner 回到旧式 single-stage scoring：只生成并评分 `completion1`。
- 不应该为了对齐旧分数而让旧项目启用 two-stage；旧项目当前这套 single-stage + strategy A/B/C 是基准。

### 2. Prompt / sampling 正式配置不同

当前 `configs/_templates.toml` 的 Math prompt 是较短版本：

- 要求 one clean solution path
- 要求 final answer 放入 `\boxed{...}`
- 没有旧项目里的 anti-repeat / anti-choice / no-first-component 额外句子

旧项目 `configs/_templates.toml` 更长：

- 多了不要只返回第一个 component
- 多了不要使用 choice letter，除非题目要求 letter
- 多了不要 restart/repeat/enumerate/re-check/loop 的约束

同名 TOML 已确认有差异：

- `configs/_templates.toml`
- `configs/aime24.toml`
- `configs/amc23.toml`
- `configs/gaokao2023en.toml`
- `configs/olympiadbench.toml`
- `configs/rwkv7_g1e_13_3b_20260309_ctx8192/amc23.toml`
- `configs/rwkv7_g1e_7_2b_20260301_ctx8192/amc23.toml`
- `configs/rwkv7_g1f_13_3b_20260415_ctx8192/amc23.toml`
- `configs/rwkv7_g1f_7_2b_20260414_ctx8192/amc23.toml`

影响：

- 这些会直接改变生成分布或最终答案格式，不能用来判断代码是否对齐。
- 做分数对齐实验时，旧项目 Math TOML/prompt 是 source of truth。当前 GitHub 分支要用同一 prompt 和采样参数再比较。

### 3. Judge routing 基本一致，但注册来源不同

当前：

- `src/eval/benchmark_registry.py` 明确给部分 Math benchmark 绑定 `free_response_judge`。
- `src/eval/scheduler/jobs.py` 从 registry 生成 `LLM_JUDGE_DATASET_SLUGS` 和 `MATH_DATASET_SLUGS_FOR_FREE_RESPONSE`。

旧项目：

- `src/eval/scheduler/jobs.py` 通过 `available_free_answer_datasets()` 自动收集 Math 数据集。
- judge 列表硬编码为 `gsm8k_test`, `math_500_test`, `answer_judge_test`, `gaokao2023en_test`, `comp_math_24_25_test`, `minerva_math_test`, `amc23_test`, `olympiadbench_test`。

影响：

- 对共同 benchmark 来说，路由大体一致。
- 当前 registry 还显式包含 `polymath_all`、`simpleqa_verified`、`hle_all`、`answer_judge_test` 等；旧项目是否调度取决于 prepper registry 是否注册。
- 如果同一 benchmark 被一边 route 到 exact、另一边 route 到 judge，分数不可比。对齐前要固定 job：`free_response` 或 `free_response_judge`。

### 4. Scoring 基准：旧项目 completion1 + strategy A/B/C

旧项目基准：

- 只读取 `completion1`。
- `strategy_a`：直接用原始 completion。
- `strategy_b`：如果 `<think>` 未闭合，补 `</think>`。
- `strategy_c`：在 `strategy_b` 基础上，必要时追加 final-answer cue。
- exact scoring 使用 `math_verify.parse()` + `verify(strict=False)`。
- `free_response_judge` 只 judge math verify 没过的样本。
- primary score 仍以 `strategy_a` 为主，同时保留 `strategy_metrics` / `strategy_diagnostics` 给 Space 展示或排查。

当前差异：

- 当前 `metrics/free_response.py` 支持 `completion2`，并在两阶段 payload 上跳过旧的 strategy repair。
- 旧项目只按 `completion1` 做 strategy A/B/C repair。
- 当前 eval payload 带 `pass_index`，旧项目没有。

影响：

- 当前必须避免把 `completion2` 带入正式 Math 分数；否则 strategy A/B/C 语义会被绕开，分数会和旧项目不可比。
- 当前即使保留 `pass_index` 字段，只要 `pass_index=0` 且 scoring 只看 `completion1`，对旧项目分数影响应很小。

### 5. 生成 engine 不是完全相同

当前：

- runner 通过 `src.infer.backend` 构建 backend，可选 local classic、local lightning、remote OpenAI-compatible service。
- 当前 `src/infer/engine.py` 比旧项目新增 prefill chunk、约束、UTF-8 stop suffix、token delta 等能力。

旧项目：

- bin 入口直接 `ModelLoadConfig` + `InferenceEngine` 本地生成。
- engine 更简单，逐 token prefill/generate。

影响：

- 如果当前用远端或 lightning，不能和旧项目本地 engine 直接做分数对齐判断。
- 对齐实验应固定当前为 local classic，并固定同一 `batch_size`、同一 seed 规则、同一 stop token。
- 当前 `sample_repeat_seed(... pass_index=0)` 保持旧 seed 公式；普通 avg/pass 样本在 pass_index=0 时 seed 应接近旧项目。

## Formal Config Differences To Normalize First

在跑分前至少要消除这些差异：

1. `_templates.toml`
   - 当前短 prompt vs 旧项目长 prompt。

2. `aime24.toml`
   - 当前有显式 CoT prompt 和采样：`max_generate_tokens=16384`, `temperature=0.55`, `top_k=66`, `top_p=0.79`, `alpha_presence=0.14`, `alpha_frequency=0.01`, `stop_tokens=[0,261,24281]`。
   - 旧项目使用 `template = "math_cot_default"` 和 `template = "free_response_final_default"`。

3. `amc23.toml`
   - 当前有 `alpha_frequency=0.25`, `alpha_decay=0.99` 和自定义短 prompt / final prompt。
   - 旧项目只引用模板。

4. `gaokao2023en.toml`
   - 当前有 `alpha_frequency=0.25`, `alpha_decay=0.99` 和自定义 prompt / final prompt。
   - 旧项目只引用模板。

5. `olympiadbench.toml`
   - 当前有显式采样与自定义 prompt：`max_generate_tokens=4096`, `temperature=0.3`, `top_k=500`, `top_p=0.4`, `alpha_presence=1.0`, `alpha_frequency=0.3`。
   - 旧项目使用模板，并且 final prompt 文案不同。

6. model-specific AMC23
   - 当前 model-specific prompt 更短。
   - 旧项目 model-specific prompt 包含 boxed、完整 component、不要 choice letter 等额外约束。

## Recommended Alignment Procedure

先不要大改架构。按这个顺序验证分数是否可对齐：

1. 选 source of truth
   - 以 `/home/chase/rwkv-skills/configs` 的 Math prompt/sampling 为准。
   - 当前 GitHub 分支只用于改到同一 completion/scoring 行为后比较分数。

2. 固定 stage 语义
   - 当前 Math runner 需要旧式 single-stage mode。
   - 禁用 final answer stage。
   - 只生成和评分 `completion1`。

3. 固定 job
   - 同一 benchmark 必须都走 `free_response` 或都走 `free_response_judge`。
   - judge benchmark 必须固定同一 judge model、base URL、temperature 0、max tokens、prompt。

4. 固定 engine
   - 优先本地 classic engine 对齐，不混用 remote / lightning。
   - 固定同一 `batch_size`、stop tokens、ban tokens、no penalty ids、pad zero。

5. 小样本验收
   - 每个有差异的 benchmark 先跑 10 到 30 题同一 sample indices。
   - 比较 `completion1` 文本、strategy A/B/C extracted answer、math_verify pass、judge pass。
   - 只有 completion 和评分输入一致后，再跑完整 benchmark。

## Candidate Code Changes For Score Alignment

这些是下一位重构人员最可能要做的最小改动：

1. 给 `src/eval/maths/runner.py` 增加旧式 single-stage 兼容开关，例如 `--math-stage-mode single|two_stage`，正式 Math 复现旧项目时默认/强制用 `single`。
2. single-stage 模式下只调用 `resolve_generation_sampling()`，不要读取/传入 `final_answer_template` 和 `final_sampling`。
3. single-stage 模式下让 `src/eval/maths/pipeline.py` 只产出 `completion1`，不要产出 `completion2`。
4. single-stage 模式下保持 `src/eval/metrics/free_response.py` 的旧项目 scoring path：`completion1` + strategy A/B/C + math verify + optional judge fallback。
5. 写一个 config parity 检查脚本/测试，逐项比较旧项目 Math TOML 解析后的 prompt 和 sampling config。
6. 写一个 tiny dataset golden test：同一 fake backend 输出下，当前 single-stage eval 的 `strategy_metrics`、primary score、judge fallback 与旧项目一致。

## Confirmed Not Done

- 未改 runner/config 代码。
- 未跑 benchmark。
- 未做分数实测。
- 本文只记录 Math / Free Response 中会影响分数对齐的差异。
