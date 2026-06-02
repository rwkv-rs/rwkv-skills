# Free-response 生成停止与 math_verify 评分改造方案

日期：2026-06-01

## 分支现状与建议分支名

当前工作分支：

- `fullgen-mathverify`

当前远端实际分支：

- `origin/agent-bench`
- `origin/experiment/legacy-eval-alignment`
- `origin/main`
- `origin/nano-vllm-rwkv-engine`

`refs/remotes/origin/HEAD` 只是远端 HEAD 伪引用，不应按实际分支计数。

因此，实际本地+远端分支引用数为 `6` 个。如果把 `origin/HEAD` 也按 git ref 原样算进去，则是 `7` 个 ref。按去重后的分支名看，共 `4` 个：`agent-bench`、`nano-vllm-rwkv-engine`、`experiment/legacy-eval-alignment`、`main`。

建议新分支名：

`fix/free-response-full-generation-mathverify`

原因：这个名字覆盖了本次核心变化：free-response 不再两阶段强行续写 final prompt，而是完整生成；评分切换到 math_verify；同时不把范围误导成只修 CoT prompt。

当前工作区已切到 `fullgen-mathverify`。如果继续在该分支实现，应避免把 unrelated function-calling/frontend/long-doc 文件纳入本次改造。

## 目标行为

free-response 的 CoT 与未来可能的 no-CoT/direct 模式都应遵守同一条生成规则：

1. 一直生成，直到生成 token `0`，或生成文本中包含 `"\nUser:"`。
2. 如果没有满足条件 1，则生成到 `max_generate_tokens` 后停止。
3. 每条 completion 必须记录停止原因：`stop_condition` 或 `max_tokens`。建议额外记录 `stop_detail`：`token_0`、`user_sentinel`、`max_tokens`。
4. `stop_reason == max_tokens` 的样本视为截断。需要在 completion context 和 score metrics 中统计截断率。
5. CoT 不能在 `</think>` 后强行拼接 `Therefore...` 再让模型生成 boxed 答案。正常完整输出应直接交给 math_verify 评分。

建议默认实现方法 C，同时保留 A/B/C 为配置或 CLI 选项，便于复现实验：

- 方法 A：所有样本直接用 math_verify 对完整输出评分，不修补。
- 方法 B：如果有 `<think>` 但没有 `</think>`，对评分输入补 `</think>`，再追加 `Therefore, the final answer is ` 后交给 math_verify；其它样本走 A。
- 方法 C：先执行 B；否则如果仍是截断，说明可能已经结束 think 但 answer 未完整生成，对评分输入追加 `\nTherefore, the final answer is ` 后交给 math_verify；其它样本走 A。

注意：B/C 的“插入 Therefore”建议先实现为评分输入文本修补，而不是再次触发第二阶段模型生成。否则会重新引入本次要移除的两阶段生成路径。

## 评分口径与必须统计指标

本次改造后的 DB 可见分数应保持当前 score 形状，不额外发明一组 `raw_math_verify_accuracy`、`final_accuracy` 之类的新主分数。需要存在的是三类信息：

1. 截断率：`stop_rate`。
2. 真实分数：方法 A，不使用修补策略，直接对完整生成文本评分。
3. 使用策略后的分数：方法 B / 方法 C 修补后的评分。

每一个“分数口径”都是一组 score metrics。每组内部继续使用当前已经存在的指标名，例如：

- `exact_accuracy`
- `judge_accuracy`
- `pass@k`
- `avg@k`

改造后只是在每组 metrics 里增加一个 `stop_rate`。一共三组评分口径：`strategy_a`、`strategy_b`、`strategy_c`。其中 `strategy_a` 就是 raw/真实直评，不再额外存一份 `raw`。

示例结构：

```json
{
  "strategy_a": {
    "exact_accuracy": 0.42,
    "judge_accuracy": 0.45,
    "avg@1": 0.42,
    "stop_rate": 0.08
  },
  "strategy_b": {
    "exact_accuracy": 0.44,
    "judge_accuracy": 0.47,
    "avg@1": 0.44,
    "stop_rate": 0.08
  },
  "strategy_c": {
    "exact_accuracy": 0.45,
    "judge_accuracy": 0.48,
    "avg@1": 0.45,
    "stop_rate": 0.08
  }
}
```

实现上应把三组放在同一条 `scores.metrics` JSONB 中，避免多条 `scores` row 被 latest-score 查询折叠或覆盖。

### 单条 completion 的评分流程

1. 读取单阶段生成结果：
   - `prompt1`: 实际输入 prompt。
   - `completion1`: 模型一次生成的完整文本。
   - `stop_reason1`: `stop_condition` 或 `max_tokens`。
   - `stats.stop_detail`: `token_0`、`user_sentinel`、`max_tokens`。
   - `stats.truncated`: `stop_reason1 == "max_tokens"`。
2. 构造评分文本：
   - 从 `completion1` 取生成文本。
   - 如果包含 `"\nUser:"`，只使用第一次出现前的文本评分。
   - token `0` 不应进入 decoded completion。
3. 方法 A / 真实分数口径：
   - 直接对评分文本调用 math_verify / judge。
   - 输出一组与当前完全同形状的 score metrics。
4. 方法 B：
   - 如果有 `<think>` 但没有 `</think>`，就在评分文本末尾补 `</think>`，再插入 `Therefore, the final answer is ` 后评分。
   - 其它情况与方法 A 一样。
5. 方法 C：
   - 先执行方法 B 的未闭合 think 修补。
   - 否则，如果仍然属于截断，且属于已经生成完 think 但没生成完 answer 的情况，就插入 `\nTherefore, the final answer is ` 后评分。
   - 其它情况与方法 A 一样。
6. 每个方法都重新走同一套 math_verify / judge / pass@k / avg@k 计算，并输出一组同形状 score metrics。
7. judge fallback：
   - 每个方法组都单独执行 judge fallback，因为不同方法的评分输入可能不同。
   - judge 是本地/自有 vLLM 服务，API 成本不是本方案的限制因素。
   - 可对完全相同的 judge 输入做去重缓存，但不能因为省请求而跳过某个方法组。
8. `stop_rate`：
   - `stop_rate = stop_reason == "max_tokens"` 的 completion 数 / completion 总数。
   - 同一批 completions 的三组方法使用同一个 `stop_rate`。
   - `stop_detail=token_0` 和 `stop_detail=user_sentinel` 可以留在 completion context 或 task detail 里做排查，不作为 DB 可见主分数。

## 当前实现问题

### 1. 生成引擎停止条件不符合新规则

文件：`src/infer/engine.py`

- `src/infer/engine.py:191-195` 从 `SamplingConfig.stop_tokens` 读取停止 token。
- `src/infer/engine.py:272-289` 当前只区分 `stop_token` 与 `max_length`，并把“命中文本 stop suffix”也归为 `stop_token`。
- `src/infer/engine.py:277-280` 已有 `prompt_stop_suffixes` 能检测生成文本包含指定字符串，但 free-response 当前没有使用。
- `src/infer/engine.py:423-426` `_matches_stop_suffix` 判断的是 `suffix in text`，可用于 `"\nUser:"`。

需要修改：

- 对 free-response 生成强制使用 `stop_tokens=(0,)`，不要再用当前 math 模板里的 `261`、`24281` 作为停止 token。
- 调用 `InferenceEngine.generate(...)` 时传入每个 prompt 的 `prompt_stop_suffixes=[("\nUser:",)]`。
- `GenerationOutput` 需要能记录更细的停止详情。可在 `src/infer/sampling.py:32-40` 的 `GenerationOutput` 增加可选字段：
  - `finish_reason`: `stop_condition` 或 `max_tokens`
  - `finish_detail`: `token_0`、`user_sentinel`、`max_tokens`
  - `truncated`: `bool`
- 对已有其它 evaluator 保持兼容：如果不想改全局语义，可先让 engine 保持旧 `finish_reason`，在 free-response pipeline 层把 `stop_token`/`max_length` 归一化为新字段。但更干净的做法是在 engine 层直接区分。

### 2. 默认采样模板会提前在 `</think>` token 停止

文件：`src/infer/sampling.py`

- `src/infer/sampling.py:14-23` 默认 `stop_tokens=(0, 261, 24281)`。

文件：`configs/_templates.toml`

- `configs/_templates.toml:57-72` `free_response_cot_default` 仍使用 `stop_tokens = [0, 261, 24281]`。
- `configs/_templates.toml:74-87` `free_response_final_default` 仍定义了第二阶段 final prompt 与 final sampling。
- `configs/_templates.toml:1-16` `math_cot_default` 也使用 `stop_tokens = [0, 261, 24281]`。

需要修改：

- free-response 相关生成阶段的停止 token 应改为只允许 `[0]`。
- `"\nUser:"` 不建议做成 token stop，因为它是文本哨兵，应该走 `prompt_stop_suffixes`。
- `free_response_final_default` 后续应仅作为 legacy 兼容或删除使用路径，不能再参与正式 free-response score-bearing 生成。

### 3. FreeResponsePipeline 当前是错误的两阶段生成

文件：`src/eval/evaluators/free_response.py`

- `src/eval/evaluators/free_response.py:22-27` 默认 prompt 是 `Assistant: <think`，默认 final prompt 是 `<Q><COT>\nTherefore, the answer is \(\boxed{`。
- `src/eval/evaluators/free_response.py:121-170` 第一阶段生成 CoT，并把输出记录为 `_stage = "cot"`。
- `src/eval/evaluators/free_response.py:165-170` 构造第二阶段 prompt：`final_answer_template.replace("<Q>", cot_prompts[...]).replace("<COT>", cot_text)`。
- `src/eval/evaluators/free_response.py:172-217` 第二阶段再次生成答案，并把第二阶段结果作为 `_stage = "answer"` 写入 DB。

需要修改：

- 新增或重构成单阶段 full-generation 路径：
  - CoT 模式 prompt 仍可从 `cot_prompt_template` 来，例如 `User: ...\n\nAssistant: <think`。
  - no-CoT/direct 模式 prompt 应从 `direct_prompt_template` 或新增 `DEFAULT_DIRECT_PROMPT` 来，例如 `User: <Q>\n\nAssistant:`。
  - 只调用一次 `engine.generate`。
  - completion 存完整模型输出，不再拆成 cot stage 和 final stage。
- `SampleRecord` 建议只写一段 stage：
  - `prompt1`: 实际输入 prompt
  - `completion1`: 完整生成文本，建议在 `\nUser:` 前截断用于评分，raw 是否保留需要权衡 DB 体积
  - `stop_reason1`: `stop_condition` 或 `max_tokens`
  - `stats`: `{ "truncated": bool, "stop_detail": "...", "generated_tokens": n }`
- 新数据库不需要兼容旧两阶段 free-response payload；正式结果只接受新单阶段 full-generation payload。

### 4. 入口脚本仍强绑定 cot/final 两套 sampling

文件：`src/bin/eval_free_response.py`

- `src/bin/eval_free_response.py:135-148` 解析 `cot_prompt_template` 和 `final_prompt_template`。
- `src/bin/eval_free_response.py:167-182` 同时解析 `cot_sampling` 与 `final_sampling`。
- `src/bin/eval_free_response.py:277-290` 调用 `pipeline.run(... final_answer_template=..., cot_sampling=..., final_sampling=...)`。
- `src/bin/eval_free_response.py:344-353` 写分数 payload 时可继续使用 `metrics_payload` 和 `task_details` 承载截断率。

文件：`src/bin/eval_free_response_judge.py`

- `src/bin/eval_free_response_judge.py:154-167` 同样解析 final prompt。
- `src/bin/eval_free_response_judge.py:193-208` 同样解析两套 sampling。
- `src/bin/eval_free_response_judge.py:315-327` 调用两阶段 pipeline。
- `src/bin/eval_free_response_judge.py:373-390` 写 judge metrics，可加入 truncation metrics 与 stop counts。

文件：`src/bin/param_search_free_response.py`

- `src/bin/param_search_free_response.py:154-169` 参数搜索也解析 final sampling。
- `src/bin/param_search_free_response.py:218-227` 调用同一个两阶段 `FreeResponsePipeline.run`。

文件：`src/bin/param_search_free_response_judge.py`

- `src/bin/param_search_free_response_judge.py:176-191` 参数搜索 judge 也解析 final sampling。
- `src/bin/param_search_free_response_judge.py:282-292` 调用两阶段 pipeline。

需要修改：

- 对正式 free-response 与 free-response_judge，入口只需要一套 generation sampling。
- 为兼容配置，可以先把 `cot_sampling` 当 full-generation sampling 使用，并忽略 `final_sampling`。
- CLI 上建议保留 `--cot-max-tokens` 的兼容别名，但内部重命名为 `--max-tokens` 或 `generation_max_tokens`。
- param-search 的 grid 应搜索 full-generation sampling，而不是只搜索旧 cot 阶段 sampling。

### 5. 当前 free-response 评分不是 math_verify

文件：`src/eval/metrics/free_response.py`

- `src/eval/metrics/free_response.py:74-79` `_strip_thinking_for_answer` 会删除闭合 think block；如果 `<think>` 未闭合，则直接返回 `<think>` 前文本，这对新规则不合适。
- `src/eval/metrics/free_response.py:82-120` 当前依赖 final stage prompt 是否以 `{`、`\(`、`[` 等结尾来抽取答案。
- `src/eval/metrics/free_response.py:123-124` `_is_exact_match` 只是规范化字符串后精确匹配。
- `src/eval/metrics/free_response.py:315-341` 主循环拿最后一个 stage 的 completion 做抽取和 exact match。
- `src/eval/metrics/free_response.py:343-379` judge 只在 exact false 时调用，后续仍可保留。

需要修改：

- 增加 `math-verify` 依赖。当前 `pyproject.toml:7-62` 未声明 `math-verify`。
- 在 `evaluate_free_response` 中：
  - 从 completion payload 组装完整生成文本，而不是只读最后 stage。
  - 先按 `\nUser:` 截断评分文本。
  - 按 A/B/C 策略得到 `math_verify_input_text`。
  - 用 math_verify 对 `reference` 和 `math_verify_input_text` 进行解析与 verify。参考实现：
    - `gold = parse(f"$\\boxed{{{reference}}}$")`
    - `pred = parse(math_verify_input_text)`
    - `passed = bool(pred and verify(gold, pred, strict=False))`
  - 预测答案提取交给 math_verify 按最后答案/最终候选处理，不再手写旧的 final-stage boxed 截取逻辑。
  - exact/mathematical pass 成功时不再调用 LLM judge。
  - LLM judge 仍可作为 math_verify 未通过时的补充，并且每个方法组分别运行 fallback。
- DB 可见 metrics 不新增一批独立主指标，只按“真实分数组”和“策略分数组”输出当前同形状分数：
  - 每组保留当前已有的 `exact_accuracy` / `judge_accuracy` / `pass@k` / `avg@k`。
  - 每组增加 `stop_rate`。
  - `stop_detail`、parse 状态、策略名等排查信息放 completion context 或 task detail，不作为主分数字段。
- `answer` 字段建议存一个短 display answer。可以先存 math_verify parse 结果的字符串；parse 失败则存截断后的完整输出前若干字符，完整上下文仍在 completion context 中。

### 6. DB 与 artifact schema 基本能承载，但需要统一字段

文件：`src/eval/evaluators/common.py`

- `src/eval/evaluators/common.py:27-55` `StageRecord` 只有 `prompt`、`completion`、`stop_reason`，`SampleRecord` 会把它们展开为 `promptN/completionN/stop_reasonN`。

文件：`src/db/eval_db_service.py`

- `src/db/eval_db_service.py:453-469` completion payload 写入 DB。
- `src/db/eval_db_service.py:471-497` batch completion 写入 DB。
- `src/db/eval_db_service.py:992-1015` `_build_completion_context` 已保存 `stages`、`events`、`stats`、`final_answer`、`sampling_config`。
- `src/db/eval_db_service.py:697-714` 读 completion 时会把 context 里的 `stats` 和 `stop_reason` 还原到 payload。

文件：`src/db/orm.py`

- `src/db/orm.py:220-238` completions.context 是 JSONB，可直接承载新增 `stats`。
- `src/db/orm.py:262-278` scores.metrics 也是 JSONB，可直接承载截断率。
- `src/db/orm.py:242-259` evals 当前没有方法组字段；如果要保存 A/B/C 三套逐样本 pass/fail，需要增加分组字段或等价 JSON 字段。

需要修改：

- 本次按新建数据库处理，不需要旧库迁移、旧 eval 回填或旧两阶段结果重算兼容。
- 新 schema 需要直接支持 completion 停止详情、三组 `scores.metrics`、以及 eval 分组字段。
- eval 表需要支持每个 completion 对应多组逐样本 pass/fail。建议增加 `eval_group` 或 `strategy` 字段，取值 `strategy_a` / `strategy_b` / `strategy_c`，并把去重键从 `completion_id` 改为 `(completion_id, eval_group)`。
- 需要扩展 `SampleRecord` 或在 free-response pipeline 生成 payload 后手动加入：
  - `stats.truncated`
  - `stats.stop_detail`
  - `stats.generated_token_count`
- 修补策略是评分时行为，不应写成“模型生成事实”。DB scores 表里应体现为 `scores.metrics.strategy_a` / `strategy_b` / `strategy_c` 三组同形状 score metrics。
- `make_eval_payload` 需要增加方法组参数，并让 `ingest_eval_payloads` / repo 插入逻辑允许同一 completion 写入三条 eval 结果，便于后续逐样本对比分析。
- Space/UI 保持现有展示方式：主表仍显示当前分数形状，鼠标移动到分数位置时展示三组 metrics 和 `stop_rate`。

### 7. Scheduler job 选择不需要大改，但 no-CoT free-response 目前不存在

文件：`src/eval/scheduler/jobs.py`

- `src/eval/scheduler/jobs.py:248-269` 当前 `free_response` 和 `free_response_judge` 都声明 `is_cot=True`。
- `src/eval/scheduler/jobs.py:425-433` `detect_job_from_dataset(dataset_slug, is_cot)` 根据 `is_cot` 选择 job。

需要修改：

- 如果只修当前正式数学队列，scheduler job 名可以暂时不变，仍是 `free_response` / `free_response_judge`，但内部改成 full-generation CoT。
- 如果要真正支持 no-CoT free-response，需要新增 job，例如：
  - `free_response_plain`
  - `free_response_judge_plain`
  - `is_cot=False`
  - 对应入口可复用同一个 pipeline，但传 direct prompt。
- 不要把现有 multiple-choice plain 路径误改成生成式 no-CoT；`src/eval/evaluators/multi_choice.py:89-146` 当前是 logits-only 评分，不属于 math_verify free-response 范围。

### 8. 配置文件中 final prompt 应停止参与正式测评

文件：

- `configs/_templates.toml:74-87`
- `configs/aime25.toml:7-11`
- `configs/gsm8k.toml:7-11`
- `configs/college_math.toml:10-13`
- `configs/olympiadbench.toml:10-13`
- `configs/rwkv7_g1f_13_3b_20260415_ctx8192/amc23.toml:14-15`
- `configs/rwkv7_g1f_7_2b_20260414_ctx8192/amc23.toml:14-15`
- `configs/rwkv7_g1e_13_3b_20260309_ctx8192/amc23.toml:13-14`
- `configs/rwkv7_g1e_7_2b_20260301_ctx8192/amc23.toml:14-15`

需要修改：

- 对正式 free-response，新入口不要再读取 `[final]` 的 `final_prompt_template` 来生成。
- 可以暂时保留 TOML 字段，避免破坏旧脚本，但文档和代码应标记 legacy。
- free-response `[cot]` 或新 `[generation]` 的 `stop_tokens` 应明确为 `[0]`。

### 9. 测试需要更新

当前测试固定了旧行为：

- `tests/test_free_response_answer_matching.py:31-42` 测试 final prompt boxed brace 抽取。
- `tests/test_free_response_answer_matching.py:87-123` 测试从第二阶段 prompt/completion 抽取 judge 输入。
- `tests/test_free_response_judge_config.py:145-149` 要求 `olympiadbench` final prompt 以 `\(\boxed{` 结尾。

需要新增或改写测试：

- engine 级测试：
  - token `0` 停止时记录 `stop_condition/token_0`。
  - 文本包含 `\nUser:` 停止时记录 `stop_condition/user_sentinel`。
  - 超过 `max_generate_tokens` 时记录 `max_tokens` 且 `truncated=True`。
- free-response pipeline 测试：
  - CoT prompt 只调用一次 generation。
  - completion payload 只有一个正式 answer stage。
  - 不再产生 `<Q><COT>\nTherefore...` final prompt。
- metrics 测试：
  - 方法 A：完整输出含 `\boxed{...}` 可被 math_verify 判对。
  - 方法 B：有 `<think>` 无 `</think>` 时评分输入被修补。
  - 方法 C：已生成完 think 但 answer 截断时追加 `\nTherefore...`。
  - 思维链里有中间 boxed、最终答案有正确 boxed 时，math_verify 应按最终候选判定；用回归测试固定该行为。
  - 多个 boxed、多次改答案、无 boxed 但有自然语言答案时，应保留 parse/verify 错误详情供排查。
  - reference parse 失败时不能把样本静默记成模型错误，应进入 fallback 或错误详情，不新增 DB 可见主分数。
  - `stop_rate` 统计正确，并被加入 `strategy_a` / `strategy_b` / `strategy_c` 三组。
  - 三组都输出当前同形状 metrics：`exact_accuracy` / `judge_accuracy` / `pass@k` / `avg@k`。
  - 每组 math_verify 通过时不调用 LLM judge；未通过时才调用本组 judge fallback。
- eval DB 测试：
  - 同一个 completion 可以写入 `strategy_a` / `strategy_b` / `strategy_c` 三条逐样本 eval。
  - eval 去重键使用 `(completion_id, eval_group)`，不能再只按 `completion_id` 去重。
- config 测试：
  - free-response 正式配置不依赖 final prompt。
  - `stop_tokens` 对 free-response 为 `[0]`。

## 推荐改造顺序

1. 新分支：`fix/free-response-full-generation-mathverify`。
2. 加依赖：`pyproject.toml` / `uv.lock` 加 `math-verify`。
3. 先改 engine 停止详情，保证不破坏其它 evaluator。
4. 改 `FreeResponsePipeline` 为单阶段 full generation，不保留旧两阶段 free-response payload 写入路径。
5. 增加 eval 方法组字段或等价结构，让同一个 completion 可保存 A/B/C 三套逐样本 pass/fail。
6. 改 `evaluate_free_response` 使用 math_verify，并实现 A/B/C repair mode。
7. 更新入口脚本与参数搜索脚本，移除正式路径中的 final generation。
8. 更新 configs，让 free-response generation 只用 token `0` + 文本 `\nUser:` 停止。
9. 更新 Space/UI 的 hover 展示，让分数位置能看到 `strategy_a` / `strategy_b` / `strategy_c` 三组和 `stop_rate`，主表交互保持现状。
10. 更新测试。
11. 先跑单元测试，不跑正式 benchmark。正式 score-bearing run 需要后续按 AGENTS 规则单独准备 full-dataset/full-matrix 命令并等待审批。

## 已确认实现约束与剩余风险

- B/C 中“插入 Therefore 并提取答案”只修补 math_verify 输入，不允许触发第二阶段模型生成。
- math_verify 对部分自然语言、集合、区间、单位答案的 parse 可能失败。需要保留 LLM judge fallback；parse 细节作为排查信息记录，不作为 DB 可见主分数。
- 本次会新建数据库，旧 DB completion 不进入正式重算范围；实现不需要为旧两阶段 free-response 结果做迁移或回填。
- math_verify 预测解析采用 `pred = parse(scoring_text)`，按其最终候选提取逻辑处理完整输出。必须用单测覆盖 `<think>` 内中间 boxed、多个 boxed、先错后改等场景。
- 方法 C 按既定规则处理：先修补未闭合 `<think>`；否则如果仍然属于截断，且属于生成完 think 但没生成完 answer 的情况，再插入 `\nTherefore...`。如果截断发生在思维链中间，不能走第二个修补分支。
- `\nUser:` 只应作为生成文本里的 sentinel；不要检查 prompt 本身。命中后评分文本应截断到 sentinel 前，但 raw completion 可按 DB 体积策略保留。
- token `0` 和文本 sentinel 同时相关时，以实际先触发的停止条件为准。采样到 token `0` 的那一步不应把 token `0` 解码进 completion。
- `max_generate_tokens` 的边界应定义为生成 token 数 `>= max_generate_tokens` 停止，避免 off-by-one 导致截断率不稳定。
- 当前工作区已有未提交/未跟踪文件。实现时应避免把 unrelated function-calling/frontend/long-doc 改动纳入本次改造。

## 待确认细节清单

这些不是方向性疑问，而是落代码前需要固定的细节。

### 1. eval 表分组字段

建议字段名使用 `eval_group`，取值：

- `strategy_a`
- `strategy_b`
- `strategy_c`

需要确认：

- 是否增加唯一约束或等价去重逻辑：`(completions_id, eval_group)`。
- `fetch_existing_eval_completion_ids` 这类只按 `completion_id` 去重的函数，需要改成按 `(completion_id, eval_group)` 判断。

### 2. scores.metrics 主展示组

三组都放在同一条 `scores.metrics` JSONB 中已经确定。还需要固定 Space 主表默认显示哪组：

- 推荐主表默认显示 `strategy_a`，因为它等于真实直评，和当前“测出来的分数”语义最接近。
- 鼠标 hover 时展示 `strategy_a` / `strategy_b` / `strategy_c` 三组完整 metrics 和 `stop_rate`。

如果希望主表默认显示修补后分数，应明确改成 `strategy_c`，否则实现时按 `strategy_a`。

### 3. 方法 C 的程序判定

语义已经确定：C 先修补未闭合 `<think>`；否则如果截断发生在“think 已生成完但 answer 没生成完”的阶段，就插入 `\nTherefore, the final answer is `。

需要实现成可测试的判定：

- 有 `<think>` 且无 `</think>`：走 B 分支，不走 C 的 answer 截断分支。
- 有闭合 `</think>` 且 `stop_reason=max_tokens`：可认为 think 已完成，C 可追加 `\nTherefore...`。
- 没有 `<think>` 的 direct/no-CoT 输出且 `stop_reason=max_tokens`：需要确认是否也按“answer 截断”处理。建议是 yes，因为没有 think 区域时生成阶段天然就是 answer 区域。

### 4. 修补插入文本

当前文档使用：

```text
Therefore, the final answer is
```

和：

```text
\nTherefore, the final answer is
```

需要确认是否固定这个字符串，还是使用旧模板里的：

```text
Therefore, the answer is \(\boxed{
```

建议不要使用旧 boxed-prefix 模板，因为会重新引入“强迫 boxed 提取”的偏置；只给 math_verify 一个自然语言 final-answer cue。

### 5. completion 存储文本

生成时遇到 `\nUser:` 应停止；评分时一定只用 `\nUser:` 前的文本。

需要确认 DB `completion1` 存哪一种：

- 推荐存已经截断到 `\nUser:` 前的 completion，和示例脚本一致。
- 如果要保留 raw sentinel 后文本，应放到 completion context 的 debug 字段里，不参与评分，不作为常规展示文本。

### 6. stop_detail 命名

`stop_rate` 只依赖 `stop_reason == "max_tokens"`，但 completion context 里还会记录细节。

需要固定命名：

- 文档固定使用：`token_0`、`user_sentinel`、`max_tokens`。
- 示例脚本使用：`eod`、`user_stop`、`max_tokens`。

新数据库内统一用文档命名；如果临时导入示例脚本结果，应在导入时把 `eod` / `user_stop` 转成 `token_0` / `user_sentinel`。

### 7. reference 包装方式

参考实现当前写法是：

```python
gold = parse(f"$\\boxed{{{reference}}}$")
```

需要确认是否始终包一层 `\boxed{...}`。建议先按这个实现固定，并加测试覆盖：

- reference 是纯数字/表达式。
- reference 已经包含 `\boxed{}`。
- reference 是集合、区间、带单位答案。

如果 math_verify 对“已 boxed 再包 boxed”表现不好，再加条件避免重复包裹。

### 8. judge fallback 的逐组 eval

已确定每组都可以跑 judge fallback。实现时还需要固定：

- `judge_accuracy` 在每个 strategy 组内仍表示 `math_verify_pass or judge_pass`。
- eval 表每组的 `answer` 应保存该组实际送入 verifier/judge 的短 display answer。
- `fail_reason` 应能区分 `math_verify_false`、`math_verify_error`、`judge_false`、`judge_error`，但这些不进入 DB 可见主分数字段。

### 9. pass@k / avg@k 计算口径

每个 strategy 组都基于同一批 `(sample_index, repeat_index)` completions 计算自己的 rows。

需要确认：

- 如果某组 verifier 出错，该 row 记 false，并在 fail_reason/debug 里记录错误。
- `stop_rate` 对三组相同，不按 pass/fail 重新计算。

### 10. Space / export 读 nested metrics

`scores.metrics` 从 flat 变成 grouped 后，所有读取方都要兼容：

- Space 主表。
- hover 详情。
- result export。
- score 查询脚本。
- 任何假设 `metrics["avg@1"]` 在顶层的代码。

建议兼容策略：

- 如果 metrics 顶层有 `strategy_a`，按 grouped metrics 处理。
- 如果没有，按非 free-response 等 flat metrics 处理。

## 外部参考

- Math-Verify 官方仓库：`https://github.com/huggingface/Math-Verify`
- 包名安装形式：`math-verify[antlr4_13_2]`
- 官方示例 API：`from math_verify import parse, verify`
