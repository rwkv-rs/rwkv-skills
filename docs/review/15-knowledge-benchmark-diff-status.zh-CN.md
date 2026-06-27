# Knowledge Benchmark Diff Status

日期：2026-06-06

范围：

- 当前项目：`/home/chase/GitHub/rwkv-skills`
- 旧项目：`/home/chase/rwkv-skills`
- 本文只覆盖 Knowledge / multiple-choice 家族，不覆盖 Math、Coding、Instruction Following、Function Calling。

## 结论

Knowledge 的正式顶层 TOML 配置在两边基本一致；真正需要重构人员处理的是 registry / scheduler / runner 结构差异。

如果目标是“按旧项目行为对齐”，Knowledge 先不要包含 `include`，也不要调度 `fake_cot`。旧项目的正式 Knowledge 路径是 `multi_choice_plain` 和 `multi_choice_cot` 两条；当前 GitHub checkout 把 Knowledge 显式注册成 `no_cot`、`fake_cot`、`cot` 三种模式，并把 `include` 放进了 `BenchmarkField.KNOWLEDGE`。

## 当前 GitHub Checkout

核心入口：

- `src/eval/benchmark_registry.py`
- `src/eval/runner_registry.py`
- `src/eval/scheduler/jobs.py`
- `src/eval/knowledge/runner.py`
- `src/eval/knowledge/pipeline.py`
- `src/eval/prompt_builders.py`

Knowledge registry：

- `BenchmarkField.KNOWLEDGE` 是显式领域。
- `_THREE_MODE_KNOWLEDGE = (NO_COT, FAKE_COT, COT)`。
- `_MULTI_CHOICE_JOBS = ("multi_choice_plain", "multi_choice_fake_cot", "multi_choice_cot")`。
- 显式纳入：`include`、`mmlu`、`cmmlu`、`ceval`、`mmlu_pro`、`mmlu_redux`、`mmmlu`、`gpqa_main`、`gpqa_extended`、`gpqa_diamond`、`supergpqa`。
- `gpqa_main/gpqa_extended/gpqa_diamond` 共享 `dataset_name="gpqa"`，但 default split 分别是 `main/extended/diamond`。

Runner：

- 三个 runner 都走 `src.eval.knowledge.runner`。
- `multi_choice_plain` 传 `--cot-mode no_cot`。
- `multi_choice_fake_cot` 传 `--cot-mode fake_cot`。
- `multi_choice_cot` 传 `--cot-mode cot`。
- Scheduler 从 `ALL_BENCHMARKS` / `ALL_RUNNERS` 生成 job catalogue，而不是像旧项目那样主要从 data prepper 动态枚举。

Prompt：

- 实际 runner 会读取 `configs/<benchmark>.toml` 的 `direct/cot/final` stage override；如果没有 override，就回退到 `src/eval/knowledge/pipeline.py` 的默认模板。
- 默认英文 direct prompt、CoT prompt、final-answer prompt 与旧项目的 `src/eval/evaluators/multi_choice.py` 基本一致。
- 中文模板选择规则也一致：dataset name 中包含 `ceval`、`zh`、`cn` 时走中文模板。
- `src/eval/prompt_builders.py` 已经有 rwkv-rs 风格的 expected-context builder，并定义了 `FAKE_COT` 的空 think 块形式；但当前 Knowledge runner 的实际 prompt 仍主要由 `src/eval/knowledge/pipeline.py` 的模板路径决定。

## 旧项目

核心入口：

- `src/eval/scheduler/jobs.py`
- `src/bin/eval_multi_choice.py`
- `src/bin/eval_multi_choice_cot.py`
- `src/eval/evaluators/multi_choice.py`
- `src/eval/datasets/data_prepper/multiple_choice/*`

Knowledge / multiple-choice catalogue：

- 没有 `BenchmarkField.KNOWLEDGE` registry。
- Scheduler 从 `available_multiple_choice_datasets()` 动态枚举 data prepper 注册的数据集。
- 默认 split 只特殊处理 `gpqa -> main` 和 `ceval -> test`，其它 multiple-choice 默认 `test`。
- Scheduler 额外追加 `ceval_exam_test`。
- Job catalogue 只有 `multi_choice_plain` 和 `multi_choice_cot`；没有 `multi_choice_fake_cot`。
- 旧项目没有看到 `include` multiple-choice prepper 或 scheduler 注册。

Runner：

- `multi_choice_plain` -> `src.bin.eval_multi_choice`
- `multi_choice_cot` -> `src.bin.eval_multi_choice_cot`
- 这两个 bin entrypoint 直接创建旧 `MultipleChoicePipeline`。

Prompt：

- 默认模板在 `src/eval/evaluators/multi_choice.py`。
- 英文 direct prompt：专家身份 + question + choices，`Assistant: The answer is` 后接 logits scoring。
- 英文 CoT prompt：同样题面，`Assistant: <think` 生成 CoT，再通过 final prompt 接 logits scoring。
- 中文 prompt 用于 `ceval` / `zh` / `cn` 数据集。
- 没有 fake-CoT prompt 分支。

## 正式 TOML 配置

以下顶层 TOML 两边 diff 为空：

- `configs/mmlu.toml`
- `configs/cmmlu.toml`
- `configs/ceval.toml`
- `configs/mmlu_pro.toml`
- `configs/mmlu_redux.toml`
- `configs/mmmlu.toml`
- `configs/gpqa.toml`
- `configs/supergpqa.toml`

共同配置形态：

- `template = "multi_choice_cot_default"`
- `pass_k = []`
- `report_pass_k = []`
- `mmlu/cmmlu/ceval/mmlu_pro/mmlu_redux/supergpqa`: `avg_k = [1]`
- `mmmlu`: `avg_k = [0.2]`
- `gpqa`: `avg_k = [16]`

`multi_choice_cot_default` 模板目前也没有 Knowledge 相关 diff；此前看到的 `_templates.toml` 差异只影响 math/free-response prompt，不影响 Knowledge。

## 需要对齐的差异

1. `include`

   当前 GitHub checkout 把 `include` 放进 Knowledge registry。旧项目没有对应 multiple-choice prepper / scheduler 注册。按旧项目对齐时，应先从 Knowledge 正式调度范围排除 `include`，除非后续明确需要单独支持。

2. `fake_cot`

   当前 GitHub checkout 有 `multi_choice_fake_cot` runner 和 `FAKE_COT` mode。旧项目没有这一路。按旧项目对齐时，应避免把 fake-CoT 纳入正式 Knowledge benchmark matrix。

3. GPQA split 范围

   当前 GitHub checkout 显式注册 `gpqa_main`、`gpqa_extended`、`gpqa_diamond`。旧项目 scheduler 默认只把 `gpqa` 作为 `main` split 调度。按旧项目对齐时，正式默认应先只保留 `gpqa_main`；extended/diamond 可以作为显式扩展，而不是默认 Knowledge 全量。

4. Scheduler 生成方式

   当前 GitHub checkout 是 registry-first：benchmark metadata 决定 dataset slug 和 job。旧项目是 prepper-first：data prepper 可用数据集决定 scheduler catalogue，再手写 job。重构时不要只改 TOML；必须同步 registry、runner registry、scheduler job generation。

5. Resume / attempt identity

   当前 GitHub checkout 的 Knowledge pipeline 使用 `AttemptKey(sample_index, repeat_index, pass_index)`，payload 中有 `pass_index`。旧项目主要是 `(sample_index, repeat_index)`。如果后续要完全复刻旧项目 DB 行为，需要确认 Space/DB 是否接受新 `pass_index=0` 语义，或者统一在迁移层兼容。

## 推荐下一步

先只做 Knowledge 的最小行为对齐：

1. 从正式 Knowledge matrix 排除 `include`。
2. 暂停或移除 `multi_choice_fake_cot` 的正式调度入口。
3. GPQA 默认只保留 `gpqa_main`，把 `extended/diamond` 作为非默认扩展。
4. 保留现有 TOML 和默认 prompt，不先改 prompt。
5. 加一个小测试验证 `BENCHMARKS_BY_FIELD[KNOWLEDGE]`、`JOB_CATALOGUE` 和实际 scheduler dataset slugs 不含 `include/fake_cot`，并且默认 GPQA 只出 `gpqa_main`。

## 已确认未做

- 没有修改 runner / scheduler / config 代码。
- 没有运行 benchmark。
- 没有处理 Math、Coding、Instruction Following、Function Calling 的差异。
