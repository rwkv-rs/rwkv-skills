# 11 Benchmark 对齐与线路收口计划

目标：把 `rwkv-skills` 的 benchmark 组织方式、领域划分、运行入口逐步收敛到 `rwkv-rs` 的风格，同时保留 Python 侧更灵活的 scheduler / DB / 分布式能力。

## 1. 当前对齐结果

基于 `rwkv-rs/crates/rwkv-eval/src/datasets` 与 `rwkv-skills/src/eval/benchmark_registry.py` 当前代码：

- `rwkv-rs` benchmark 总数：50
- `rwkv-skills` benchmark 总数：53
- 同名且同领域共有：43

### 1.1 真正缺失于 `rwkv-skills` 的 benchmark

- `include`
- `polymath`

这两个是实质缺口，需要补 benchmark 接入、dataset spec、runner 适配和 scheduler matrix。

### 1.2 主要是建模方式不同，不是完全缺失

- `gpqa_main` / `gpqa_extended` / `gpqa_diamond`
  - `rwkv-rs`：三个独立 benchmark 名
  - `rwkv-skills`：一个 `gpqa` benchmark，靠 split 区分
- `tau_bench`
  - `rwkv-rs`：一个 benchmark，内部再按 domain 处理
  - `rwkv-skills`：拆成 `tau_bench_retail` / `tau_bench_airline` / `tau_bench_telecom`
- `arena_hard_v2`
  - `rwkv-rs`：显式带版本号
  - `rwkv-skills`：当前是 `arena_hard`

这类差异的核心不是“能不能跑”，而是“catalog 是否统一”。后续要决定 canonical name 是向 `rwkv-rs` 靠齐，还是保留 Python 侧更细的 domain slug，再加 alias。

### 1.3 `rwkv-skills` 当前独有 benchmark

- `flores200`
- `ifbench`
- `tau2_bench_retail`
- `tau2_bench_airline`
- `tau2_bench_telecom`
- `tau_bench_retail`
- `tau_bench_airline`
- `tau_bench_telecom`
- `gpqa`
- `arena_hard`

其中：

- `flores200` / `ifbench` 属于功能扩展，不应删除。
- `tau2_*` 是 Python 侧新增能力，也不应向 `rwkv-rs` 回退。
- `gpqa` / `arena_hard` / `tau_bench_*` 属于命名与抽象层不一致，应该收敛。

## 2. 当前 `rwkv-skills` 的测评线路

当前“官方可调度”线路由 `src/eval/scheduler/jobs.py` 定义，可分为 6 大类。

### 2.1 Multi-choice

- `multi_choice_plain`
  - 直接 logits 选项打分，不生成 CoT。
- `multi_choice_fake_cot`
  - 仍是 logits 打分，但 prompt 模板带 fake-CoT 风格。
- `multi_choice_cot`
  - 先生成 CoT，再基于 CoT 上下文做最终选择。

### 2.2 Free-response / Maths

- `free_response`
  - 两阶段生成：CoT + final answer。
  - 以规则/数值/标准答案匹配为主。
- `free_response_judge`
  - 生成流程与 `free_response` 基本一致。
  - 评测阶段额外依赖外部 LLM judge。

### 2.3 Coding

- `code_human_eval`
  - 代码生成后跑本地 verdict。
- `code_mbpp`
  - 支持 `no_cot` / `fake_cot` / `cot` 三种 prompt 模式。
- `code_livecodebench`
  - 代码任务更重，通常是 CoT + final code 的两阶段生成，再跑 verdict。

### 2.4 Instruction-following

- `instruction_following`
  - 单轮生成。
  - 规则型 checker / 指令匹配为主。

### 2.5 Function-calling

- `function_browsecomp`
  - CoT + final answer，最后走 judge。
- `function_mcp_bench`
  - planning / decision / final answer 多阶段循环。
  - 依赖官方 worker bridge。
- `function_tau_bench`
  - 本地 tool loop，环境在 Python 侧。
- `function_tau2_bench`
  - 与 `function_tau_bench` 类似，但 runtime 是 tau2。

### 2.6 Param-search

- `param_search_free_response`
- `param_search_free_response_judge`
- `param_search_select`

这是 scheduler 的调参支线，不是 benchmark 本身，但现在仍算一条独立运行线路。

## 3. 当前线路的主要问题

### 3.1 线路不是按“领域 + runner”收敛，而是按脚本横向增长

现在的主入口仍是：

- `src/bin/eval_multi_choice*.py`
- `src/bin/eval_free_response*.py`
- `src/bin/eval_code_*.py`
- `src/bin/eval_function_*.py`

问题不是只有文件多，而是：

- task 创建逻辑重复
- writer 生命周期重复
- score payload 拼装重复
- probe / resume / failure 收尾重复

### 3.2 同一领域内部，dataset-specific 与 runner-specific 逻辑还混着

例如：

- multi-choice 其实已经能抽成一个领域 runner，只在 `cot_mode` 和采样阶段不同。
- coding 其实能抽成 “prompt builder + execution verdict backend” 两层。
- function-calling 其实能抽成 “单轮问答型 / 多轮工具型 / 外部 worker bridge 型” 三个 runner 模板。

### 3.3 还有遗留旁路

例如：

- `src/bin/eval_agent_tau_bench.py`
- `src/bin/eval_agent_tau2_bench.py`

这些当前不在正式 scheduler matrix 里，说明项目里仍然存在“同类任务多条实现线并存”的情况。

## 4. 建议的对齐目标

### 4.1 Benchmark catalog 对齐到 `rwkv-rs` 风格

统一保留以下中心信息：

- `name`
- `field`
- `default_split`
- `cot_modes`
- `scheduler_jobs`
- `n_shots`
- `avg_ks`
- `pass_ks`

并提供：

- `ALL_BENCHMARKS`
- `BENCHMARKS_BY_FIELD`
- `get_benchmarks_with_field(field)`

这样 scheduler、Space、单入口、文档都只依赖一份 catalog。

### 4.2 数据集实现保留 Python 的 `DatasetSpec`，但 benchmark 命名尽量向 `rwkv-rs` 对齐

建议：

- `gpqa`
  - 改成显式支持 `gpqa_main` / `gpqa_extended` / `gpqa_diamond`
  - 内部仍可复用一个 spec/factory
- `arena_hard`
  - 明确是 `arena_hard_v2` 还是旧版，名字不要含糊
- `tau_bench`
  - 保留 domain-specific manifest 没问题
  - 但 catalog 层要决定 canonical name 是聚合还是拆分

### 4.3 Runner 按领域模板化

目标应是：

- `knowledge_runner`
- `maths_runner`
- `coding_runner`
- `instruction_following_runner`
- `function_calling_runner`

dataset-specific 只提供：

- prompt/context builder
- dataset loader
- evaluator / verdict adapter

## 5. 下一阶段执行顺序

### P1. 先收 catalog

- 已完成一部分：`benchmark_registry.py` 现在已带 `field/default_split/scheduler_jobs`，并提供按领域分组。
- 已完成一部分：`src/eval/evaluating/task_persistence.py` 现在已提供 `auto/new/resume/rerun` 四种 run mode，
  scheduler CLI 也支持 `--run-mode`，runner 已切到统一 `prepare_task_execution(...)`。
- 下一步：
  - 把 scheduler CLI / Space / 文档都改成读取 `BENCHMARKS_BY_FIELD`
  - 不再从 `jobs.py` 反推 benchmark 分类

### P2. 先补真缺口 benchmark

优先补：

1. `include`
2. `polymath`

原因：

- 这两个是 `rwkv-rs` 有而 `rwkv-skills` 真没有的 benchmark。
- 补完之后，两边 benchmark 覆盖才能谈“对齐”。

### P3. 再收命名差异

优先决策：

1. `gpqa` 是否拆成 `gpqa_main/extended/diamond`
2. `arena_hard` 是否改名为 `arena_hard_v2`
3. `tau_bench` 是保留 domain slug 还是增加聚合 alias

这里先定 canonical name，再做数据路径/结果路径迁移，不然会反复改 score layout。

### P4. Runner 收口

按风险从低到高：

0. 已完成一部分：`src/eval/runner_registry.py` 已按 `knowledge / maths / coding / instruction_following / function_calling / param_search`
   建好 runner registry，`scheduler/jobs.py` 已改成从 registry 派生 `JOB_CATALOGUE`。
1. 已完成一部分：`src/bin/knowledge_runner.py` 现在已经把 `multi_choice_plain / multi_choice_fake_cot / multi_choice_cot`
   收成单入口，`runner_registry.py` 里的 knowledge jobs 已统一指向它；`src/bin/eval_multi_choice*.py` 只保留兼容壳。
2. 已完成一部分：`src/bin/maths_runner.py` 现在已经把 `free_response / free_response_judge` 收成单入口，
   `runner_registry.py` 里的 maths jobs 已统一指向它；`src/bin/eval_free_response*.py` 只保留兼容壳。
3. 已完成一部分：runner / pipeline 实现现在开始按领域下沉到 `src/eval/knowledge/` 和 `src/eval/maths/`，
   更接近 `rwkv-rs/crates/rwkv-eval/src/datasets/<field>/...` 的组织方式。
4. coding runner 合并
5. instruction-following runner 合并
6. function-calling runner 分三类模板收口

### P5. 分布式拆分

建议拆成两端：

- inference worker
  - 只负责模型加载、批处理生成、tool decision 推理
- eval orchestrator
  - 只负责 dataset resolve、任务编排、judge、verdict、DB 写入、重试恢复

这样才符合你说的“像 rwkv-rs 一样只注册 runner 和调度器”，同时又保住 Python 侧分布式能力。

### P6. 最后删 legacy

只有在以下三项完成后再删：

- dataset catalog 单一真相源稳定
- runner registry 收口完成
- 剩余 legacy preparer 都迁完 spec

删除顺序建议：

1. 不再进入 scheduler 的旧入口脚本
2. `LegacyPreparerDatasetSpec` 覆盖的剩余 preparer
3. 旧 agent_bench 旁路线

## 6. 建议的短期落地点

如果按“最小成本、最大收益”排序，下一轮最值得做的是：

1. 补 `include` 与 `polymath`
2. 把 `gpqa` 收成 `gpqa_main/extended/diamond`
3. 引入 runner registry，先把 multi-choice / free-response 两个领域从 `src/bin` 收出来

这样做完以后：

- benchmark 覆盖能更接近 `rwkv-rs`
- catalog 命名更统一
- 入口层不再继续膨胀
