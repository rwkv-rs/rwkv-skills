# 01 架构层审阅

## 高优先级问题（必须改）

### A1. 入口分裂严重，业务流程被复制到多个 CLI 脚本
- 证据：`src/bin/` 下约 20 个入口脚本，且 `main()` 结构高度重复（如 `src/bin/eval_free_response.py:128`, `src/bin/eval_free_response_judge.py:149`, `src/bin/eval_code_human_eval.py:57`, `src/bin/eval_multi_choice_cot.py:96`）。
- 影响：
  - 同一业务流程（建 task -> 写 completion -> 评估 -> 写 score -> 导出）多处复制，修一个漏一个。
  - 参数、默认值、异常处理逐步漂移，行为不一致。
- 建议改法：
  1. 保留一个统一入口 `main.py`。
  2. 使用配置驱动（YAML/TOML）描述任务类型、模型、数据集、采样策略。
  3. CLI 脚本只留兼容壳，内部统一调用 `main.py` 任务编排器。

### A2. 调度层与执行层边界不清，导致职责混乱
- 证据：`src/eval/scheduler/actions.py:122` 的 `action_dispatch` 同时做了队列生成、GPU 探测、环境拼装、命令组装、日志备份、进程拉起、失败恢复等几乎全部职责。
- 影响：
  - 单函数接近 God Function，难以测试/替换。
  - 后续改成多进程统一管理时，迁移成本高。
- 建议改法：拆成独立组件：
  - `QueuePlanner`
  - `ResourceAllocator`
  - `RunLauncher`
  - `FailurePolicy`
  - `RunStateStore`

### A3. 隐式环境变量协议过重
- 证据：运行时强依赖大量 `RWKV_*` 环境变量（如 `src/eval/scheduler/actions.py:321-333`, `src/bin/eval_free_response.py:187-188`）。
- 影响：
  - 调用链不可见，跨进程/跨模块调试困难。
  - 容易出现“本地能跑、线上失效”的隐式依赖问题。
- 建议改法：改为显式 `RunContext` 对象，运行参数通过配置和进程间消息传递，不再靠全局环境变量拼接。

## 中优先级问题

### A4. 模块间存在“配置读取重复”
- 证据：多个入口重复解析 benchmark/model 的采样配置（如 `src/bin/eval_free_response.py:142-153`, `src/bin/eval_code_livecodebench.py:61-72`）。
- 建议：抽取统一的 `SamplingResolver` 服务。

### A5. 旧脚本/迁移脚本耦合主代码路径
- 证据：`src/bin/migrate_old_results.py`（604 行）等脚本仍在主包中。
- 建议：迁移脚本下沉到 `tools/legacy/`，不参与主流程。

## 目标架构建议

- 单入口：`main.py`
- 单配置：`run_config.toml`（或 `yaml`）
- 单编排：`Orchestrator`（多进程）
- 单任务协议：`TaskSpec` / `RunContext`
- 单结果契约：统一 completion/eval/score 数据结构

