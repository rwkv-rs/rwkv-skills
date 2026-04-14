# 10 重构路线图：`main.py + 配置驱动 + 多进程统一管理`

本路线图以“先保正确性，再收敛架构”为原则，目标是彻底替代当前分裂的 CLI 入口。

## 当前状态（2026-04-03）

- 已落地最小版统一入口：
  - 仓库根 `main.py`
  - `src/main.py`
  - `pyproject.toml` 中的 `rwkv-skills` script entry
- 已落地最小版 `RunConfig`：
  - 目前支持 TOML / JSON；YAML 仅在环境里存在 `PyYAML` 时启用
  - 已支持 `run / dataset / model / runner` 四段配置
  - 已支持 benchmark -> runner 的统一解析、dataset auto-prepare、`run_mode` / `job_name` 环境注入
- 已完成的里程碑口径：
  - M1 的“单 dataset 配置驱动跑通 completion/eval/score”已经有最小闭环
- 已额外确认：
  - `.venv/bin/pytest -q` 当前可直接通过，仓库现状为 `133 passed`
- 仍未完成：
  - `TaskSpec / RunContext / ResultEnvelope`
  - 多进程 `Orchestrator`
  - worker 池 + 独立 DB writer 进程

## 0. 先决条件（必须先做）

### R0-1. 先修会污染业务结果的 correctness 问题
- 关键修复点：
  - `src/eval/metrics/at_k.py:27-30`, `src/eval/metrics/at_k.py:54-56`（重复样本去重）
- 当前校验结果：
  - `src/db/sql_repo.py` 的 completion 写入已使用严格非负整数校验
  - schema 关键唯一约束已经体现在 PostgreSQL 建表脚本与显式 SQL 主链中
  - `src/eval/metrics/code_generation/livecodebench/evaluation.py` 的二次消费问题已修，并补了 generator 输入回归测试
- 原因：如果先做架构迁移，会把错误行为“稳定复制”到新系统。

### R0-2. 恢复测试可用性
- 状态：**已完成**
- 已完成：
  - `src/eval/metrics/instruction_following/instructions_util.py` 已改为惰性初始化，不再在 import 阶段下载 NLTK 资源。
  - 历史 `test_db_integration.py` 已不再阻塞 collection；当前 `.venv/bin/pytest -q` 为 `129 passed`
- 仍需做：
  - 补 repo/service 级 correctness 回归测试，重点覆盖 DB 幂等写入与 `pass@k/avg@k` 去重

---

## 1. 定义统一运行契约（配置先行）

### R1-1. 新建 `RunConfig`（YAML/TOML）
- 状态：**已明显吸收进来（最小可用版）**
- 建议字段：
  - `run.id`, `run.mode(eval|param_search)`
  - `model.weights`, `model.signature`
  - `dataset.slug`, `dataset.split`, `dataset.prepare`
  - `sampling`, `batch`, `resume_policy`, `db`, `artifacts`
- 当前已落地字段：
  - `run.id`, `run.mode`, `run.job`, `run.run_mode`, `run.batch_size`, `run.max_samples`, `run.probe_only`
  - `dataset.name`, `dataset.split`, `dataset.path`, `dataset.prepare`
  - `model.path`, `model.device`, `model.infer_*`
  - `runner.cot_mode`, `runner.judge_mode`, `runner.benchmark_kind` 及各领域常用 override
- 当前缺口：
  - `db` / `artifacts` 仍未真正接到各 runner
  - 还没有独立的 `sampling` 顶层配置对象，仍主要复用 benchmark config 与 runner 参数

### R1-2. 新建强类型对象
- 状态：**未完成**
- `TaskSpec`：描述一个待执行任务（model + dataset + eval mode）。
- `RunContext`：运行期上下文（路径、db、attempt id、随机种子）。
- `ResultEnvelope`：统一 completion/eval/score 写入契约。
- 说明：
  - Python 侧不必机械复刻 Rust 的 trait / state struct；重点是显式运行契约替代环境变量散射，而不是追求一比一的数据结构命名。

### R1-3. 入口收敛
- 状态：**部分完成**
- 已落地：
  - 新建 `main.py`（仓库根）
  - 新建 `src/main.py`（实际实现）
  - 已支持 `eval` / `param_search` 两类配置驱动运行，并复用现有 field runner / param-search runner
- 未落地：
  - 现有 `src/bin/*` 尚未批量改成兼容壳
  - `scheduler CLI` 仍是独立入口
  - `eval_agent_tau*` 仍是 legacy 旁路

#### 当前最小配置示例

```toml
[run]
mode = "eval"
run_mode = "auto"
batch_size = 32

[dataset]
name = "mmlu"

[model]
path = "weights/BlinkDL__rwkv7-g1/rwkv7-g1d-1.5b-20260212-ctx8192.pth"
device = "cuda"

[runner]
cot_mode = "cot"
db_write_queue = 4096
```

执行方式：

```bash
python main.py --config path/to/run.toml
```

或：

```bash
rwkv-skills --config path/to/run.toml
```

---

## 2. 统一编排器（多进程）

### R2-1. 引入 `Orchestrator`（controller 进程）
- 职责：
  - 读配置
  - 构建任务图
  - 资源分配
  - 生命周期管理（start/retry/abort/complete）
- 说明：
  - 这里应保留 Python 侧对本地进程 / GPU / DB 的灵活控制，不建议为了贴近 `rwkv-rs` 而硬搬其 async scheduler 状态机形状。

### R2-2. 引入 worker 池（模型执行进程）
- 设计：每个 worker 固定绑定设备，负责“加载模型 -> 执行生成 -> 发回 completion”。
- IPC：`multiprocessing.Queue` 或 `ProcessPoolExecutor` + 显式消息协议。

### R2-3. 单独 DB writer 进程
- 目标：避免并发写导致顺序漂移。
- writer 只处理 `ResultEnvelope`，并统一做幂等检查。
- 当前基础：
  - 已有 `src/db/async_writer.py` 作为线程内异步写入缓冲，但还不是跨进程统一 writer。

### R2-4. 调度器逐步瘦身
- 重点替换 `src/eval/scheduler/actions.py:122` 的 God Function。
- 按 `plan/allocate/launch/observe/reconcile` 切分服务。

---

## 3. 评估执行层模板化

### R3-1. 评估器 Runner 统一模板
- 现状重复点：`src/eval/evaluators/coding.py`, `free_response.py`, `multi_choice.py`。
- 目标：
  - 统一 probe/resume/chunk/on_complete 流程
  - dataset-specific 只保留 hook

### R3-2. 指标层“先规范数据，再计算指标”
- 将去重、脏数据过滤前置到 `MetricInputNormalizer`。
- `pass@k/avg@k` 只接受标准化输入。

### R3-3. 结果写入统一 API
- 以 `EvalDbService` 暴露少量稳定接口，禁止入口脚本直接拼装 repo 细节。

---

## 4. 数据集与缓存协议统一

### R4-1. `DatasetPrepareContext` 统一缓存/输出根
- 状态：**已明显吸收进来**
- 已完成：
  - `CallableRowsDatasetSpec` / `MaterializingDatasetSpec` 现在会把 `DatasetPrepareContext` 与运行时缓存环境下传到 loader。
  - 原先散落在 prepper 内部的 `Path("data")` / `data/hf_cache` 已改为走 runtime context 或运行期环境覆盖。
- 剩余工作：
  - `RunConfig` 顶层还没有独立 `artifacts/cache` 配置对象，这些路径还没从 `main.py` 显式暴露给用户。

### R4-2. `DatasetManifest` 全链路追溯
- 输出里必须携带 `source_dataset/source_split/revision/checksum`。

### R4-3. 并发安全
- 下载/解压加文件锁，缓存按 revision 分层。

---

## 5. Space 与结果消费解耦

### R5-1. 不再让 Space 直接扫目录
- 状态：**部分完成**
- 已完成：
  - `Space` 主分数加载已切到 `score_index.jsonl`。
  - score 写入阶段会同步追加 `results/space/score_index.jsonl`，Space 不再依赖 DB 聚合查询主表。
- 未完成：
  - 当前 `score_index.jsonl` 还是由 score 写入阶段直接维护，不是独立 orchestrator 统一维护。
  - Space 明细回看仍通过 `task_id` 查 PostgreSQL。

### R5-2. `src/space/app.py` 拆模块
- 至少拆为：selection、table、chart、export、UI binding。

---

## 6. 迁移策略（低风险落地）

1. **配置主线阶段**：`main.py` 继续作为唯一正式入口收口 eval 主线。
2. **编排阶段**：补 `Orchestrator + worker pool + DB writer process`。
3. **调参阶段**：把 `param_search` 线路接入同一运行契约。
4. **清理阶段**：删除 `src/bin` 中仍承载业务逻辑的 legacy 脚本。

---

## 里程碑验收标准

- M1：`main.py` 可用配置跑通一个 dataset（含 completion/eval/score）。
  - 状态：**已完成最小版（eval path）**
- M2：多进程 worker + DB writer 稳定，resume/retry 行为与预期一致。
- M3：所有旧 CLI 已变“兼容壳”，核心逻辑仅在 orchestrator。
- M4：关键回归用例全绿（DB correctness + metrics correctness + E2E）。
