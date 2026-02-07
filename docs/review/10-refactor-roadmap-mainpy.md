# 10 重构路线图：`main.py + 配置驱动 + 多进程统一管理`

本路线图以“先保正确性，再收敛架构”为原则，目标是彻底替代当前分裂的 CLI 入口。

## 0. 先决条件（必须先做）

### R0-1. 先修会污染业务结果的 correctness 问题
- 关键修复点：
  - `src/db/eval_db_repo.py:417-426`（索引兜底为 0）
  - `src/db/eval_db_repo.py:476-479`（宽条件更新 eval）
  - `src/eval/metrics/at_k.py:27-30`, `src/eval/metrics/at_k.py:54-56`（重复样本去重）
  - `src/eval/metrics/code_generation/livecodebench/evaluation.py:170`, `src/eval/metrics/code_generation/livecodebench/evaluation.py:233`（二次消费迭代器）
- 原因：如果先做架构迁移，会把错误行为“稳定复制”到新系统。

### R0-2. 恢复测试可用性
- 必做：
  - `tests/test_db_integration.py` 迁移到现有 ORM/service。
  - `src/eval/metrics/instruction_following/instructions_util.py:27-35` 改为惰性资源初始化。

---

## 1. 定义统一运行契约（配置先行）

### R1-1. 新建 `RunConfig`（YAML/TOML）
- 建议字段：
  - `run.id`, `run.mode(eval|param_search)`
  - `model.weights`, `model.signature`
  - `dataset.slug`, `dataset.split`, `dataset.prepare`
  - `sampling`, `batch`, `resume_policy`, `db`, `artifacts`

### R1-2. 新建强类型对象
- `TaskSpec`：描述一个待执行任务（model + dataset + eval mode）。
- `RunContext`：运行期上下文（路径、db、attempt id、随机种子）。
- `ResultEnvelope`：统一 completion/eval/score 写入契约。

### R1-3. 入口收敛
- 新建 `main.py`（唯一正式入口）。
- 现有 `src/bin/eval_*.py` / `param_search_*.py` 先变兼容壳，只做“解析旧参数 -> 生成 RunConfig -> 调 main”。

---

## 2. 统一编排器（多进程）

### R2-1. 引入 `Orchestrator`（controller 进程）
- 职责：
  - 读配置
  - 构建任务图
  - 资源分配
  - 生命周期管理（start/retry/abort/complete）

### R2-2. 引入 worker 池（模型执行进程）
- 设计：每个 worker 固定绑定设备，负责“加载模型 -> 执行生成 -> 发回 completion”。
- IPC：`multiprocessing.Queue` 或 `ProcessPoolExecutor` + 显式消息协议。

### R2-3. 单独 DB writer 进程
- 目标：避免并发写导致顺序漂移。
- writer 只处理 `ResultEnvelope`，并统一做幂等检查。

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
- 解决当前 `Path("data")` 硬编码散落问题（见 `06-datasets.md`）。

### R4-2. `DatasetManifest` 全链路追溯
- 输出里必须携带 `source_dataset/source_split/revision/checksum`。

### R4-3. 并发安全
- 下载/解压加文件锁，缓存按 revision 分层。

---

## 5. Space 与结果消费解耦

### R5-1. 不再让 Space 直接扫目录
- 由 orchestrator 维护 `score_index.jsonl`（或 DB 视图），Space 只读索引。

### R5-2. `src/space/app.py` 拆模块
- 至少拆为：selection、table、chart、export、UI binding。

---

## 6. 迁移策略（低风险落地）

1. **双写阶段**：旧 CLI 与 `main.py` 同时可跑，新路径结果写到新目录并做 diff。
2. **灰度阶段**：先迁移 free_response / multi_choice，再迁移 code / instruction_following。
3. **只读兼容阶段**：旧 CLI 仅保留参数解析与提示，不再承载业务逻辑。
4. **收口阶段**：删除 `src/bin` 业务实现，只保留极薄 alias。

---

## 里程碑验收标准

- M1：`main.py` 可用配置跑通一个 dataset（含 completion/eval/score）。
- M2：多进程 worker + DB writer 稳定，resume/retry 行为与预期一致。
- M3：所有旧 CLI 已变“兼容壳”，核心逻辑仅在 orchestrator。
- M4：关键回归用例全绿（DB correctness + metrics correctness + E2E）。

