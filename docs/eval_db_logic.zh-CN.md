## 评估数据库业务逻辑梳理

本文聚焦 `src/db/*` 与 `src/bin/eval_*` 的数据库读写链路，覆盖：
- 版本管理（task/version）
- 中断续跑（auto/new/resume/rerun）
- 结果写入幂等（completion/eval/score）
- 调度器侧读取（latest score / completed jobs）

---

## 1) 数据模型与职责

### 核心表
- `benchmark`：数据集维度（`benchmark_name + benchmark_split`）
- `model`：模型维度（`model_name + arch_version + data_version + num_params`）
- `task`：一次评测运行（可视为“version”）
- `completions`：样本级生成结果（按 `task_id + sample_index + repeat_index + pass_index` 唯一）
- `eval`：样本级判分
- `scores`：任务级聚合指标

### 代码分层
- `pool.py` / `database.py`：PostgreSQL connection pool 初始化与共享 `Db { pool }`
- `sql_repo.py`：显式 SQL 的数据库 CRUD 与查询
- `eval_db_service.py`：业务编排（创建任务、续跑、写入、导出用读取）
- `src/bin/eval_*.py`：评测入口，负责串联“生成 -> 评估 -> 入库”

---

## 2) 版本管理设计（task 即 version）

### 版本创建
统一入口已切到 `prepare_task_execution(...)`：
- 常规 runner：`src/eval/evaluating/task_persistence.py`
  - 统一提供 `auto/new/resume/rerun`
  - 底层仍通过 `EvalDbService.get_resume_context()` + `create_task_from_context()` 完成
- 旧兼容路径 / 个别脚本：
  - 仍可能直接调用 `EvalDbService`，但主线语义已经统一到 `prepare_task_execution(...)`

`task` 会记录：
- 运行元信息：`evaluator/job_name`、`git_hash`、`sampling_config`、`log_path`
- 维度外键：`model_id`、`benchmark_id`
- 生命周期：`status`（running/completed/failed）

### 最新版本判定
“最新”依赖时间排序，当前已统一为**稳定排序**：
- 先按 `created_at DESC`
- 再按自增主键 `task_id/score_id DESC` 兜底

这可以避免同秒写入导致的“最新版本抖动”。

### overwrite 行为
当前主语义是 `RWKV_EVAL_RUN_MODE`：

- `auto`
  - 无 matching task 时创建新 task
  - 存在唯一 resumable task 时续跑
  - 其余存在 matching task 但不可续跑的情况，创建新 task，并把有效语义视为 rerun
- `new`
  - 遇到 matching task 直接拒绝
- `resume`
  - 遇到 completed task 拒绝
  - 找不到唯一 resumable task 也拒绝
- `rerun`
  - 强制创建新 task

`RWKV_SCHEDULER_OVERWRITE=1` 只是旧接口兼容层，等价于 `run_mode=rerun`。

---

## 3) 中断续跑设计

### 三层检索（ResumeContext）
`get_resume_context()` 一次性返回：
1. `benchmark_id` / `model_id`
2. `matching_tasks` / `completed_task_ids` / `resumable_task_ids` / `task_id` / `can_resume`
3. `completed_keys`

### stage 语义
pipeline 内部仍可能产生两类 payload：
- `_stage="cot"`：仅中间阶段
- `_stage="answer"`：完整可评估结果

但当前 DB 落地口径已经收敛为：
- 只有 `_stage="answer"` 会进入 `completions`
- `_stage="cot"` 不作为可续跑 completion 单独持久化
- `completed_keys` 直接来自已完成 answer completion 的 `(sample_index, repeat_index, pass_index)`
- 评测入口统一用 `skip_keys=completed_keys`

### 异常时任务状态回写
部分 runner 在异常路径中会校验 `count_completions(status="Completed")`，避免把未跑满 expected attempt 的 task 误标记成 completed。

---

## 4) 写入幂等与一致性

### completions：幂等 Upsert
`insert_completion()` 使用 `ON CONFLICT (task_id, sample_index, repeat_index, pass_index) DO UPDATE`。
因此每个 attempt key 只保留一行。

### eval：按 completion 键去重写入
`ingest_eval_payloads()` 会先把 `(sample_index, repeat_index, pass_index)` 映射到 `completions_id`，再跳过已有 `eval` 的 completion。
repo 层 `insert_eval()` 只负责插入，真正的幂等收口发生在 service 层预筛选 + `eval.completions_id` 唯一约束。

并统一兜底空字段（`answer/ref_answer/fail_reason`）为字符串，避免 NULL 触发约束失败。

### score：任务聚合写入
`record_score_payload()` 写入 `scores` 后将 `task.status` 更新为 `completed`。
调度器侧是否完成以“是否有 score”为准。

---

## 5) 读取口径（防止半成品污染评测）

所有评测脚本在“生成后评估”阶段统一读取：
- `list_completion_payloads(task_id, status="Completed")`

这样可以保证评分只消费真正已落地的 answer completion。

代码类任务（HumanEval/MBPP/LiveCodeBench）的 `score.samples` 统一取 `len(completions_payloads)`，确保续跑后样本数反映全量结果，而不是“本次新增数量”。

---

## 6) 已知风险与后续建议

1. **实体去重约束**
   - `benchmark/model/completion/eval/checker/score` 现在都已有数据库唯一约束。
   - 但 `task` 仍然允许同一 identity 出现多个 version；这符合 rerun 语义，但也意味着更强的去重/claim 仍需上层控制。

2. **状态枚举约束**
   - `task/completion/score` 已有 CHECK + canonicalization。
   - 若后续扩展新状态，需要同步更新 schema 与 repo 层 canonicalization。

3. **事务边界可观测性**
   - 若后续出现并发调度写入，建议补充 task 级审计日志与幂等请求 ID。

4. **历史兼容清理**
   - 若历史库已存在重复 `eval` 行，建议一次性离线清理（按 `completions_id` 保留最新）。
