## 评估数据库业务逻辑梳理

本文聚焦 `src/db/*` 与 `src/bin/eval_*` 的数据库读写链路，覆盖：
- 版本管理（task/version）
- 中断续跑（resume）
- 结果写入幂等（completion/eval/score）
- 调度器侧读取（latest score / completed jobs）

---

## 1) 数据模型与职责

### 核心表
- `benchmark`：数据集维度（`benchmark_name + benchmark_split`）
- `model`：模型维度（`model_name + arch_version + data_version + num_params`）
- `task`：一次评测运行（可视为“version”）
- `completions`：样本级生成结果（按 `task_id + sample_index + repeat_index` 唯一）
- `eval`：样本级判分
- `scores`：任务级聚合指标

### 代码分层
- `orm.py`：Schema + session/transaction 管理
- `eval_db_repo.py`：数据库 CRUD 与查询
- `eval_db_service.py`：业务编排（创建任务、续跑、写入、导出用读取）
- `src/bin/eval_*.py`：评测入口，负责串联“生成 -> 评估 -> 入库”

---

## 2) 版本管理设计（task 即 version）

### 版本创建
统一入口：
- 常规评测：`EvalDbService.get_resume_context()` + `create_task_from_context()`
- param-search：`EvalDbService.get_or_create_task(..., allow_resume=False)` 强制新 task

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
调度器 `--overwrite` 会传递环境变量 `RWKV_SCHEDULER_OVERWRITE=1`。
评测脚本读取该变量后，`get_resume_context(force_new_task=True)`，确保创建新 task，而不是续跑旧 task。

---

## 3) 中断续跑设计

### 三层检索（ResumeContext）
`get_resume_context()` 一次性返回：
1. `benchmark_id` / `model_id`
2. `task_id` / `can_resume`
3. `completed_keys` / `cot_only_keys`

### completion 阶段语义
- `status="cot"`：仅第一阶段完成（CoT）
- `status="answer"`：完整可评估结果

续跑时：
- 仅 `status="answer"` 进入 `completed_keys`
- `status="cot"` 进入 `cot_only_keys`（仅用于观测）
- 评测入口用 `skip_keys=completed_keys`，保证不会跳过只做完 CoT 的样本

### 异常时任务状态回写
部分两阶段任务在异常路径中会校验 `count_completions(status="answer")`，避免把“只有 cot 的半成品”误标记成 completed。

---

## 4) 写入幂等与一致性

### completions：幂等 Upsert
`insert_completion()` 使用 `ON CONFLICT (task_id, sample_index, repeat_index) DO UPDATE`。
因此每个样本键只保留一行，且会被后续阶段覆盖成最新 `status/context`。

### eval：按 completion 键更新
`insert_eval()` 以 `completions_id` 为幂等键：
- 存在则更新
- 不存在则插入

并统一兜底空字段（`answer/ref_answer/fail_reason`）为字符串，避免 NULL 触发约束失败。

### score：任务聚合写入
`record_score_payload()` 写入 `scores` 后将 `task.status` 更新为 `completed`。
调度器侧是否完成以“是否有 score”为准。

---

## 5) 读取口径（防止半成品污染评测）

所有评测脚本在“生成后评估”阶段统一读取：
- `list_completion_payloads(task_id, status="answer")`

这样可以避免把中断残留的 `cot` 半成品带入评分。

代码类任务（HumanEval/MBPP/LiveCodeBench）的 `score.samples` 统一取 `len(completions_payloads)`，确保续跑后样本数反映全量结果，而不是“本次新增数量”。

---

## 6) 已知风险与后续建议

1. **实体去重约束**
   - 当前 `benchmark/model` 逻辑键未加数据库唯一约束；虽然查询已做“取最新一条”容错，但建议后续补 migration。

2. **状态枚举约束**
   - `task/completion` 的 `status` 目前是字符串自由写入，建议收敛为枚举或 CHECK。

3. **事务边界可观测性**
   - 若后续出现并发调度写入，建议补充 task 级审计日志与幂等请求 ID。

4. **历史兼容清理**
   - 若历史库已存在重复 `eval` 行，建议一次性离线清理（按 `completions_id` 保留最新）。

