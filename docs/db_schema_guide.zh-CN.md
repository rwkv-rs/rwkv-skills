## 目的
本项目当前同时保留 JSONL 输出与数据库写入（便于对比）。前后端主要通过数据库读取评测结果；本文说明表结构与字段含义，并给出常用查询与交接操作指引。

## 表总览（6 张）
- `eval_run`：一次评测运行（前端主列表/汇总）
- `eval_sample`：样本元信息（按 benchmark+split+index 去重）
- `eval_run_sample`：运行与样本关联 + 样本级结果
- `eval_attempt`：样本尝试（重试/并发/分片）
- `eval_stage_output`：阶段输出（CoT / final）
- `eval_cot_checkpoint`：CoT 中断续跑检查点

## 字段字典

### eval_run（评测运行汇总）
- `id`：UUID 主键
- `benchmark_name`：基准名称（如 `gsm8k`）
- `dataset`：数据集完整 slug（如 `gsm8k_test`）
- `dataset_split`：数据划分（如 `test`）
- `model_name`：模型名称（权重文件名 stem）
- `model_slug`：模型安全名（slug）
- `model_revision`：模型版本/修订号（若有）
- `model_path`：权重路径
- `cot`：是否 CoT
- `run_tag`：运行标签（用于区分多次运行）
- `sampling_config`：采样参数（JSON）
- `runtime_config`：运行配置（JSON）
- `code_version`：代码版本
- `task`：任务名（如 `free_response`）
- `task_details`：任务细节（JSON）
- `metrics`：评测指标汇总（JSON；如 `pass@1`）
- `samples`：样本数
- `problems`：题目数
- `log_path`：完成记录文件路径
- `eval_details_path`：评测详情文件路径（eval results jsonl）
- `status`：运行状态
- `error_msg`：错误信息
- `created_at`：创建时间
- `started_at`：开始时间（真实）
- `finished_at`：结束时间（真实）

### eval_sample（样本元信息）
- `id`：UUID 主键
- `benchmark_name`：基准名称
- `dataset_split`：数据划分
- `sample_index`：样本序号
- `question`：题目/上下文文本
- `ref_answer`：参考答案
- `meta`：附加元数据（JSON）
- `created_at`：创建时间

### eval_run_sample（样本级结果）
- `id`：UUID 主键
- `run_id`：评测运行
- `sample_id`：样本
- `repeat_index`：重复次数序号
- `current_stage`：当前阶段（如 `cot`/`final`）
- `latest_attempt_index`：最新尝试序号
- `answer`：模型最终答案
- `is_passed`：是否正确
- `fail_reason`：失败原因
- `status`：运行状态
- `error_msg`：错误信息
- `created_at`：创建时间
- `started_at`：开始时间（真实）
- `finished_at`：结束时间（真实）

### eval_attempt（尝试）
- `id`：UUID 主键
- `run_sample_id`：所属样本运行
- `attempt_index`：尝试序号
- `worker_id`：工作进程标识
- `shard_id`：分片编号
- `shard_count`：分片总数
- `seed`：随机种子
- `status`：状态
- `error_msg`：错误信息
- `created_at`：创建时间
- `started_at`：开始时间（真实）
- `finished_at`：结束时间（真实）

### eval_stage_output（阶段输出）
- `id`：UUID 主键
- `attempt_id`：所属尝试
- `stage`：阶段名（如 `cot` / `final`）
- `seq`：序号（多段输出排序）
- `prompt`：阶段输入
- `completion`：阶段输出
- `finish_reason`：结束原因
- `provider_request_id`：请求 ID
- `raw_response`：原始响应（JSON）
- `token_count_prompt`：提示词 token 数
- `token_count_response`：输出 token 数
- `latency_ms`：延迟
- `cost_usd`：费用
- `is_partial`：是否部分输出
- `is_final`：是否最终输出
- `created_at`：写入时间（真实）

### eval_cot_checkpoint（CoT 续跑）
- `id`：UUID 主键
- `attempt_id`：所属尝试
- `stage`：阶段名（通常 `cot`）
- `token_offset`：token 偏移量
- `partial_completion`：部分输出
- `kv_cache_ref`：缓存引用
- `rng_state`：随机状态（JSON）
- `status`：状态
- `latest`：是否最新检查点
- `created_at`：写入时间（真实）

## 常用查询（前端可直接使用）
1) 运行列表（模型/权重/准确率）
```sql
SELECT id, model_name, model_path, cot, metrics, samples, problems, status, started_at, finished_at
FROM eval_run
ORDER BY created_at DESC;
```

2) 运行详情（样本级结果）
```sql
SELECT s.sample_index, rs.repeat_index, s.question, s.ref_answer, rs.answer, rs.is_passed, rs.fail_reason
FROM eval_run_sample rs
JOIN eval_sample s ON s.id = rs.sample_id
WHERE rs.run_id = $1
ORDER BY s.sample_index, rs.repeat_index;
```

3) 查看 CoT / final 输出
```sql
SELECT so.stage, so.prompt, so.completion, so.finish_reason, so.created_at
FROM eval_stage_output so
JOIN eval_attempt a ON a.id = so.attempt_id
JOIN eval_run_sample rs ON rs.id = a.run_sample_id
WHERE rs.run_id = $1 AND rs.sample_id = $2 AND so.is_final = TRUE
ORDER BY so.created_at DESC;
```

## 当前流程（简述）
- 评测流程生成 JSONL（保留用于对比）
- 同时写入数据库：`eval_run` 记录汇总；`eval_sample`/`eval_run_sample` 记录样本与结果
- CoT 阶段输出写入 `eval_stage_output`，部分输出会写入 `eval_cot_checkpoint`

## 出差交接与远程开发指引
1) 在笔记本保持相同 WSL 配置（版本、Python、依赖）
2) SSH 登录开发机/服务器
3) 拉取自己的分支并继续开发
```bash
git fetch
git checkout feature/your-branch
git pull
```
4) 确保 DB 连接配置（`.env` 或配置文件）与本地一致
