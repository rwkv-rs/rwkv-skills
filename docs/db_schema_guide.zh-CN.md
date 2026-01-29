## 目的
当前评测流程以数据库写入为主（默认不再输出 JSONL 结果）。本文说明表结构与字段含义，并给出常用查询。

## 表总览（5 张 + 2 视图）
- `version`：评测版本/运行元信息（job/dataset/model/git）
- `completions`：模型输出轨迹（prompt/completion/stop/context）
- `eval`：样本级评测结果（pass/fail + answer/ref）
- `score`：汇总指标（dataset/model/cot）
- `logs`：调度/运行事件
- `view_score_latest`：最新 score 视图（按 dataset/model/cot）
- `view_eval_completion`：仅最新版本的 eval+completions 样本级视图

## 字段字典

### version（版本/运行元信息）
- `id`：UUID 主键
- `job_name` / `job_id`：调度任务信息
- `dataset` / `model`：评测维度
- `git_sha`：代码版本
- `is_param_search`：是否网格选参
- `created_at`：创建时间

### completions（输出轨迹）
- `id`：UUID 主键
- `version_id`：版本外键
- `benchmark_name` / `dataset_split` / `sample_index` / `repeat_index`
- `sampling_config`：采样参数（JSONB）
- `context`：阶段化上下文（JSONB）

### eval（样本评测）
- `id`：UUID 主键
- `version_id`：版本外键
- `benchmark_name` / `dataset_split` / `sample_index` / `repeat_index`
- `context` / `answer` / `ref_answer`
- `is_passed` / `fail_reason`

### score（汇总指标）
- `id`：UUID 主键
- `version_id`：版本外键
- `dataset` / `model` / `cot`
- `metrics`：指标汇总（JSONB）
- `samples` / `problems` / `created_at`
- `log_path` / `task` / `task_details`

### logs（运行事件）
- `id`：UUID 主键
- `version_id`：可空外键
- `event` / `job_id` / `payload`
- `created_at`

### view_score_latest
按 `(dataset, model, cot)` 维度选择 `created_at` 最新的一条 score。

### view_eval_completion
仅包含 `view_score_latest` 对应 `version_id` 的样本级结果，用于保证“最新分数对应最新评测内容”。

## 常用查询
1) 最新分数列表（含 job 信息）
```sql
SELECT s.*, v.job_name, v.job_id
FROM view_score_latest s
LEFT JOIN version v ON v.id = s.version_id
ORDER BY s.created_at DESC;
```

2) 按数据集/模型筛选历史分数
```sql
SELECT *
FROM score
WHERE dataset = $1 AND model = $2 AND is_param_search = $3
ORDER BY created_at DESC;
```

3) 最新版本样本级明细（含输出轨迹）
```sql
SELECT *
FROM view_eval_completion
WHERE version_id = $1
ORDER BY sample_index, repeat_index;
```
