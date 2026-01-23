## 目的
本项目当前同时保留 JSONL 输出与数据库写入（便于对比）。前后端主要通过数据库读取评测结果；本文说明表结构与字段含义，并给出常用查询与交接操作指引。

## 表总览（5 张 + 2 视图）
- `version`：一次评测版本/运行元信息（job/dataset/model/git）
- `completions`：模型输出轨迹（prompt/complete/stop 原始信息）
- `eval`：评测结果（样本级 pass/fail + answer/ref）
- `score`：评测汇总指标（按 dataset/model/cot）
- `logs`：调度与运行事件日志
- `view_score_latest`：按 dataset/model/cot 取最新 score
- `view_eval_completion`：eval 与 completions 的样本级 join

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
按样本维度 join `eval` 与 `completions`，用于定位具体输出与评测结果。

## 常用查询（前端可直接使用）
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

3) 样本级明细（含输出轨迹）
```sql
SELECT *
FROM view_eval_completion
WHERE version_id = $1
ORDER BY sample_index, repeat_index;
```

## 当前流程（简述）
- 评测流程生成 JSONL（保留用于对比）
- 同时写入数据库：`version`/`score`/`completions`/`eval`/`logs`
- `view_score_latest` 提供最新结果视图，`view_eval_completion` 提供样本级联查

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
