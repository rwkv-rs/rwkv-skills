-- 扩展：UUID 生成函数，需在使用 gen_random_uuid() 前启用。
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 表：评测主题（前端：主题列表/过滤）。
CREATE TABLE IF NOT EXISTS eval_subject (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：数据集标识（slug）。
    dataset_slug VARCHAR(255) NOT NULL,
    -- 字段：领域/学科。
    domain VARCHAR(128),
    -- 字段：数据集版本。
    dataset_version VARCHAR(128),
    -- 字段：数据集元数据（JSON）。
    dataset_meta JSONB,
    -- 字段：模型标识（slug）。
    model_slug VARCHAR(255) NOT NULL,
    -- 字段：模型名称（展示）。
    model_name VARCHAR(255),
    -- 字段：模型版本/修订号。
    model_revision VARCHAR(255),
    -- 字段：模型提供方。
    provider VARCHAR(128),
    -- 字段：模型元数据（JSON）。
    model_meta JSONB,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：主题唯一键（数据集 + 模型 + 修订号）。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_subject_key
ON eval_subject(dataset_slug, model_slug, COALESCE(model_revision, ''));
-- 索引：按数据集过滤。
CREATE INDEX IF NOT EXISTS idx_eval_subject_dataset ON eval_subject(dataset_slug);
-- 索引：按模型过滤。
CREATE INDEX IF NOT EXISTS idx_eval_subject_model ON eval_subject(model_slug);
-- 索引：按领域过滤。
CREATE INDEX IF NOT EXISTS idx_eval_subject_domain ON eval_subject(domain);

-- 变更：为 split 表补充主题外键。
ALTER TABLE IF EXISTS eval_split ADD COLUMN IF NOT EXISTS subject_id UUID;
-- 变更：为 sample 表补充主题外键。
ALTER TABLE IF EXISTS eval_sample ADD COLUMN IF NOT EXISTS subject_id UUID;
-- 变更：为 task 表补充主题外键。
ALTER TABLE IF EXISTS eval_task ADD COLUMN IF NOT EXISTS subject_id UUID;

-- 表：数据划分（前端：train/val/test 标签页）。
CREATE TABLE IF NOT EXISTS eval_split (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属主题。
    subject_id UUID NOT NULL REFERENCES eval_subject(id) ON DELETE CASCADE,
    -- 字段：划分名称（train/val/test）。
    split_name VARCHAR(128) NOT NULL,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：主题 + 划分名称唯一。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_split_subject_name ON eval_split(subject_id, split_name);
-- 索引：按主题过滤。
CREATE INDEX IF NOT EXISTS idx_eval_split_subject ON eval_split(subject_id);

-- 表：样本（前端：样本列表/详情）。
CREATE TABLE IF NOT EXISTS eval_sample (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属主题。
    subject_id UUID NOT NULL REFERENCES eval_subject(id) ON DELETE CASCADE,
    -- 字段：所属划分。
    split_id UUID NOT NULL REFERENCES eval_split(id) ON DELETE CASCADE,
    -- 字段：样本序号（同一划分内）。
    sample_index INT NOT NULL,
    -- 字段：问题/输入。
    question TEXT,
    -- 字段：参考答案。
    reference_answer TEXT,
    -- 字段：样本元数据（JSON）。
    meta JSONB,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：样本唯一键（主题 + 划分 + 序号）。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_sample_lookup ON eval_sample(subject_id, split_id, sample_index);
-- 索引：按主题/划分/序号检索。
CREATE INDEX IF NOT EXISTS idx_eval_sample_lookup ON eval_sample(subject_id, split_id, sample_index);

-- 表：任务（前端：任务列表/过滤）。
CREATE TABLE IF NOT EXISTS eval_task (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：任务外部 ID。
    task_id VARCHAR(255) NOT NULL,
    -- 字段：所属主题。
    subject_id UUID NOT NULL REFERENCES eval_subject(id) ON DELETE CASCADE,
    -- 字段：任务标签。
    task_tag VARCHAR(255),
    -- 字段：任务元数据（JSON）。
    meta JSONB,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：任务 ID 唯一。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_task_id ON eval_task(task_id);
-- 索引：按主题过滤。
CREATE INDEX IF NOT EXISTS idx_eval_task_subject ON eval_task(subject_id);

-- 表：运行（前端：运行列表/状态）。
CREATE TABLE IF NOT EXISTS eval_run (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属任务。
    task_id UUID NOT NULL REFERENCES eval_task(id) ON DELETE CASCADE,
    -- 字段：运行标签。
    run_tag VARCHAR(255),
    -- 字段：采样配置（JSON）。
    sampling_config JSONB,
    -- 字段：运行时配置（JSON）。
    runtime_config JSONB,
    -- 字段：代码版本。
    code_version VARCHAR(255),
    -- 字段：运行状态。
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    -- 字段：错误信息。
    error_msg TEXT,
    -- 字段：开始时间。
    started_at TIMESTAMPTZ,
    -- 字段：结束时间。
    finished_at TIMESTAMPTZ,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：按任务 + 状态过滤。
CREATE INDEX IF NOT EXISTS idx_eval_run_task_status ON eval_run(task_id, status);
-- 索引：按运行标签过滤。
CREATE INDEX IF NOT EXISTS idx_eval_run_run_tag ON eval_run(run_tag);
-- 索引：任务 + 运行标签唯一。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_task_tag ON eval_run(task_id, run_tag);

-- 表：运行样本（前端：样本运行状态）。
CREATE TABLE IF NOT EXISTS eval_run_sample (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属运行。
    run_id UUID NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    -- 字段：所属样本。
    sample_id UUID NOT NULL REFERENCES eval_sample(id) ON DELETE CASCADE,
    -- 字段：重复次数序号。
    repeat_index INT NOT NULL,
    -- 字段：样本运行状态。
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    -- 字段：当前阶段。
    current_stage VARCHAR(32),
    -- 字段：最新尝试序号。
    latest_attempt_index INT DEFAULT 0,
    -- 字段：错误信息。
    error_msg TEXT,
    -- 字段：开始时间。
    started_at TIMESTAMPTZ,
    -- 字段：结束时间。
    finished_at TIMESTAMPTZ,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：运行样本唯一键（运行 + 样本 + 重复）。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_sample_unique ON eval_run_sample(run_id, sample_id, repeat_index);
-- 索引：按运行 + 状态过滤。
CREATE INDEX IF NOT EXISTS idx_eval_run_sample_status ON eval_run_sample(run_id, status);
-- 索引：按样本过滤。
CREATE INDEX IF NOT EXISTS idx_eval_run_sample_sample ON eval_run_sample(sample_id);

-- 表：尝试（前端：调试/重试详情）。
CREATE TABLE IF NOT EXISTS eval_attempt (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属运行样本。
    run_sample_id UUID NOT NULL REFERENCES eval_run_sample(id) ON DELETE CASCADE,
    -- 字段：尝试序号。
    attempt_index INT NOT NULL,
    -- 字段：工作进程标识。
    worker_id VARCHAR(255),
    -- 字段：分片编号。
    shard_id INT,
    -- 字段：分片总数。
    shard_count INT,
    -- 字段：随机种子。
    seed BIGINT,
    -- 字段：尝试状态。
    status VARCHAR(32) NOT NULL DEFAULT 'running',
    -- 字段：错误信息。
    error_msg TEXT,
    -- 字段：开始时间。
    started_at TIMESTAMPTZ,
    -- 字段：结束时间。
    finished_at TIMESTAMPTZ,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 字段：更新时间。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：尝试唯一键（运行样本 + 尝试序号）。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_attempt_unique ON eval_attempt(run_sample_id, attempt_index);
-- 索引：按运行样本 + 尝试序号倒序。
CREATE INDEX IF NOT EXISTS idx_eval_attempt_rs ON eval_attempt(run_sample_id, attempt_index DESC);

-- 表：阶段输出（前端：prompt/completion 展示）。
CREATE TABLE IF NOT EXISTS eval_stage_output (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属尝试。
    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
    -- 字段：阶段名称。
    stage VARCHAR(32) NOT NULL,
    -- 字段：序号（多段输出时排序）。
    seq INT NOT NULL DEFAULT 0,
    -- 字段：提示词文本。
    prompt TEXT,
    -- 字段：模型输出文本。
    completion TEXT,
    -- 字段：结束原因。
    finish_reason VARCHAR(32),
    -- 字段：提供方请求 ID。
    provider_request_id VARCHAR(255),
    -- 字段：原始响应（JSON）。
    raw_response JSONB,
    -- 字段：提示词 token 数。
    token_count_prompt INT,
    -- 字段：输出 token 数。
    token_count_response INT,
    -- 字段：延迟毫秒数。
    latency_ms INT,
    -- 字段：费用（美元）。
    cost_usd NUMERIC,
    -- 字段：是否为部分输出。
    is_partial BOOLEAN NOT NULL DEFAULT FALSE,
    -- 字段：是否为最终输出。
    is_final BOOLEAN NOT NULL DEFAULT FALSE,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：按尝试 + 阶段 + 时间倒序获取最新输出。
CREATE INDEX IF NOT EXISTS idx_eval_stage_latest ON eval_stage_output(attempt_id, stage, created_at DESC);
-- 索引：按阶段 + 完成原因过滤。
CREATE INDEX IF NOT EXISTS idx_eval_stage_finish ON eval_stage_output(stage, finish_reason);
-- 索引：每个尝试/阶段仅保留一条最终输出。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_stage_final ON eval_stage_output(attempt_id, stage) WHERE is_final;

-- 表：CoT 检查点（前端：中间过程查看）。
CREATE TABLE IF NOT EXISTS eval_cot_checkpoint (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属尝试。
    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
    -- 字段：阶段名称。
    stage VARCHAR(32) NOT NULL,
    -- 字段：token 偏移量。
    token_offset INT,
    -- 字段：部分输出文本。
    partial_completion TEXT,
    -- 字段：KV 缓存引用。
    kv_cache_ref TEXT,
    -- 字段：随机状态（JSON）。
    rng_state JSONB,
    -- 字段：状态。
    status VARCHAR(32),
    -- 字段：是否为最新检查点。
    latest BOOLEAN NOT NULL DEFAULT TRUE,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：每个尝试/阶段仅保留一个 latest。
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_ckpt_latest ON eval_cot_checkpoint(attempt_id, stage) WHERE latest;
-- 索引：按尝试 + 阶段 + 时间倒序查询。
CREATE INDEX IF NOT EXISTS idx_eval_ckpt_lookup ON eval_cot_checkpoint(attempt_id, stage, created_at DESC);

-- 表：指标（前端：指标列表）。
CREATE TABLE IF NOT EXISTS eval_metric (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属运行样本。
    run_sample_id UUID NOT NULL REFERENCES eval_run_sample(id) ON DELETE CASCADE,
    -- 字段：指标名称。
    name VARCHAR(255) NOT NULL,
    -- 字段：数值型指标。
    value_num DOUBLE PRECISION,
    -- 字段：文本型指标。
    value_text TEXT,
    -- 字段：指标元数据（JSON）。
    meta JSONB,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 索引：按指标名称过滤。
CREATE INDEX IF NOT EXISTS idx_eval_metric_name ON eval_metric(name);
-- 索引：按运行样本 + 指标名称过滤。
CREATE INDEX IF NOT EXISTS idx_eval_metric_rs ON eval_metric(run_sample_id, name);

-- 表：运行事件（前端：事件时间线）。
CREATE TABLE IF NOT EXISTS eval_run_event (
    -- 字段：主键 UUID。
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 字段：所属运行（可为空）。
    run_id UUID REFERENCES eval_run(id) ON DELETE CASCADE,
    -- 字段：所属运行样本（可为空）。
    run_sample_id UUID REFERENCES eval_run_sample(id) ON DELETE CASCADE,
    -- 字段：事件类型。
    event_type VARCHAR(255) NOT NULL,
    -- 字段：事件消息。
    message TEXT,
    -- 字段：事件元数据（JSON）。
    meta JSONB,
    -- 字段：创建时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 约束：运行和运行样本至少其一不为空。
    CHECK (run_id IS NOT NULL OR run_sample_id IS NOT NULL)
);
-- 索引：按运行 + 时间排序。
CREATE INDEX IF NOT EXISTS idx_eval_event_run_time ON eval_run_event(run_id, created_at);
