-- Extensions needed for UUID generation.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Table: evaluation runs (front-end main list).
CREATE TABLE IF NOT EXISTS eval_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_name VARCHAR(128) NOT NULL,
    dataset VARCHAR(255) NOT NULL,
    dataset_split VARCHAR(128) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_slug VARCHAR(255),
    model_revision VARCHAR(255),
    model_path TEXT,
    cot BOOLEAN NOT NULL DEFAULT FALSE,
    run_tag VARCHAR(255),
    sampling_config JSONB,
    runtime_config JSONB,
    code_version VARCHAR(255),
    task VARCHAR(128),
    task_details JSONB,
    metrics JSONB,
    samples INT,
    problems INT,
    log_path TEXT,
    eval_details_path TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_eval_run_dataset ON eval_run(dataset);
CREATE INDEX IF NOT EXISTS idx_eval_run_model ON eval_run(model_name);
CREATE INDEX IF NOT EXISTS idx_eval_run_status ON eval_run(status);
CREATE INDEX IF NOT EXISTS idx_eval_run_cot ON eval_run(cot);
CREATE INDEX IF NOT EXISTS idx_eval_run_created ON eval_run(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_run_lookup
    ON eval_run(benchmark_name, dataset, dataset_split, model_name, cot);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_tag
    ON eval_run(benchmark_name, dataset, dataset_split, model_name, cot, run_tag);

-- Table: benchmark samples (deduplicated by benchmark+split+index).
CREATE TABLE IF NOT EXISTS eval_sample (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_name VARCHAR(128) NOT NULL,
    dataset_split VARCHAR(128) NOT NULL,
    sample_index INT NOT NULL,
    question TEXT,
    ref_answer TEXT,
    meta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_sample_key
    ON eval_sample(benchmark_name, dataset_split, sample_index);
CREATE INDEX IF NOT EXISTS idx_eval_sample_lookup
    ON eval_sample(benchmark_name, dataset_split);

-- Table: run-sample status and summary result.
CREATE TABLE IF NOT EXISTS eval_run_sample (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    sample_id UUID NOT NULL REFERENCES eval_sample(id) ON DELETE CASCADE,
    repeat_index INT NOT NULL,
    current_stage VARCHAR(32),
    latest_attempt_index INT DEFAULT 0,
    answer TEXT,
    is_passed BOOLEAN,
    fail_reason TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_sample_key
    ON eval_run_sample(run_id, sample_id, repeat_index);
CREATE INDEX IF NOT EXISTS idx_eval_run_sample_run
    ON eval_run_sample(run_id, status);
CREATE INDEX IF NOT EXISTS idx_eval_run_sample_sample
    ON eval_run_sample(sample_id);

-- Table: attempts for retry/worker execution.
CREATE TABLE IF NOT EXISTS eval_attempt (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_sample_id UUID NOT NULL REFERENCES eval_run_sample(id) ON DELETE CASCADE,
    attempt_index INT NOT NULL,
    worker_id VARCHAR(255),
    shard_id INT,
    shard_count INT,
    seed BIGINT,
    status VARCHAR(32) NOT NULL DEFAULT 'running',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_attempt_key
    ON eval_attempt(run_sample_id, attempt_index);
CREATE INDEX IF NOT EXISTS idx_eval_attempt_status
    ON eval_attempt(run_sample_id, status);

-- Table: stage output (prompt/completion).
CREATE TABLE IF NOT EXISTS eval_stage_output (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
    stage VARCHAR(32) NOT NULL,
    seq INT NOT NULL DEFAULT 0,
    prompt TEXT,
    completion TEXT,
    finish_reason VARCHAR(32),
    provider_request_id VARCHAR(255),
    raw_response JSONB,
    token_count_prompt INT,
    token_count_response INT,
    latency_ms INT,
    cost_usd NUMERIC,
    is_partial BOOLEAN NOT NULL DEFAULT FALSE,
    is_final BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_eval_stage_latest
    ON eval_stage_output(attempt_id, stage, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_stage_finish
    ON eval_stage_output(stage, finish_reason);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_stage_final
    ON eval_stage_output(attempt_id, stage) WHERE is_final;

-- Table: CoT checkpoints for resume.
CREATE TABLE IF NOT EXISTS eval_cot_checkpoint (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
    stage VARCHAR(32) NOT NULL,
    token_offset INT,
    partial_completion TEXT,
    kv_cache_ref TEXT,
    rng_state JSONB,
    status VARCHAR(32),
    latest BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_ckpt_latest
    ON eval_cot_checkpoint(attempt_id, stage) WHERE latest;
CREATE INDEX IF NOT EXISTS idx_eval_ckpt_lookup
    ON eval_cot_checkpoint(attempt_id, stage, created_at DESC);
