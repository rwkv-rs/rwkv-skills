-- Extensions needed for UUID generation.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Table: version (task run metadata)
CREATE TABLE IF NOT EXISTS version (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name TEXT,
    job_id TEXT,
    dataset TEXT,
    model TEXT,
    git_sha TEXT,
    is_param_search BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_version_created_at
    ON version(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_version_job
    ON version(job_name, job_id);
CREATE INDEX IF NOT EXISTS idx_version_model_dataset
    ON version(model, dataset);
CREATE INDEX IF NOT EXISTS idx_version_param_search
    ON version(is_param_search);

-- Table: completions (results/completions/*.jsonl)
CREATE TABLE IF NOT EXISTS completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES version(id) ON DELETE CASCADE,
    is_param_search BOOLEAN NOT NULL DEFAULT FALSE,
    benchmark_name TEXT NOT NULL,
    dataset_split TEXT NOT NULL,
    sample_index INT NOT NULL,
    repeat_index INT NOT NULL,
    sampling_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    context JSONB
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_completions_sample
    ON completions(version_id, benchmark_name, dataset_split, sample_index, repeat_index);
CREATE INDEX IF NOT EXISTS idx_completions_lookup
    ON completions(benchmark_name, dataset_split, sample_index);
CREATE INDEX IF NOT EXISTS idx_completions_version
    ON completions(version_id, is_param_search);

-- Table: eval (results/eval/*_results.jsonl)
CREATE TABLE IF NOT EXISTS eval (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES version(id) ON DELETE CASCADE,
    is_param_search BOOLEAN NOT NULL DEFAULT FALSE,
    benchmark_name TEXT NOT NULL,
    dataset_split TEXT NOT NULL,
    sample_index INT NOT NULL,
    repeat_index INT NOT NULL,
    context TEXT,
    answer TEXT,
    ref_answer TEXT,
    is_passed BOOLEAN NOT NULL DEFAULT FALSE,
    fail_reason TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_sample
    ON eval(version_id, benchmark_name, dataset_split, sample_index, repeat_index);
CREATE INDEX IF NOT EXISTS idx_eval_lookup
    ON eval(benchmark_name, dataset_split, sample_index);
CREATE INDEX IF NOT EXISTS idx_eval_version
    ON eval(version_id, is_param_search);

-- Table: score (results/scores/*.json)
CREATE TABLE IF NOT EXISTS score (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES version(id) ON DELETE CASCADE,
    is_param_search BOOLEAN NOT NULL DEFAULT FALSE,
    dataset TEXT NOT NULL,
    model TEXT NOT NULL,
    cot BOOLEAN NOT NULL DEFAULT FALSE,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    samples INT NOT NULL,
    problems INT,
    created_at TIMESTAMPTZ NOT NULL,
    log_path TEXT,
    task TEXT,
    task_details JSONB
);
CREATE INDEX IF NOT EXISTS idx_score_dataset_model
    ON score(dataset, model, cot);
CREATE INDEX IF NOT EXISTS idx_score_created_at
    ON score(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_score_version
    ON score(version_id, is_param_search);

-- Table: logs (results/logs/*.log)
CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID REFERENCES version(id) ON DELETE SET NULL,
    event TEXT NOT NULL,
    job_id TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_logs_job_id
    ON logs(job_id);
CREATE INDEX IF NOT EXISTS idx_logs_created_at
    ON logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_version
    ON logs(version_id);


-- View: latest score per dataset/model/cot for quick listing.
CREATE OR REPLACE VIEW view_score_latest AS
SELECT
    version_id,
    dataset,
    model,
    cot,
    metrics,
    samples,
    problems,
    created_at,
    log_path,
    task,
    task_details
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY dataset, model, cot
            ORDER BY created_at DESC
        ) AS rn
    FROM score
    WHERE is_param_search = FALSE
) latest
WHERE latest.rn = 1;

-- View: join eval with completions for sample-level inspection.
CREATE OR REPLACE VIEW view_eval_completion AS
SELECT
    e.version_id,
    e.benchmark_name,
    e.dataset_split,
    e.sample_index,
    e.repeat_index,
    e.context,
    e.answer,
    e.ref_answer,
    e.is_passed,
    e.fail_reason,
    c.sampling_config,
    c.context AS completion_context
FROM eval e
JOIN completions c
  ON c.version_id = e.version_id
 AND c.benchmark_name = e.benchmark_name
 AND c.dataset_split = e.dataset_split
 AND c.sample_index = e.sample_index
 AND c.repeat_index = e.repeat_index
WHERE e.is_param_search = FALSE
  AND c.is_param_search = FALSE;
