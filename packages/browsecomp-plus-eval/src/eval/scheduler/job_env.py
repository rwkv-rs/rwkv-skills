from __future__ import annotations

import os
import re
import time


_JOB_ID_ENV = "RWKV_SKILLS_JOB_ID"


def _normalize_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "job"


def ensure_job_id(prefix: str) -> str:
    current = os.environ.get(_JOB_ID_ENV)
    if current:
        return current
    job_id = f"{_normalize_prefix(prefix)}_{int(time.time())}"
    os.environ[_JOB_ID_ENV] = job_id
    return job_id
