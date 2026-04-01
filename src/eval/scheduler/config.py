

"""Shared configuration primitives for the scheduler stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

from src.eval.env_config import load_env_file


REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(REPO_ROOT / ".env")
RESULTS_ROOT = Path(os.environ.get("RUN_RESULTS_DIR", REPO_ROOT / "results"))


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return default


DEFAULT_LOG_DIR = _path_env("RUN_LOG_DIR", RESULTS_ROOT / "scores")
DEFAULT_COMPLETION_DIR = _path_env("RUN_COMPLETION_DIR", RESULTS_ROOT / "completions")
DEFAULT_EVAL_RESULT_DIR = _path_env("RUN_EVAL_RESULT_DIR", RESULTS_ROOT / "eval")
DEFAULT_CHECK_RESULT_DIR = _path_env("RUN_CHECK_RESULT_DIR", RESULTS_ROOT / "check")
DEFAULT_RUN_LOG_DIR = _path_env("RUN_RUN_LOG_DIR", RESULTS_ROOT / "logs")
DEFAULT_PID_DIR = _path_env("RUN_PID_DIR", RESULTS_ROOT / "pids")

_MODEL_GLOBS_ENV = os.environ.get("RUN_MODEL_GLOBS")
if _MODEL_GLOBS_ENV:
    DEFAULT_MODEL_GLOBS = tuple(filter(None, _MODEL_GLOBS_ENV.split(os.pathsep)))
else:
    DEFAULT_MODEL_GLOBS = (
        "/public/home/ssjxzkz/Weights/BlinkDL__rwkv7-g1/*.pth",
        "/home/caizus/Weights/BlinkDL__rwkv7-g1/*.pth",
        str(REPO_ROOT / "weights" / "rwkv7-*.pth"),
    )

DEFAULT_GPU_IDLE_MAX_MEM = int(os.environ.get("RUN_GPU_IDLE_MAX_MEM", "1000"))
DEFAULT_PYTHON = os.environ.get("RUN_PYTHON", sys.executable or "python3")
DEFAULT_DISPATCH_POLL_SECONDS = int(os.environ.get("RUN_DISPATCH_POLL", "30"))
DEFAULT_TAIL_LINES = int(os.environ.get("RUN_TAIL_LINES", "60"))
DEFAULT_ROTATE_SECONDS = int(os.environ.get("RUN_ROTATE_SECONDS", "15"))


@dataclass(slots=True)
class SchedulerPaths:
    log_dir: Path = DEFAULT_LOG_DIR
    pid_dir: Path = DEFAULT_PID_DIR
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR


@dataclass(slots=True)
class QueueCliDefaults:
    model_globs: tuple[str, ...] = DEFAULT_MODEL_GLOBS
    gpu_idle_max_mem: int = DEFAULT_GPU_IDLE_MAX_MEM
    dispatch_poll_seconds: int = DEFAULT_DISPATCH_POLL_SECONDS
    tail_lines: int = DEFAULT_TAIL_LINES
    rotate_seconds: int = DEFAULT_ROTATE_SECONDS


__all__ = [
    "REPO_ROOT",
    "RESULTS_ROOT",
    "SchedulerPaths",
    "QueueCliDefaults",
    "DEFAULT_LOG_DIR",
    "DEFAULT_COMPLETION_DIR",
    "DEFAULT_EVAL_RESULT_DIR",
    "DEFAULT_CHECK_RESULT_DIR",
    "DEFAULT_PID_DIR",
    "DEFAULT_RUN_LOG_DIR",
    "DEFAULT_MODEL_GLOBS",
    "DEFAULT_GPU_IDLE_MAX_MEM",
    "DEFAULT_PYTHON",
    "DEFAULT_DISPATCH_POLL_SECONDS",
    "DEFAULT_TAIL_LINES",
    "DEFAULT_ROTATE_SECONDS",
    "DBConfig",
    "DEFAULT_DB_CONFIG",
]


@dataclass(slots=True)
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "rwkv-eval"
    sslmode: str = "prefer"
    startup_recovery: bool = False


def _env_first(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def _env_bool(*names: str, default: bool) -> bool:
    raw = _env_first(*names, default="1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_db_config() -> DBConfig:
    return DBConfig(
        host=_env_first("RWKV_EVAL_SPACE_DB_HOST", "PG_HOST", default="localhost"),
        port=int(_env_first("RWKV_EVAL_SPACE_DB_PORT", "PG_PORT", default="5432")),
        user=_env_first("RWKV_EVAL_SPACE_DB_USERNAME", "PG_USER", default="postgres"),
        password=_env_first("RWKV_EVAL_SPACE_DB_PASSWORD", "PG_PASSWORD", default=""),
        dbname=_env_first("RWKV_EVAL_SPACE_DB_DATABASE_NAME", "PG_DBNAME", default="rwkv-eval"),
        sslmode=_env_first("RWKV_EVAL_SPACE_DB_SSLMODE", "PG_SSLMODE", default="prefer"),
        startup_recovery=_env_bool(
            "RWKV_EVAL_STARTUP_RECOVERY",
            "RUN_STARTUP_RECOVERY",
            default=False,
        ),
    )

DEFAULT_DB_CONFIG = _load_db_config()
