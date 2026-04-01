from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.eval.scheduler.config import DBConfig

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None
_INITIALIZED = False
_RECOVERY_APPLIED = False


def _build_db_url(config: DBConfig, dbname: str | None = None) -> str:
    db = dbname if dbname is not None else config.dbname
    return (
        f"postgresql+psycopg://{config.user}:{config.password}"
        f"@{config.host}:{config.port}/{db}"
    )


def _engine_connect_args(config: DBConfig) -> dict[str, str]:
    sslmode = str(getattr(config, "sslmode", "") or "").strip()
    return {"sslmode": sslmode} if sslmode else {}


def init_orm(config: DBConfig | None = None) -> None:
    """Initialize ORM engine and session factory.

    Can be called multiple times safely - only initializes once.
    Caller must provision the target PostgreSQL database and schema first.
    If config is None, uses DEFAULT_DB_CONFIG.
    """
    global _ENGINE, _SESSION_FACTORY, _INITIALIZED, _RECOVERY_APPLIED
    if _INITIALIZED:
        return
    if config is None:
        from src.eval.scheduler.config import DEFAULT_DB_CONFIG
        config = DEFAULT_DB_CONFIG
    _ENGINE = create_engine(
        _build_db_url(config),
        pool_pre_ping=True,
        future=True,
        connect_args=_engine_connect_args(config),
    )
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, class_=Session)
    if getattr(config, "startup_recovery", False) and not _RECOVERY_APPLIED:
        _recover_running_tasks(_ENGINE)
        _RECOVERY_APPLIED = True
    _INITIALIZED = True


def _recover_running_tasks(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE task
                SET status = 'Failed'
                WHERE status IN ('running', 'Running')
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE completions
                SET status = 'Failed'
                WHERE status IN ('running', 'Running')
                """
            )
        )


def is_initialized() -> bool:
    """Check if ORM has been initialized."""
    return _INITIALIZED


@contextmanager
def get_session(existing: Session | None = None) -> Iterator[Session]:
    """Get a database session.

    If `existing` is provided, yields it without managing lifecycle (for transaction composition).
    Otherwise creates a new session with auto-commit/rollback.
    """
    if existing is not None:
        yield existing
        return
    if _SESSION_FACTORY is None:
        raise RuntimeError("ORM not initialized. Call init_orm() first.")
    session = _SESSION_FACTORY()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def transaction() -> Iterator[Session]:
    """Create a transaction scope for composing multiple operations.

    Usage:
        with transaction() as session:
            repo.insert_benchmark(session, ...)
            repo.insert_model(session, ...)
            # All operations commit together or rollback together
    """
    if _SESSION_FACTORY is None:
        raise RuntimeError("ORM not initialized. Call init_orm() first.")
    session = _SESSION_FACTORY()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Base(DeclarativeBase):
    pass


class Benchmark(Base):
    __tablename__ = "benchmark"

    benchmark_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_name: Mapped[str] = mapped_column(String(255), nullable=False)
    benchmark_split: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="benchmark", lazy="select")

    __table_args__ = (
        CheckConstraint(
            "status IN ('Todo', 'Buggy', 'Low', 'DataSynthesizing', 'Completed')",
            name="chk_benchmark_status",
        ),
        UniqueConstraint("benchmark_name", "benchmark_split", name="uq_benchmark_name_split"),
    )


class Model(Base):
    __tablename__ = "model"

    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_version: Mapped[str] = mapped_column(String(255), nullable=False)
    arch_version: Mapped[str] = mapped_column(String(255), nullable=False)
    num_params: Mapped[str] = mapped_column(String(255), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationships
    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="model", lazy="select")

    __table_args__ = (
        UniqueConstraint(
            "arch_version",
            "data_version",
            "num_params",
            "model_name",
            name="uq_model_identity",
        ),
    )


class Task(Base):
    __tablename__ = "task"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evaluator: Mapped[str] = mapped_column(String(255), nullable=False)
    is_param_search: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_tmp: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)
    status: Mapped[str] = mapped_column(String(255), nullable=False)
    git_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model.model_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    benchmark_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("benchmark.benchmark_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    desc: Mapped[Optional[str]] = mapped_column("desc", Text, nullable=True)
    sampling_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    log_path: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="tasks", lazy="joined")
    benchmark: Mapped["Benchmark"] = relationship("Benchmark", back_populates="tasks", lazy="joined")
    completions: Mapped[list["Completion"]] = relationship("Completion", back_populates="task", lazy="select")
    scores: Mapped[list["Score"]] = relationship("Score", back_populates="task", lazy="select")

    __table_args__ = (
        Index("idx_task_model", "model_id"),
        Index("idx_task_benchmark", "benchmark_id"),
        Index("idx_task_is_tmp_created_at", "is_tmp", "created_at"),
        Index("idx_task_status_created_at", "status", "created_at"),
        Index("idx_task_identity_lookup", "model_id", "benchmark_id", "evaluator", "git_hash", "config_path"),
        CheckConstraint("status IN ('Running', 'Completed', 'Failed')", name="chk_task_status"),
    )


class Completion(Base):
    __tablename__ = "completions"

    completions_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("task.task_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    context: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_repeat_index: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)
    status: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="completions", lazy="select")
    evals: Mapped[list["Eval"]] = relationship("Eval", back_populates="completion", lazy="select")
    checker: Mapped["Checker | None"] = relationship("Checker", back_populates="completion", lazy="select")

    __table_args__ = (
        Index("idx_completions_task", "task_id"),
        Index("idx_completions_task_status", "task_id", "status"),
        Index("uq_completions_sample", "task_id", "sample_index", "avg_repeat_index", "pass_index", unique=True),
        CheckConstraint("sample_index >= 0", name="chk_completions_sample_index"),
        CheckConstraint("avg_repeat_index >= 0", name="chk_completions_avg_repeat_index"),
        CheckConstraint("pass_index >= 0", name="chk_completions_pass_index"),
        CheckConstraint("status IN ('Running', 'Completed', 'Failed')", name="chk_completions_status"),
    )


class Eval(Base):
    __tablename__ = "eval"

    eval_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    completions_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("completions.completions_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    ref_answer: Mapped[str] = mapped_column(Text, nullable=False)
    is_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fail_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    # Relationships
    completion: Mapped["Completion"] = relationship("Completion", back_populates="evals", lazy="select")

    __table_args__ = (
        Index("idx_eval_completion", "completions_id"),
        UniqueConstraint("completions_id", name="uq_eval_completion"),
    )


class Checker(Base):
    __tablename__ = "checker"

    checker_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    completions_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("completions.completions_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    answer_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    instruction_following_error: Mapped[bool] = mapped_column(Boolean, nullable=False)
    world_knowledge_error: Mapped[bool] = mapped_column(Boolean, nullable=False)
    math_error: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reasoning_logic_error: Mapped[bool] = mapped_column(Boolean, nullable=False)
    thought_contains_correct_answer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    completion: Mapped["Completion"] = relationship("Completion", back_populates="checker", lazy="select")

    __table_args__ = (
        Index("idx_checker_completion", "completions_id"),
        Index("idx_checker_needs_human_review", "needs_human_review"),
        UniqueConstraint("completions_id", name="uq_checker_completion"),
    )


class Score(Base):
    __tablename__ = "scores"

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("task.task_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    cot_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="scores", lazy="select")

    __table_args__ = (
        Index("idx_scores_task", "task_id"),
        UniqueConstraint("task_id", name="uq_scores_task"),
        CheckConstraint("cot_mode IN ('NoCoT', 'FakeCoT', 'CoT')", name="chk_scores_cot_mode"),
    )


__all__ = [
    "Base",
    "Benchmark",
    "Checker",
    "Completion",
    "Eval",
    "Model",
    "Score",
    "Task",
    "get_session",
    "init_orm",
    "is_initialized",
    "transaction",
]
