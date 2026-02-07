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


def _build_db_url(config: DBConfig, dbname: str | None = None) -> str:
    db = dbname if dbname is not None else config.dbname
    return (
        f"postgresql+psycopg://{config.user}:{config.password}"
        f"@{config.host}:{config.port}/{db}"
    )


def _ensure_database_exists(config: DBConfig) -> None:
    """Connect to 'postgres' db and create target database if it doesn't exist."""
    admin_engine = create_engine(
        _build_db_url(config, dbname="postgres"),
        isolation_level="AUTOCOMMIT",
    )
    with admin_engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
            {"dbname": config.dbname},
        )
        if result.fetchone() is None:
            conn.execute(text(f'CREATE DATABASE "{config.dbname}"'))
    admin_engine.dispose()


def init_orm(config: DBConfig | None = None) -> None:
    """Initialize ORM engine and session factory.

    Can be called multiple times safely - only initializes once.
    If config is None, uses DEFAULT_DB_CONFIG.
    """
    global _ENGINE, _SESSION_FACTORY, _INITIALIZED
    if _INITIALIZED:
        return
    if config is None:
        from src.eval.scheduler.config import DEFAULT_DB_CONFIG
        config = DEFAULT_DB_CONFIG
    _ensure_database_exists(config)
    _ENGINE = create_engine(_build_db_url(config), pool_pre_ping=True, future=True)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, class_=Session)
    Base.metadata.create_all(_ENGINE)
    _INITIALIZED = True


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
            "model_name", "arch_version", "data_version", "num_params",
            name="uq_model_identity",
        ),
    )


class Task(Base):
    __tablename__ = "task"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evaluator: Mapped[str] = mapped_column(String(255), nullable=False)
    is_param_search: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
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
    repeat_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)
    status: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="completions", lazy="select")
    evals: Mapped[list["Eval"]] = relationship("Eval", back_populates="completion", lazy="select")

    __table_args__ = (
        Index("idx_completions_task", "task_id"),
        Index("uq_completions_sample", "task_id", "sample_index", "repeat_index", unique=True),
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
    fail_reason: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    # Relationships
    completion: Mapped["Completion"] = relationship("Completion", back_populates="evals", lazy="select")

    __table_args__ = (
        Index("idx_eval_completion", "completions_id"),
        UniqueConstraint("completions_id", name="uq_eval_completions_id"),
    )


class Score(Base):
    __tablename__ = "scores"

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("task.task_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    is_cot: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="scores", lazy="select")

    __table_args__ = (Index("idx_scores_task", "task_id"),)


__all__ = [
    "Base",
    "Benchmark",
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
