from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional
from contextlib import contextmanager

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.eval.scheduler.config import DBConfig

_ENGINE = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _build_db_url(config: DBConfig) -> str:
    return (
        f"postgresql+psycopg://{config.user}:{config.password}"
        f"@{config.host}:{config.port}/{config.dbname}"
    )


def init_orm(config: DBConfig) -> None:
    global _ENGINE, _SESSION_FACTORY
    if not config.enabled:
        return
    if _ENGINE is not None and _SESSION_FACTORY is not None:
        return
    _ENGINE = create_engine(_build_db_url(config), pool_pre_ping=True, future=True)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, class_=Session)


@contextmanager
def get_session() -> Iterator[Session]:
    if _SESSION_FACTORY is None:
        raise RuntimeError("ORM session factory not initialized")
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

    __table_args__ = (
        CheckConstraint(
            "status IN ('Todo', 'Buggy', 'Low', 'DataSynthesizing', 'Completed')",
            name="chk_benchmark_status",
        ),
    )


class Model(Base):
    __tablename__ = "model"

    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_version: Mapped[str] = mapped_column(String(255), nullable=False)
    arch_version: Mapped[str] = mapped_column(String(255), nullable=False)
    num_params: Mapped[str] = mapped_column(String(255), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)


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
    fail_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    __table_args__ = (Index("idx_eval_completion", "completions_id"),)


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
]
