from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy import (
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column



class Base(DeclarativeBase):
    pass


class Benchmark(Base)：
    _tablename__ = "benchmarks"
    
    benchmark_id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_name:Mapped[str] = mapped_column(String(255), nullable=False)
    benchmark_split:Mapped[str] = mapped_column(String(225) , nullable=False)
    status:Mapped[str] = mapped_column(String(50), nullable=False)
    url:Mapped[Optional[str]] = mapped_column(String(2083), nullable=True)
    num_samples:Mapped[int] = mapped_columu(Integer, nullable=False)

    _table_args__ = (
        CheckConstraint(
            "status IN ('Todo', 'Buggy', 'Low', 'DataSynthesizing', 'Completed')",
            name="chk_benchmark_status",
        ),
    )


class Model(Base):
    _tablename__ = "models"

    model_id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name:Mapped[str] = mapped_column(String(255), nullable=False)
    data_version:Mapped[str] = mapped_column(String(225), nullable=False)
    arch_version:MaPPED[str] = mapped_column(String(225), nullable=False)
    num_params:Mapped[int] = mapped_column(Integer, nullable=False)

class Task(Base):
    _tablename__ = "tasks"

    task_id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_pathj:Mapped[str] = mapped_column(String(225), nullable=False)
    is_params:Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    evaluator:Mapped[str] = mapped_column(String(225), nullable=False)
    status:Mapped[str] = mapped_column(String(225), nullable=False)
    git_hash:Mapped[str] = mapped_column(String(225), nullable=False)
    model_id:Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model.model_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    benchmark_id:Mapped[int] = mapped_column(
        Integer,
        ForeignKey("benchmark.benchmark_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    decs:Mapped[str] = mapped_column(Text, nullable=True)
        sampling_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )
    log_path: Mapped[Optional[str]] = mapped_column(String(225), nullable=True)

    _table_args__ = (
        Index("idx_task_model_id", "model_id"),
        Index("idx_task_benchmark_id", "benchmark_id"),
    )

class Completion(Base):
    _tablename__ = "completions"
    completion_id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("task.task_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    context:Mapped[Dict[str, Any]] = mapped_column(JSONB, nullabl=False)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    repeat_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)
    status: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (
        Index("idx_completions_task", "task_id"),
        UniqueConstraint(
            "task_id", "sample_index", "repeat_index", name="uq_completions_sample"
        ),
    )

class Eval(Base):
    _tablename__ = "eval"

    eval_id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    completion_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("completions.completion_id", ondelete="RESTRICT", onupdate="RESTRICT"),
        nullable=False,
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    ref_answer: Mapped[str] = mapped_column(Text, nullable=False)
    is_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fail_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(6), nullable=False)

    __table_args__ = (Index("idx_eval_completion", "completions_id"),)

class Scores(Base):
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
