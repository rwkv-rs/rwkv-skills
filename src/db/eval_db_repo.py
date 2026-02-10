from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import Integer, case, cast, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .orm import Benchmark, Completion, Eval, Model, Score, Task
from src.eval.results.schema import IndexValidationError, strict_nonneg_int


class EvalDbRepository:
    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        return {column.name: getattr(model, column.name) for column in model.__table__.columns}

    def get_benchmark_id(
        self,
        session: Session,
        *,
        benchmark_name: str,
        benchmark_split: str,
    ) -> int | None:
        stmt = select(Benchmark.benchmark_id).where(
            Benchmark.benchmark_name == benchmark_name,
            Benchmark.benchmark_split == benchmark_split,
        )
        return session.execute(stmt).scalar_one_or_none()

    def get_benchmark_num_samples(
        self,
        session: Session,
        *,
        benchmark_id: int,
    ) -> int | None:
        stmt = select(Benchmark.num_samples).where(Benchmark.benchmark_id == benchmark_id)
        return session.execute(stmt).scalar_one_or_none()

    def update_benchmark_num_samples(
        self,
        session: Session,
        *,
        benchmark_id: int,
        num_samples: int,
    ) -> None:
        stmt = (
            update(Benchmark)
            .where(Benchmark.benchmark_id == benchmark_id)
            .values(num_samples=num_samples)
        )
        session.execute(stmt)

    def insert_benchmark(
        self,
        session: Session,
        *,
        benchmark_name: str,
        benchmark_split: str,
        url: str | None,
        status: str,
        num_samples: int,
    ) -> int:
        benchmark = Benchmark(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            url=url,
            status=status,
            num_samples=num_samples,
        )
        session.add(benchmark)
        session.flush()
        return int(benchmark.benchmark_id)

    def get_model_id(
        self,
        session: Session,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int | None:
        stmt = select(Model.model_id).where(
            Model.model_name == model_name,
            Model.arch_version == arch_version,
            Model.data_version == data_version,
            Model.num_params == num_params,
        )
        return session.execute(stmt).scalar_one_or_none()

    def insert_model(
        self,
        session: Session,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int:
        model = Model(
            model_name=model_name,
            arch_version=arch_version,
            data_version=data_version,
            num_params=num_params,
        )
        session.add(model)
        session.flush()
        return int(model.model_id)

    def insert_task(
        self,
        session: Session,
        *,
        config_path: str | None,
        evaluator: str,
        is_param_search: bool,
        created_at: datetime,
        status: str,
        git_hash: str,
        model_id: int,
        benchmark_id: int,
        desc: str | None,
        sampling_config: dict[str, Any] | None,
        log_path: str,
    ) -> int:
        task = Task(
            config_path=config_path,
            evaluator=evaluator,
            is_param_search=is_param_search,
            created_at=created_at,
            status=status,
            git_hash=git_hash,
            model_id=model_id,
            benchmark_id=benchmark_id,
            desc=desc,
            sampling_config=sampling_config,
            log_path=log_path,
        )
        session.add(task)
        session.flush()
        return int(task.task_id)

    def update_task_status(self, session: Session, *, task_id: int, status: str) -> None:
        stmt = update(Task).where(Task.task_id == task_id).values(status=status)
        session.execute(stmt)

    def get_latest_task_id(
        self,
        session: Session,
        *,
        benchmark_id: int,
        model_id: int,
        is_param_search: bool,
    ) -> int | None:
        stmt = (
            select(Task.task_id)
            .where(
                Task.benchmark_id == benchmark_id,
                Task.model_id == model_id,
                Task.is_param_search == is_param_search,
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )
        return session.execute(stmt).scalar_one_or_none()

    def task_has_score(self, session: Session, *, task_id: int) -> bool:
        stmt = select(Score.score_id).where(Score.task_id == task_id).limit(1)
        return session.execute(stmt).first() is not None

    @staticmethod
    def _dataset_label() -> Any:
        return case(
            (Benchmark.benchmark_split != "", func.concat(Benchmark.benchmark_name, "_", Benchmark.benchmark_split)),
            else_=Benchmark.benchmark_name,
        )

    def fetch_latest_scores(self, session: Session) -> list[dict[str, Any]]:
        row_number = func.row_number().over(
            partition_by=(Task.model_id, Task.benchmark_id, Score.is_cot),
            order_by=Score.created_at.desc(),
        ).label("rn")
        subquery = (
            select(
                Score.task_id.label("task_id"),
                Score.is_cot.label("cot"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Task.is_param_search.label("is_param_search"),
                Task.model_id.label("model_id"),
                Task.benchmark_id.label("benchmark_id"),
                row_number,
            )
            .join(Task, Task.task_id == Score.task_id)
            .subquery()
        )
        stmt = (
            select(
                subquery.c.task_id,
                subquery.c.cot,
                subquery.c.metrics,
                subquery.c.created_at,
                subquery.c.is_param_search,
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
                cast(None, Integer).label("samples"),
                cast(None, Integer).label("problems"),
            )
            .join(Task, Task.task_id == subquery.c.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(subquery.c.rn == 1, Task.is_param_search.is_(False))
        )
        return list(session.execute(stmt).mappings().all())

    def fetch_latest_scores_for_space(
        self,
        session: Session,
        *,
        include_param_search: bool,
    ) -> list[dict[str, Any]]:
        row_number = func.row_number().over(
            partition_by=(Task.model_id, Task.benchmark_id, Score.is_cot),
            order_by=(Score.created_at.desc(), Score.score_id.desc()),
        ).label("rn")
        subquery = (
            select(
                Score.task_id.label("task_id"),
                Score.is_cot.label("cot"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Task.is_param_search.label("is_param_search"),
                row_number,
            )
            .join(Task, Task.task_id == Score.task_id)
            .subquery()
        )
        stmt = (
            select(
                subquery.c.task_id,
                subquery.c.cot,
                subquery.c.metrics,
                subquery.c.created_at,
                subquery.c.is_param_search,
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
                Benchmark.num_samples.label("samples"),
                Benchmark.num_samples.label("problems"),
                Task.evaluator.label("task"),
                cast(None, Integer).label("task_details"),
                Task.log_path.label("log_path"),
            )
            .join(Task, Task.task_id == subquery.c.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(subquery.c.rn == 1)
        )
        if not include_param_search:
            stmt = stmt.where(Task.is_param_search.is_(False))
        return list(session.execute(stmt).mappings().all())

    def fetch_scores_by_benchmark(
        self,
        session: Session,
        *,
        benchmark_name: str,
        benchmark_split: str,
        model_name: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                Score.task_id.label("task_id"),
                Score.is_cot.label("cot"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Task.is_param_search.label("is_param_search"),
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
                cast(None, Integer).label("samples"),
                cast(None, Integer).label("problems"),
            )
            .join(Task, Task.task_id == Score.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(
                Benchmark.benchmark_name == benchmark_name,
                Benchmark.benchmark_split == benchmark_split,
                Model.model_name == model_name,
                Task.is_param_search == is_param_search,
            )
            .order_by(Score.created_at.desc())
        )
        return list(session.execute(stmt).mappings().all())

    def count_completions(
        self,
        session: Session,
        *,
        task_id: int,
        status: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(Completion).where(Completion.task_id == task_id)
        if status:
            stmt = stmt.where(Completion.status == status)
        return int(session.execute(stmt).scalar_one())

    def fetch_completions(
        self,
        session: Session,
        *,
        task_id: int,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                Benchmark.benchmark_name.label("benchmark_name"),
                Benchmark.benchmark_split.label("benchmark_split"),
                Completion.sample_index,
                Completion.repeat_index,
                Completion.context,
            )
            .join(Task, Task.task_id == Completion.task_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(Completion.task_id == task_id)
            .order_by(Completion.sample_index.asc(), Completion.repeat_index.asc())
        )
        if status:
            stmt = stmt.where(Completion.status == status)
        return list(session.execute(stmt).mappings().all())

    def fetch_completion_keys(
        self,
        session: Session,
        *,
        task_id: int,
        status: str | None = None,
    ) -> list[tuple[int, int]]:
        stmt = select(Completion.sample_index, Completion.repeat_index).where(Completion.task_id == task_id)
        if status:
            stmt = stmt.where(Completion.status == status)
        rows = session.execute(stmt).all()
        return [(int(row[0]), int(row[1])) for row in rows]

    def fetch_completion_id_map(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> dict[tuple[int, int], int]:
        stmt = select(
            Completion.completions_id,
            Completion.sample_index,
            Completion.repeat_index,
        ).where(Completion.task_id == task_id)
        rows = session.execute(stmt).all()
        mapping: dict[tuple[int, int], int] = {}
        for completions_id, sample_index, repeat_index in rows:
            mapping[(int(sample_index), int(repeat_index))] = int(completions_id)
        return mapping

    def fetch_existing_eval_completion_ids(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> set[int]:
        stmt = (
            select(Eval.completions_id)
            .join(Completion, Completion.completions_id == Eval.completions_id)
            .where(Completion.task_id == task_id)
        )
        rows = session.execute(stmt).all()
        return {int(row[0]) for row in rows}

    def fetch_score_by_task(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> dict[str, Any] | None:
        stmt = (
            select(
                Score.task_id.label("task_id"),
                Score.is_cot.label("cot"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
            )
            .join(Task, Task.task_id == Score.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(Score.task_id == task_id)
            .order_by(Score.created_at.desc())
            .limit(1)
        )
        row = session.execute(stmt).mappings().first()
        return dict(row) if row else None

    def fetch_task(self, session: Session, *, task_id: int) -> dict[str, Any] | None:
        stmt = select(Task).where(Task.task_id == task_id)
        row = session.execute(stmt).scalar_one_or_none()
        return self._model_to_dict(row) if row else None

    def fetch_latest_task_by_names(
        self,
        session: Session,
        *,
        benchmark_name: str,
        benchmark_split: str,
        model_name: str,
        is_param_search: bool,
    ) -> dict[str, Any] | None:
        stmt = (
            select(Task)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .join(Model, Model.model_id == Task.model_id)
            .where(
                Benchmark.benchmark_name == benchmark_name,
                Benchmark.benchmark_split == benchmark_split,
                Model.model_name == model_name,
                Task.is_param_search == is_param_search,
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )
        row = session.execute(stmt).scalar_one_or_none()
        return self._model_to_dict(row) if row else None

    def fetch_model(self, session: Session, *, model_id: int) -> dict[str, Any] | None:
        stmt = select(Model).where(Model.model_id == model_id)
        row = session.execute(stmt).scalar_one_or_none()
        return self._model_to_dict(row) if row else None

    def fetch_benchmark(self, session: Session, *, benchmark_id: int) -> dict[str, Any] | None:
        stmt = select(Benchmark).where(Benchmark.benchmark_id == benchmark_id)
        row = session.execute(stmt).scalar_one_or_none()
        return self._model_to_dict(row) if row else None

    def fetch_completions_rows(self, session: Session, *, task_id: int) -> list[dict[str, Any]]:
        stmt = select(Completion).where(Completion.task_id == task_id).order_by(Completion.completions_id.asc())
        rows = session.execute(stmt).scalars().all()
        return [self._model_to_dict(row) for row in rows]

    def fetch_eval_rows(self, session: Session, *, task_id: int) -> list[dict[str, Any]]:
        stmt = (
            select(Eval)
            .join(Completion, Completion.completions_id == Eval.completions_id)
            .where(Completion.task_id == task_id)
            .order_by(Eval.eval_id.asc())
        )
        rows = session.execute(stmt).scalars().all()
        return [self._model_to_dict(row) for row in rows]

    def fetch_scores_rows(self, session: Session, *, task_id: int) -> list[dict[str, Any]]:
        stmt = select(Score).where(Score.task_id == task_id).order_by(Score.created_at.desc())
        rows = session.execute(stmt).scalars().all()
        return [self._model_to_dict(row) for row in rows]

    def insert_completion(
        self,
        session: Session,
        *,
        task_id: int,
        payload: dict[str, Any],
        context: dict[str, Any] | None,
        created_at: datetime,
        status: str,
    ) -> None:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        stmt = pg_insert(Completion).values(
            task_id=task_id,
            context=context or {},
            sample_index=sample_index,
            repeat_index=repeat_index,
            created_at=created_at,
            status=status,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Completion.task_id, Completion.sample_index, Completion.repeat_index],
            set_={
                "context": stmt.excluded.context,
                "created_at": stmt.excluded.created_at,
                "status": stmt.excluded.status,
            },
        )
        session.execute(stmt)

    def insert_eval(
        self,
        session: Session,
        *,
        completions_id: int,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        eval_row = Eval(
            completions_id=completions_id,
            answer=payload.get("answer"),
            ref_answer=payload.get("ref_answer"),
            is_passed=bool(payload.get("is_passed", False)),
            fail_reason=payload.get("fail_reason"),
            created_at=created_at,
        )
        session.add(eval_row)

    def insert_score(
        self,
        session: Session,
        *,
        task_id: int,
        payload: dict[str, Any],
    ) -> None:
        created_at = payload.get("created_at")
        if not created_at:
            created_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)
        score = Score(
            task_id=task_id,
            is_cot=bool(payload.get("cot", False)),
            metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
            created_at=created_at,
        )
        session.add(score)
