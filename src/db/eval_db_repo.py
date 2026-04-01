from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import Integer, Text, case, cast, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .orm import Benchmark, Checker, Completion, Eval, Model, Score, Task
from src.eval.results.schema import IndexValidationError, strict_nonneg_int


def _canonical_task_status(status: str) -> str:
    raw = str(status or "").strip().lower()
    mapping = {
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
    }
    return mapping.get(raw, status)


def _canonical_completion_status(status: str | None) -> str:
    raw = str(status or "").strip().lower()
    mapping = {
        "completed": "Completed",
        "running": "Running",
        "failed": "Failed",
    }
    if raw in mapping:
        return mapping[raw]
    raise ValueError(f"unsupported completion status: {status!r}")


def _parse_cot_mode(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    mapping = {
        "nocot": "NoCoT",
        "no_cot": "NoCoT",
        "no-cot": "NoCoT",
        "fakecot": "FakeCoT",
        "fake_cot": "FakeCoT",
        "fake-cot": "FakeCoT",
        "cot": "CoT",
    }
    return mapping.get(raw)


def _canonical_score_cot_mode(payload: dict[str, Any]) -> str:
    task_details = payload.get("task_details")
    sampling_config = payload.get("sampling_config")
    for candidate in (
        payload.get("cot_mode"),
        task_details.get("cot_mode") if isinstance(task_details, dict) else None,
        sampling_config.get("cot_mode") if isinstance(sampling_config, dict) else None,
    ):
        parsed = _parse_cot_mode(candidate)
        if parsed is not None:
            return parsed
    return "CoT" if bool(payload.get("cot", False)) else "NoCoT"


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
        is_tmp: bool,
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
            is_tmp=is_tmp,
            created_at=created_at,
            status=_canonical_task_status(status),
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
        stmt = update(Task).where(Task.task_id == task_id).values(status=_canonical_task_status(status))
        session.execute(stmt)

    def find_tasks_by_identity(
        self,
        session: Session,
        *,
        config_path: str | None,
        evaluator: str,
        git_hash: str,
        model_id: int,
        benchmark_id: int,
        sampling_config: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                Task.task_id.label("task_id"),
                Task.status.label("status"),
            )
            .where(
                Task.evaluator == evaluator,
                Task.git_hash == git_hash,
                Task.model_id == model_id,
                Task.benchmark_id == benchmark_id,
                Task.is_tmp.is_(False),
            )
            .order_by(Task.task_id.asc())
        )
        if config_path is None:
            stmt = stmt.where(Task.config_path.is_(None))
        else:
            stmt = stmt.where(Task.config_path == config_path)
        if sampling_config is None:
            stmt = stmt.where(Task.sampling_config.is_(None))
        else:
            stmt = stmt.where(Task.sampling_config == sampling_config)
        return list(session.execute(stmt).mappings().all())

    def get_latest_task_id(
        self,
        session: Session,
        *,
        benchmark_id: int,
        model_id: int,
        is_param_search: bool,
        evaluator: str | None = None,
    ) -> int | None:
        stmt = (
            select(Task.task_id)
            .where(
                Task.benchmark_id == benchmark_id,
                Task.model_id == model_id,
                Task.is_param_search == is_param_search,
                Task.is_tmp.is_(False),
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )
        if evaluator:
            stmt = stmt.where(Task.evaluator == evaluator)
        return session.execute(stmt).scalar_one_or_none()

    def task_has_score(self, session: Session, *, task_id: int) -> bool:
        stmt = select(Score.score_id).where(Score.task_id == task_id).limit(1)
        return session.execute(stmt).first() is not None

    def delete_scores_by_task_id(self, session: Session, *, task_id: int) -> None:
        session.execute(delete(Score).where(Score.task_id == task_id))

    @staticmethod
    def _dataset_label() -> Any:
        return case(
            (Benchmark.benchmark_split != "", func.concat(Benchmark.benchmark_name, "_", Benchmark.benchmark_split)),
            else_=Benchmark.benchmark_name,
        )

    @staticmethod
    def _score_cot_bool() -> Any:
        return case((Score.cot_mode == "NoCoT", False), else_=True)

    def fetch_latest_scores(self, session: Session) -> list[dict[str, Any]]:
        row_number = func.row_number().over(
            partition_by=(Task.model_id, Task.benchmark_id, Task.evaluator, Task.sampling_config),
            order_by=Score.created_at.desc(),
        ).label("rn")
        subquery = (
            select(
                Score.task_id.label("task_id"),
                self._score_cot_bool().label("cot"),
                Score.cot_mode.label("cot_mode"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Task.is_param_search.label("is_param_search"),
                Task.model_id.label("model_id"),
                Task.benchmark_id.label("benchmark_id"),
                Task.evaluator.label("task"),
                Task.sampling_config.label("sampling_config"),
                row_number,
            )
            .join(Task, Task.task_id == Score.task_id)
            .subquery()
        )
        stmt = (
            select(
                subquery.c.task_id,
                subquery.c.cot,
                subquery.c.cot_mode,
                subquery.c.metrics,
                subquery.c.created_at,
                subquery.c.is_param_search,
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
                cast(None, Integer).label("samples"),
                cast(None, Integer).label("problems"),
                subquery.c.task,
                subquery.c.sampling_config,
            )
            .join(Task, Task.task_id == subquery.c.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(subquery.c.rn == 1, Task.is_param_search.is_(False), Task.is_tmp.is_(False))
        )
        return list(session.execute(stmt).mappings().all())

    def fetch_latest_scores_for_space(
        self,
        session: Session,
        *,
        include_param_search: bool,
    ) -> list[dict[str, Any]]:
        row_number = func.row_number().over(
            partition_by=(Task.model_id, Task.benchmark_id, Task.evaluator, Task.sampling_config),
            order_by=(Score.created_at.desc(), Score.score_id.desc()),
        ).label("rn")
        subquery = (
            select(
                Score.task_id.label("task_id"),
                self._score_cot_bool().label("cot"),
                Score.cot_mode.label("cot_mode"),
                Score.metrics.label("metrics"),
                Score.created_at.label("created_at"),
                Task.is_param_search.label("is_param_search"),
                Task.evaluator.label("task"),
                Task.sampling_config.label("sampling_config"),
                row_number,
            )
            .join(Task, Task.task_id == Score.task_id)
            .subquery()
        )
        stmt = (
            select(
                subquery.c.task_id,
                subquery.c.cot,
                subquery.c.cot_mode,
                subquery.c.metrics,
                subquery.c.created_at,
                subquery.c.is_param_search,
                Model.model_name.label("model"),
                self._dataset_label().label("dataset"),
                Benchmark.num_samples.label("samples"),
                Benchmark.num_samples.label("problems"),
                subquery.c.task,
                cast(None, Integer).label("task_details"),
                subquery.c.sampling_config,
                Task.log_path.label("log_path"),
            )
            .join(Task, Task.task_id == subquery.c.task_id)
            .join(Model, Model.model_id == Task.model_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(subquery.c.rn == 1, Task.is_tmp.is_(False))
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
                self._score_cot_bool().label("cot"),
                Score.cot_mode.label("cot_mode"),
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
                Task.is_tmp.is_(False),
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
            stmt = stmt.where(Completion.status == _canonical_completion_status(status))
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
                Completion.avg_repeat_index.label("repeat_index"),
                Completion.pass_index.label("pass_index"),
                Completion.context,
            )
            .join(Task, Task.task_id == Completion.task_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(Completion.task_id == task_id)
            .order_by(Completion.sample_index.asc(), Completion.avg_repeat_index.asc(), Completion.pass_index.asc())
        )
        if status:
            stmt = stmt.where(Completion.status == _canonical_completion_status(status))
        return list(session.execute(stmt).mappings().all())

    def fetch_completion_keys(
        self,
        session: Session,
        *,
        task_id: int,
        status: str | None = None,
    ) -> list[tuple[int, int, int]]:
        stmt = select(
            Completion.sample_index,
            Completion.avg_repeat_index,
            Completion.pass_index,
        ).where(Completion.task_id == task_id)
        if status:
            stmt = stmt.where(Completion.status == _canonical_completion_status(status))
        rows = session.execute(stmt).all()
        return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]

    def fetch_completion_id_map(
        self,
        session: Session,
        *,
        task_id: int,
        status: str | None = None,
    ) -> dict[tuple[int, int, int], int]:
        stmt = select(
            Completion.completions_id,
            Completion.sample_index,
            Completion.avg_repeat_index,
            Completion.pass_index,
        ).where(Completion.task_id == task_id)
        if status:
            stmt = stmt.where(Completion.status == _canonical_completion_status(status))
        rows = session.execute(stmt).all()
        mapping: dict[tuple[int, int, int], int] = {}
        for completions_id, sample_index, repeat_index, pass_index in rows:
            mapping[(int(sample_index), int(repeat_index), int(pass_index))] = int(completions_id)
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
                self._score_cot_bool().label("cot"),
                Score.cot_mode.label("cot_mode"),
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
        evaluator: str | None = None,
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
                Task.is_tmp.is_(False),
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )
        if evaluator:
            stmt = stmt.where(Task.evaluator == evaluator)
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

    def fetch_eval_with_completions_by_task(
        self,
        session: Session,
        *,
        task_id: int,
        only_wrong: bool,
        limit: int | None = None,
        offset: int | None = None,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        columns: list[Any] = [
            Completion.sample_index.label("sample_index"),
            Completion.avg_repeat_index.label("repeat_index"),
            Completion.pass_index.label("pass_index"),
            Eval.is_passed.label("is_passed"),
            Eval.answer.label("answer"),
            Eval.ref_answer.label("ref_answer"),
            Eval.fail_reason.label("fail_reason"),
            func.left(cast(Completion.context, Text), 80).label("context_preview"),
        ]
        if include_context:
            columns.append(Completion.context.label("context"))

        stmt = (
            select(*columns)
            .join(Eval, Eval.completions_id == Completion.completions_id)
            .where(Completion.task_id == task_id)
            .order_by(
                Completion.sample_index.asc(),
                Completion.avg_repeat_index.asc(),
                Completion.pass_index.asc(),
                Eval.eval_id.asc(),
            )
        )
        if only_wrong:
            stmt = stmt.where(Eval.is_passed.is_(False))
        if offset and offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).mappings().all())

    def fetch_eval_context_by_task_attempt(
        self,
        session: Session,
        *,
        task_id: int,
        sample_index: int,
        repeat_index: int,
        pass_index: int,
    ) -> Any | None:
        stmt = (
            select(Completion.context)
            .join(Eval, Eval.completions_id == Completion.completions_id)
            .where(
                Completion.task_id == task_id,
                Completion.sample_index == sample_index,
                Completion.avg_repeat_index == repeat_index,
                Completion.pass_index == pass_index,
            )
            .order_by(Eval.eval_id.desc())
            .limit(1)
        )
        return session.execute(stmt).scalar_one_or_none()

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
        pass_index = strict_nonneg_int(payload.get("pass_index", 0), "pass_index")
        stmt = pg_insert(Completion).values(
            task_id=task_id,
            context=context or {},
            sample_index=sample_index,
            avg_repeat_index=repeat_index,
            pass_index=pass_index,
            created_at=created_at,
            status=_canonical_completion_status(status),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Completion.task_id, Completion.sample_index, Completion.avg_repeat_index, Completion.pass_index],
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
            cot_mode=_canonical_score_cot_mode(payload),
            metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
            created_at=created_at,
        )
        session.add(score)

    def fetch_existing_checker_completion_ids(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> set[int]:
        stmt = (
            select(Checker.completions_id)
            .join(Completion, Completion.completions_id == Checker.completions_id)
            .where(Completion.task_id == task_id)
        )
        rows = session.execute(stmt).all()
        return {int(row[0]) for row in rows}

    def insert_checker(
        self,
        session: Session,
        *,
        completions_id: int,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        checker = Checker(
            completions_id=completions_id,
            answer_correct=bool(payload.get("answer_correct", False)),
            instruction_following_error=bool(payload.get("instruction_following_error", False)),
            world_knowledge_error=bool(payload.get("world_knowledge_error", False)),
            math_error=bool(payload.get("math_error", False)),
            reasoning_logic_error=bool(payload.get("reasoning_logic_error", False)),
            thought_contains_correct_answer=bool(payload.get("thought_contains_correct_answer", False)),
            needs_human_review=bool(payload.get("needs_human_review", False)),
            reason=str(payload.get("reason") or ""),
            created_at=created_at,
        )
        session.add(checker)

    def fetch_checker_rows(self, session: Session, *, task_id: int) -> list[dict[str, Any]]:
        stmt = (
            select(Checker)
            .join(Completion, Completion.completions_id == Checker.completions_id)
            .where(Completion.task_id == task_id)
            .order_by(Checker.checker_id.asc())
        )
        rows = session.execute(stmt).scalars().all()
        return [self._model_to_dict(row) for row in rows]

    def fetch_checker_keys(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> set[tuple[int, int, int]]:
        stmt = (
            select(
                Completion.sample_index,
                Completion.avg_repeat_index,
                Completion.pass_index,
            )
            .join(Checker, Checker.completions_id == Completion.completions_id)
            .where(Completion.task_id == task_id)
        )
        rows = session.execute(stmt).all()
        return {(int(row[0]), int(row[1]), int(row[2])) for row in rows}
