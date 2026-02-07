from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Iterable

from sqlalchemy import Integer, case, cast, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from sqlalchemy.orm import Session

from .orm import Benchmark, Completion, Eval, Model, Score, Task


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
        stmt = pg_insert(Benchmark).values(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            url=url,
            status=status,
            num_samples=num_samples,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Benchmark.benchmark_name, Benchmark.benchmark_split],
            set_={
                "url": stmt.excluded.url,
                "status": stmt.excluded.status,
                "num_samples": stmt.excluded.num_samples,
            },
        ).returning(Benchmark.benchmark_id)
        result = session.execute(stmt)
        return int(result.scalar_one())

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
        stmt = pg_insert(Model).values(
            model_name=model_name,
            arch_version=arch_version,
            data_version=data_version,
            num_params=num_params,
        )
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[Model.model_name, Model.arch_version, Model.data_version, Model.num_params],
        ).returning(Model.model_id)
        result = session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            return int(row)
        # Conflict occurred, fetch existing model_id
        return int(self.get_model_id(
            session,
            model_name=model_name,
            arch_version=arch_version,
            data_version=data_version,
            num_params=num_params,
        ))

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
        evaluator: str | None = None,
    ) -> int | None:
        stmt = (
            select(Task.task_id)
            .where(
                Task.benchmark_id == benchmark_id,
                Task.model_id == model_id,
                Task.is_param_search == is_param_search,
            )
        )
        if evaluator is not None:
            stmt = stmt.where(Task.evaluator == evaluator)
        # 按 created_at 降序，同秒时按 task_id 降序保证稳定性
        stmt = stmt.order_by(Task.created_at.desc(), Task.task_id.desc()).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    def task_has_score(self, session: Session, *, task_id: int, is_cot: bool | None = None) -> bool:
        stmt = select(Score.score_id).where(Score.task_id == task_id)
        if is_cot is not None:
            stmt = stmt.where(Score.is_cot == is_cot)
        stmt = stmt.limit(1)
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
    ) -> int:
        stmt = select(func.count()).select_from(Completion).where(Completion.task_id == task_id)
        return int(session.execute(stmt).scalar_one())

    def fetch_completions(
        self,
        session: Session,
        *,
        task_id: int,
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
        return list(session.execute(stmt).mappings().all())

    def iter_completions(
        self,
        session: Session,
        *,
        task_id: int,
        batch_size: int = 1000,
    ) -> Iterable[dict[str, Any]]:
        """流式迭代 completions，避免一次性加载全部到内存。"""
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
        result = session.execute(stmt).mappings()
        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                yield dict(row)

    def fetch_completion_keys(
        self,
        session: Session,
        *,
        task_id: int,
    ) -> list[tuple[int, int]]:
        stmt = select(Completion.sample_index, Completion.repeat_index).where(Completion.task_id == task_id)
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
        sample_index = payload.get("sample_index")
        repeat_index = payload.get("repeat_index")
        if sample_index is None:
            raise ValueError("sample_index is required but got None")
        if repeat_index is None:
            raise ValueError("repeat_index is required but got None")
        try:
            sample_index = int(sample_index)
        except (TypeError, ValueError) as e:
            raise ValueError(f"sample_index must be an integer, got: {sample_index}") from e
        try:
            repeat_index = int(repeat_index)
        except (TypeError, ValueError) as e:
            raise ValueError(f"repeat_index must be an integer, got: {repeat_index}") from e
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
        """插入或更新 eval 记录（upsert on completions_id）。

        fail_reason 现在是 JSONB 类型，结构为：
        {
            "type": "exact_match" | "code_execution" | ...,
            "raw": "原始错误信息",
            "llm_checker": {...}  # checker 结果会 merge 进来
        }

        注意：upsert 时会保留已有的 llm_checker 等 checker 结果。
        """
        # 构建 fail_reason JSONB
        raw_fail_reason = payload.get("fail_reason")
        if isinstance(raw_fail_reason, dict):
            fail_reason_jsonb = raw_fail_reason
        elif raw_fail_reason:
            fail_reason_jsonb = {"type": "raw", "raw": str(raw_fail_reason)}
        else:
            fail_reason_jsonb = {}

        stmt = pg_insert(Eval).values(
            completions_id=completions_id,
            answer=payload.get("answer") or "",
            ref_answer=payload.get("ref_answer") or "",
            is_passed=bool(payload.get("is_passed", False)),
            fail_reason=fail_reason_jsonb,
            created_at=created_at,
        )
        # 当 is_passed=True 时，清除旧的 fail_reason（使用新值覆盖）
        # 当 is_passed=False 时，合并旧值和新值（保留如 llm_checker 等历史结果）
        stmt = stmt.on_conflict_do_update(
            constraint="uq_eval_completions_id",
            set_={
                "answer": stmt.excluded.answer,
                "ref_answer": stmt.excluded.ref_answer,
                "is_passed": stmt.excluded.is_passed,
                "fail_reason": case(
                    (stmt.excluded.is_passed == True, stmt.excluded.fail_reason),
                    else_=Eval.fail_reason.concat(stmt.excluded.fail_reason),
                ),
                "created_at": stmt.excluded.created_at,
            },
        )
        session.execute(stmt)

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

    def iter_failed_evals_for_checker(
        self,
        session: Session,
        *,
        task_id: int,
        checker_type: str = "llm_checker",
        batch_size: int = 100,
    ) -> Iterable[dict[str, Any]]:
        """流式迭代需要 checker 处理的 eval 记录。

        返回 is_passed=False 且 fail_reason 中没有该 checker_type 结果的记录。
        """
        from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

        stmt = (
            select(
                Eval.eval_id,
                Eval.answer,
                Eval.ref_answer,
                Eval.fail_reason,
                Completion.sample_index,
                Completion.repeat_index,
                Completion.context,
                Benchmark.benchmark_name,
                Benchmark.benchmark_split,
            )
            .join(Completion, Completion.completions_id == Eval.completions_id)
            .join(Task, Task.task_id == Completion.task_id)
            .join(Benchmark, Benchmark.benchmark_id == Task.benchmark_id)
            .where(
                Completion.task_id == task_id,
                Eval.is_passed == False,
                ~Eval.fail_reason.has_key(checker_type),
            )
            .order_by(Completion.sample_index.asc(), Completion.repeat_index.asc())
        )
        result = session.execute(stmt).mappings()
        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                yield dict(row)

    def update_eval_fail_reason(
        self,
        session: Session,
        *,
        eval_id: int,
        checker_type: str,
        checker_result: dict[str, Any],
    ) -> None:
        """将 checker 结果 merge 到 eval.fail_reason 中。

        使用 jsonb_set 保留原有字段，只更新/添加 checker_type 键。
        """
        import json

        stmt = (
            update(Eval)
            .where(Eval.eval_id == eval_id)
            .values(
                fail_reason=func.jsonb_set(
                    Eval.fail_reason,
                    "{" + checker_type + "}",
                    func.cast(json.dumps(checker_result), JSONB),
                    True,  # create_if_missing
                )
            )
        )
        session.execute(stmt)

    def bulk_update_eval_fail_reason(
        self,
        session: Session,
        *,
        updates: list[tuple[int, str, dict[str, Any]]],  # [(eval_id, checker_type, result), ...]
    ) -> int:
        """批量更新 eval.fail_reason。

        Returns:
            更新的记录数
        """
        if not updates:
            return 0

        import json

        count = 0
        for eval_id, checker_type, checker_result in updates:
            stmt = (
                update(Eval)
                .where(Eval.eval_id == eval_id)
                .values(
                    fail_reason=func.jsonb_set(
                        Eval.fail_reason,
                        "{" + checker_type + "}",
                        func.cast(json.dumps(checker_result), JSONB),
                        True,
                    )
                )
            )
            session.execute(stmt)
            count += 1
        return count
