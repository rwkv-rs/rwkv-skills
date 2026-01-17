from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.eval.results.schema import dataset_slug_parts
from src.eval.scheduler.dataset_utils import canonical_slug, make_dataset_slug, safe_slug
from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.infra.database import DatabaseManager

from .eval_db_repo import EvalDbRepository, StageOutputRow


@dataclass(slots=True)
class RunContext:
    run_id: str
    benchmark_name: str
    dataset: str
    dataset_split: str
    model_name: str
    model_slug: str
    run_tag: str | None
    cot: bool


class EvalDbService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._repo = EvalDbRepository()

    def prepare_run(
        self,
        *,
        dataset_slug: str,
        split_name: str | None,
        model_path: str,
        is_cot: bool,
        run_tag: str | None,
        sampling_config: dict[str, Any] | None,
        runtime_config: dict[str, Any] | None,
        code_version: str | None,
        task_tag: str | None = None,
        overwrite_run: bool = False,
    ) -> RunContext:
        dataset_slug = canonical_slug(dataset_slug)
        if split_name:
            benchmark_name = dataset_slug
            dataset_split = split_name
        else:
            benchmark_name, dataset_split = dataset_slug_parts(dataset_slug)
        dataset = make_dataset_slug(benchmark_name, dataset_split) if dataset_split else dataset_slug

        model_name = Path(model_path).stem
        model_slug = safe_slug(model_name)
        run_tag = run_tag or safe_slug(f"{dataset}__{model_slug}")

        job_name = detect_job_from_dataset(dataset, is_cot=is_cot)
        domain = JOB_CATALOGUE[job_name].domain if job_name and job_name in JOB_CATALOGUE else None
        task_details = {"job": job_name, "domain": domain} if job_name else None
        task = task_tag or job_name

        with self._db.get_connection() as conn:
            if overwrite_run:
                self._repo.delete_run_by_tag(
                    conn,
                    benchmark_name=benchmark_name,
                    dataset=dataset,
                    dataset_split=dataset_split,
                    model_name=model_name,
                    cot=is_cot,
                    run_tag=run_tag,
                )
            run_id = self._repo.get_run_by_tag(
                conn,
                benchmark_name=benchmark_name,
                dataset=dataset,
                dataset_split=dataset_split,
                model_name=model_name,
                cot=is_cot,
                run_tag=run_tag,
            )
            if not run_id:
                run_id = self._repo.insert_run(
                    conn,
                    benchmark_name=benchmark_name,
                    dataset=dataset,
                    dataset_split=dataset_split,
                    model_name=model_name,
                    model_slug=model_slug,
                    model_revision=None,
                    model_path=model_path,
                    cot=is_cot,
                    run_tag=run_tag,
                    sampling_config=sampling_config,
                    runtime_config=runtime_config,
                    code_version=code_version,
                    task=task,
                    task_details=task_details,
                    status="running",
                )
            else:
                self._repo.update_run_status(
                    conn,
                    run_id=run_id,
                    status="running",
                    start_now=True,
                    finished=False,
                )
            return RunContext(
                run_id=run_id,
                benchmark_name=benchmark_name,
                dataset=dataset,
                dataset_split=dataset_split,
                model_name=model_name,
                model_slug=model_slug,
                run_tag=run_tag,
                cot=is_cot,
            )

    def mark_run_status(self, *, run_id: str, status: str, error_msg: str | None = None) -> None:
        with self._db.get_connection() as conn:
            self._repo.update_run_status(
                conn,
                run_id=run_id,
                status=status,
                error_msg=error_msg,
                finished=True,
                start_now=status == "running",
            )

    def upsert_sample(
        self,
        *,
        benchmark_name: str,
        dataset_split: str,
        sample_index: int,
        question: str | None,
        reference_answer: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        with self._db.get_connection() as conn:
            return self._repo.upsert_sample(
                conn,
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                sample_index=sample_index,
                question=question,
                reference_answer=reference_answer,
                meta=meta,
            )

    def upsert_run_sample(
        self,
        *,
        run_id: str,
        sample_id: str,
        repeat_index: int,
        status: str = "pending",
        current_stage: str | None = None,
    ) -> str:
        with self._db.get_connection() as conn:
            return self._repo.upsert_run_sample(
                conn,
                run_id=run_id,
                sample_id=sample_id,
                repeat_index=repeat_index,
                status=status,
                current_stage=current_stage,
            )

    def fetch_latest_stage(
        self,
        *,
        run_sample_id: str,
        stage: str,
    ) -> StageOutputRow | None:
        with self._db.get_connection() as conn:
            return self._repo.fetch_latest_final_stage(conn, run_sample_id=run_sample_id, stage=stage)

    def start_attempt(
        self,
        *,
        run_sample_id: str,
        status: str = "running",
        worker_id: str | None = None,
        shard_id: int | None = None,
        shard_count: int | None = None,
        seed: int | None = None,
        current_stage: str | None = None,
    ) -> tuple[str, int]:
        with self._db.get_connection() as conn:
            next_idx = self._repo.get_latest_attempt_index(conn, run_sample_id=run_sample_id) + 1
            attempt_id = self._repo.insert_attempt(
                conn,
                run_sample_id=run_sample_id,
                attempt_index=next_idx,
                worker_id=worker_id,
                shard_id=shard_id,
                shard_count=shard_count,
                seed=seed,
                status=status,
            )
            self._repo.update_run_sample_status(
                conn,
                run_sample_id=run_sample_id,
                status="running",
                current_stage=current_stage,
                latest_attempt_index=next_idx,
                start_now=True,
            )
            return attempt_id, next_idx

    def mark_attempt_status(
        self,
        *,
        attempt_id: str,
        status: str,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.update_attempt_status(
                conn,
                attempt_id=attempt_id,
                status=status,
                error_msg=error_msg,
                finished=finished,
                start_now=status == "running",
            )

    def mark_run_sample_status(
        self,
        *,
        run_sample_id: str,
        status: str,
        current_stage: str | None,
        latest_attempt_index: int | None = None,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.update_run_sample_status(
                conn,
                run_sample_id=run_sample_id,
                status=status,
                current_stage=current_stage,
                latest_attempt_index=latest_attempt_index,
                error_msg=error_msg,
                finished=finished,
                start_now=status == "running",
            )

    def write_stage_output(
        self,
        *,
        attempt_id: str,
        stage: str,
        seq: int,
        prompt: str | None,
        completion: str | None,
        finish_reason: str | None,
        is_partial: bool = False,
        is_final: bool = True,
        provider_request_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
        token_count_prompt: int | None = None,
        token_count_response: int | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.insert_stage_output(
                conn,
                attempt_id=attempt_id,
                stage=stage,
                seq=seq,
                prompt=prompt,
                completion=completion,
                finish_reason=finish_reason,
                provider_request_id=provider_request_id,
                raw_response=raw_response,
                token_count_prompt=token_count_prompt,
                token_count_response=token_count_response,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                is_partial=is_partial,
                is_final=is_final,
            )
            if stage == "cot" and is_partial:
                self._repo.insert_cot_checkpoint(
                    conn,
                    attempt_id=attempt_id,
                    stage=stage,
                    token_offset=None,
                    partial_completion=completion,
                    kv_cache_ref=None,
                    rng_state=None,
                    status="partial",
                )

    def ingest_eval_results(
        self,
        *,
        eval_path: str | Path,
        run_id: str,
        metric_name: str = "passed",
    ) -> int:
        path = Path(eval_path)
        if not path.exists():
            return 0
        inserted = 0
        with path.open("r", encoding="utf-8") as fh, self._db.get_connection() as conn:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                benchmark_name = str(payload.get("benchmark_name", ""))
                dataset_split = str(payload.get("dataset_split", ""))
                sample_index = int(payload.get("sample_index", 0))
                repeat_index = int(payload.get("repeat_index", 0))
                answer = payload.get("answer")
                ref_answer = payload.get("ref_answer")
                is_passed = bool(payload.get("is_passed", False))
                fail_reason = payload.get("fail_reason")
                context = payload.get("context")
                meta = {"context": context} if context else None
                sample_id = self._repo.upsert_sample(
                    conn,
                    benchmark_name=benchmark_name or "unknown",
                    dataset_split=dataset_split or "unknown",
                    sample_index=sample_index,
                    question=None if not context else str(context),
                    reference_answer=None if ref_answer is None else str(ref_answer),
                    meta=meta,
                )
                self._repo.upsert_run_sample(
                    conn,
                    run_id=run_id,
                    sample_id=sample_id,
                    repeat_index=repeat_index,
                    status="succeeded",
                    current_stage="final",
                )
                self._repo.update_run_sample_result(
                    conn,
                    run_id=run_id,
                    sample_index=sample_index,
                    repeat_index=repeat_index,
                    answer=None if answer is None else str(answer),
                    ref_answer=None if ref_answer is None else str(ref_answer),
                    is_passed=is_passed,
                    fail_reason=None if fail_reason is None else str(fail_reason),
                )
                inserted += 1
        return inserted

    def record_score_summary(
        self,
        *,
        run_id: str,
        event_type: str,
        metrics: dict[str, Any],
        samples: int,
        problems: int | None,
        log_path: str | Path,
        eval_path: str | Path,
        score_path: str | Path,
        task: str | None,
        task_details: dict[str, Any] | None,
    ) -> None:
        details = task_details.copy() if task_details else {}
        details.setdefault("score_path", str(score_path))
        details.setdefault("eval_details_path", str(eval_path))
        with self._db.get_connection() as conn:
            self._repo.update_run_summary(
                conn,
                run_id=run_id,
                metrics=metrics,
                samples=samples,
                problems=problems,
                log_path=str(log_path),
                eval_details_path=str(eval_path),
                task=task,
                task_details=details,
            )
            self._repo.update_run_status(
                conn,
                run_id=run_id,
                status="succeeded",
                finished=True,
                start_now=False,
            )
