from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug
from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.infra.database import DatabaseManager

from .eval_db_repo import EvalDbRepository, StageOutputRow


@dataclass(slots=True)
class RunContext:
    subject_id: str
    split_id: str
    task_uuid: str
    run_id: str
    dataset_slug: str
    split_name: str
    model_slug: str
    task_id: str
    run_tag: str


class EvalDbService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._repo = EvalDbRepository()

    def prepare_run(
        self,
        *,
        dataset_slug: str,
        split_name: str,
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
        split_name = split_name or "test"
        model_slug = safe_slug(Path(model_path).stem)
        task_id = f"{dataset_slug}__{model_slug}"
        run_tag = run_tag or safe_slug(f"{dataset_slug}__{model_slug}")

        job_name = detect_job_from_dataset(dataset_slug, is_cot=is_cot)
        domain = JOB_CATALOGUE[job_name].domain if job_name and job_name in JOB_CATALOGUE else None

        with self._db.get_connection() as conn:
            subject_id = self._repo.upsert_subject(
                conn,
                dataset_slug=dataset_slug,
                domain=domain,
                dataset_version=None,
                dataset_meta=None,
                model_slug=model_slug,
                model_name=model_slug,
                model_revision=None,
                provider="local",
                model_meta={"path": model_path},
            )
            split_id = self._repo.upsert_split(conn, subject_id=subject_id, split_name=split_name)
            task_uuid = self._repo.upsert_task(
                conn,
                task_id=task_id,
                subject_id=subject_id,
                task_tag=task_tag,
                meta={"job": job_name},
            )
            if overwrite_run:
                self._repo.delete_run_by_tag(conn, task_id=task_uuid, run_tag=run_tag)
            run_id = self._repo.get_run_by_tag(conn, task_id=task_uuid, run_tag=run_tag)
            if not run_id:
                run_id = self._repo.insert_run(
                    conn,
                    task_id=task_uuid,
                    run_tag=run_tag,
                    sampling_config=sampling_config,
                    runtime_config=runtime_config,
                    code_version=code_version,
                    status="running",
                )
            return RunContext(
                subject_id=subject_id,
                split_id=split_id,
                task_uuid=task_uuid,
                run_id=run_id,
                dataset_slug=dataset_slug,
                split_name=split_name,
                model_slug=model_slug,
                task_id=task_id,
                run_tag=run_tag,
            )

    def mark_run_status(self, *, run_id: str, status: str, error_msg: str | None = None) -> None:
        with self._db.get_connection() as conn:
            self._repo.update_run_status(conn, run_id=run_id, status=status, error_msg=error_msg, finished=True)

    def upsert_sample(
        self,
        *,
        subject_id: str,
        split_id: str,
        sample_index: int,
        question: str | None,
        reference_answer: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        with self._db.get_connection() as conn:
            return self._repo.upsert_sample(
                conn,
                subject_id=subject_id,
                split_id=split_id,
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
                sample_index = int(payload.get("sample_index", 0))
                repeat_index = int(payload.get("repeat_index", 0))
                run_sample_id = self._repo.get_run_sample_id(
                    conn,
                    run_id=run_id,
                    sample_index=sample_index,
                    repeat_index=repeat_index,
                )
                if not run_sample_id:
                    continue
                is_passed = bool(payload.get("is_passed", False))
                value_num = 1.0 if is_passed else 0.0
                meta = {
                    "answer": payload.get("answer"),
                    "ref_answer": payload.get("ref_answer"),
                    "fail_reason": payload.get("fail_reason"),
                }
                self._repo.insert_metric(
                    conn,
                    run_sample_id=run_sample_id,
                    name=metric_name,
                    value_num=value_num,
                    value_text=str(is_passed),
                    meta=meta,
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
        meta = {
            "metrics": metrics,
            "samples": samples,
            "problems": problems,
            "log_path": str(log_path),
            "eval_path": str(eval_path),
            "score_path": str(score_path),
            "task": task,
            "task_details": task_details or {},
        }
        with self._db.get_connection() as conn:
            self._repo.insert_run_event(
                conn,
                run_id=run_id,
                run_sample_id=None,
                event_type=event_type,
                message="score_summary",
                meta=meta,
            )

