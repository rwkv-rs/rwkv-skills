from __future__ import annotations

"""Queue planning action backed by the scheduler library."""

from . import actions_base as base
from .actions_base import QueueOptions
from .queue import QueueItem


def action_queue(opts: QueueOptions) -> list[QueueItem]:
    completed, score_records, running_entries, question_counts = base._read_scheduler_state(pid_dir=opts.pid_dir)
    failed = {
        record.key for record in score_records.values() if getattr(record, "missing_artifacts", False)
    }
    lease_manager = base._build_lease_manager(opts)
    cluster_claimed_job_ids = lease_manager.active_foreign_job_ids() if lease_manager is not None else set()
    job_priority_map = base._job_priority_map(opts.job_priority)
    pending = base._build_pending_queue(
        opts,
        completed=base._completed_for_queue(run_mode=opts.run_mode, completed=completed),
        failed=failed,
        running=tuple(set(running_entries.keys()) | cluster_claimed_job_ids),
        question_counts=question_counts,
        job_priority=job_priority_map,
    )
    base._print_queue_summary(pending, running_entries)
    return pending


__all__ = ["action_queue"]
