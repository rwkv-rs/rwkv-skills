"""Dispatch action and its launch/command helpers backed by the scheduler library."""

from __future__ import annotations

import os
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from src.eval.datasets.snapshot import (
    RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
    read_stable_file_bytes,
    validate_runtime_attestation_provenance,
)
from src.eval.evaluating.task_persistence import STRICT_RUNTIME_PROVENANCE_ENV

from . import actions_base as base
from .config import DEFAULT_PYTHON
from .actions_base import (
    CompletedKey,
    DispatchOptions,
    JobFailure,
    JobSpec,
    QueueItem,
    QueueOptions,
    RemoteConcurrencyBudget,
    RunningEntry,
    RunMode,
    RunnerGroup,
    SchedulerLeaseManager,
    SchedulerProgressSnapshot,
    SchedulerRuntimeControl,
)
from .control import DesiredState, ObservedStatus
from .remote_slots import parse_remote_model_slots, unique_remote_models


_STRICT_G1I_MODEL_RE = re.compile(
    r"^rwkv7-g1i-(?:1\.5|2\.9|7\.2|13\.3)b-\d{8}-ctx\d+$"
)


def require_strict_g1i_runtime_attestation(
    opts: DispatchOptions,
    *,
    phase: str = "dispatch",
) -> dict[str, object] | None:
    """Fail closed before a strict G1i dispatcher can mutate local/DB state.

    The shell launchers are useful orchestration boundaries, but they are not
    a security boundary: an operator can invoke this Python action directly.
    Keep the proof at the innermost shared dispatch entry point so legacy
    scripts and recovery callers cannot bypass runtime identity attestation.
    """

    slots = parse_remote_model_slots(opts.inference.models)
    models = unique_remote_models(slots)
    strict_models = tuple(
        model for model in models if _STRICT_G1I_MODEL_RE.fullmatch(model)
    )
    if not strict_models:
        return None
    if len(models) != 1 or len(strict_models) != 1:
        raise RuntimeError(
            "strict G1i dispatch requires exactly one attested physical model"
        )
    infer_base_url = str(opts.inference.base_url or "").strip()
    if not infer_base_url:
        raise RuntimeError("strict G1i dispatch requires one inference endpoint")
    normalized_base_url = infer_base_url.rstrip("/")
    if any(
        slot.base_url and slot.base_url.rstrip("/") != normalized_base_url
        for slot in slots
    ):
        raise RuntimeError(
            "strict G1i dispatch forbids per-slot endpoint identity drift"
        )

    frozen_raw = os.environ.get("RWKV_STRICT_FROZEN_RUNTIME", "").strip()
    approval_raw = os.environ.get("RWKV_GLOBAL_PROTOCOL_APPROVAL", "").strip()
    if not frozen_raw or not approval_raw:
        raise RuntimeError(
            "strict G1i dispatch requires frozen runtime and global approval"
        )
    frozen_runtime = Path(frozen_raw).expanduser().resolve(strict=True)
    executing_module = Path(__file__).resolve(strict=True)
    try:
        executing_module.relative_to(frozen_runtime)
    except ValueError as exc:
        raise RuntimeError(
            "strict G1i dispatcher is not executing from the frozen runtime"
        ) from exc
    gate = frozen_runtime / "ops/g1i_strict46/require_global_protocol_gate.py"
    lock = frozen_runtime / "ops/g1i_strict46/protocol_gate.lock.json"
    approval = frozen_runtime / "ops/g1i_strict46/approvals" / Path(
        approval_raw
    ).name
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(gate),
            "--repo",
            str(frozen_runtime),
            "--lock",
            str(lock),
            "--approval",
            str(approval),
            "--frozen-runtime",
            str(frozen_runtime),
            "--phase",
            phase,
            "--model",
            strict_models[0],
            "--infer-base-url",
            infer_base_url,
            "--infer-api-key",
            str(opts.inference.api_key or ""),
            "--require-current-python",
        ],
        cwd=frozen_runtime,
        check=True,
    )
    return _runtime_provenance_from_approval(
        approval_path=approval,
        lock_path=lock,
        model=strict_models[0],
        infer_base_url=infer_base_url,
    )


def _require_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def _self_digest(document: object, *, key: str, label: str) -> str:
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    expected = _require_sha256(document.get(key), label=f"{label}.{key}")
    unsigned = dict(document)
    unsigned.pop(key, None)
    if canonical_json_sha256(unsigned) != expected:
        raise RuntimeError(f"{label} self-digest mismatch")
    return expected


def _read_bound_json(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    try:
        payload_bytes = read_stable_file_bytes(path)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def _runtime_provenance_from_approval(
    *,
    approval_path: Path,
    lock_path: Path,
    model: str,
    infer_base_url: str,
) -> dict[str, object]:
    """Extract a compact task proof after the full live gate has passed.

    The gate validates the complete root-owned artifacts and live process
    chain.  This second stable read copies only content-addressed identities
    into task provenance so every database row remains traceable to that
    exact approval, lock, weight, GPU, unit, engine tree, and route.
    """

    approval, approval_file_sha = _read_bound_json(
        approval_path, label="global protocol approval"
    )
    approval_sha = _self_digest(
        approval,
        key="approval_sha256",
        label="global protocol approval",
    )
    lock, lock_file_sha = _read_bound_json(lock_path, label="protocol lock")
    lock_sha = _self_digest(lock, key="lock_sha256", label="protocol lock")

    approval_descriptor = lock.get("global_approval")
    if not isinstance(approval_descriptor, dict) or (
        approval_descriptor.get("sha256") != approval_file_sha
    ):
        raise RuntimeError("protocol lock does not bind the exact approval bytes")
    evidence = approval.get("runtime_attestation_evidence")
    evidence_sha = _self_digest(
        evidence,
        key="evidence_sha256",
        label="runtime attestation evidence",
    )
    if lock.get("runtime_attestation_evidence_sha256") != evidence_sha:
        raise RuntimeError("protocol lock runtime evidence binding mismatch")
    models = evidence.get("models") if isinstance(evidence, dict) else None
    entry = models.get(model) if isinstance(models, dict) else None
    if not isinstance(entry, dict) or set(entry) != {
        "runtime_attestation",
        "route",
    }:
        raise RuntimeError(f"runtime attestation is missing for {model}")
    runtime_artifact = entry["runtime_attestation"]
    runtime_sha = _self_digest(
        runtime_artifact,
        key="artifact_sha256",
        label="runtime attestation artifact",
    )
    route = entry["route"]
    if not isinstance(route, dict):
        raise RuntimeError("runtime attestation route is invalid")
    route_kind = route.get("kind")
    if route_kind not in {"local", "ssh_forward"}:
        raise RuntimeError("runtime attestation route kind is invalid")
    scheduler_endpoint = route.get("scheduler_endpoint")
    if not isinstance(scheduler_endpoint, dict):
        raise RuntimeError("runtime attestation scheduler endpoint is invalid")
    endpoint_url = (
        f"{scheduler_endpoint.get('scheme')}://{scheduler_endpoint.get('host')}:"
        f"{scheduler_endpoint.get('port')}{scheduler_endpoint.get('api_prefix')}"
    )
    if endpoint_url.rstrip("/") != infer_base_url.rstrip("/"):
        raise RuntimeError("runtime attestation endpoint changed after live verification")

    forward_sha: str | None = None
    if route_kind == "ssh_forward":
        forward_sha = _self_digest(
            route.get("forward_attestation"),
            key="artifact_sha256",
            label="forward attestation artifact",
        )
    elif set(route) != {"kind", "scheduler_endpoint"}:
        raise RuntimeError("local runtime route contains unexpected fields")

    if not isinstance(runtime_artifact, dict):
        raise RuntimeError("runtime attestation artifact is invalid")
    runtime_model = runtime_artifact.get("model")
    process = runtime_artifact.get("process")
    runtime_tree = runtime_artifact.get("runtime_tree")
    if (
        not isinstance(runtime_model, dict)
        or runtime_model.get("name") != model
        or not isinstance(runtime_model.get("weight"), dict)
        or not isinstance(process, dict)
        or not isinstance(process.get("executable"), dict)
        or not isinstance(runtime_tree, dict)
    ):
        raise RuntimeError("runtime attestation identity fields are invalid")

    provenance: dict[str, object] = {
        "schema_version": RUNTIME_ATTESTATION_PROVENANCE_SCHEMA_VERSION,
        "model": model,
        "route_kind": route_kind,
        "scheduler_endpoint": dict(scheduler_endpoint),
        "global_approval_sha256": approval_sha,
        "global_approval_file_sha256": approval_file_sha,
        "protocol_lock_sha256": lock_sha,
        "protocol_lock_file_sha256": lock_file_sha,
        "runtime_attestation_evidence_sha256": evidence_sha,
        "runtime_attestation_artifact_sha256": runtime_sha,
        "forward_attestation_artifact_sha256": forward_sha,
        "host_label": runtime_artifact.get("host_label"),
        "weight": dict(runtime_model["weight"]),
        "runtime_executable_sha256": process["executable"].get("sha256"),
        "runtime_tree_sha256": runtime_tree.get("tree_sha256"),
        "semantic_environment_sha256": canonical_json_sha256(
            process.get("environment")
        ),
        "launch_parameters_sha256": canonical_json_sha256(
            process.get("launch_parameters")
        ),
        "systemd_unit": process.get("systemd_unit"),
        "gpu_index": process.get("gpu_index"),
    }
    provenance["provenance_sha256"] = canonical_json_sha256(provenance)
    return validate_runtime_attestation_provenance(provenance)


def _select_remote_candidate(
    *,
    opts: DispatchOptions,
    queue: Sequence[QueueItem],
    skipped_remote_job_ids: set[str],
    resource_model_slug: str | None,
    active_coding_runners: int,
    resource: str,
) -> QueueItem | None:
    """Pick the next remote queue item whose model matches this worker slot.

    Marks every examined-and-rejected item (coding-limited or chosen) in
    ``skipped_remote_job_ids`` so a later slot does not reconsider it; items
    skipped only for a model-slug mismatch stay available for other slots.
    """
    for maybe in queue:
        if maybe.job_id in skipped_remote_job_ids:
            continue
        candidate_model_slug = base.safe_slug(maybe.infer_model or maybe.model_name or maybe.model_slug)
        if resource_model_slug is not None and candidate_model_slug != resource_model_slug:
            continue
        if _candidate_exceeds_coding_limit(
            opts=opts,
            candidate=maybe,
            active_coding_runners=active_coding_runners,
        ):
            base.log_job_event(
                "job_defer",
                maybe.job_id,
                reason="max_active_coding_runners",
                active_coding_runners=active_coding_runners,
                max_active_coding_runners=int(opts.coding.max_active_runners or 0),
                worker_slot=resource,
            )
            skipped_remote_job_ids.add(maybe.job_id)
            continue
        skipped_remote_job_ids.add(maybe.job_id)
        return maybe
    return None


def _claim_item_for_resource(
    *,
    opts: DispatchOptions,
    queue: Sequence[QueueItem],
    resource: str,
    remote_mode: bool,
    resource_slot_slug: str | None,
    resource_model_slug: str | None,
    queue_index: int,
    skipped_remote_job_ids: set[str],
    budgets: Mapping[str, RemoteConcurrencyBudget],
    occupied_remote_slots: set[str],
    lease_manager: SchedulerLeaseManager | None,
    active_coding_runners: int,
) -> tuple[QueueItem | None, int]:
    """Select and lease-claim the item (if any) this worker slot should launch.

    Returns the chosen item plus the advanced local-queue cursor, emitting the
    same ``job_defer`` / ``job_claim_conflict`` events the inline loop did.
    """
    while queue_index < len(queue):
        if remote_mode:
            candidate = _select_remote_candidate(
                opts=opts,
                queue=queue,
                skipped_remote_job_ids=skipped_remote_job_ids,
                resource_model_slug=resource_model_slug,
                active_coding_runners=active_coding_runners,
                resource=resource,
            )
            if candidate is None:
                break
        else:
            candidate = queue[queue_index]
            queue_index += 1
            if _candidate_exceeds_coding_limit(
                opts=opts,
                candidate=candidate,
                active_coding_runners=active_coding_runners,
            ):
                base.log_job_event(
                    "job_defer",
                    candidate.job_id,
                    reason="max_active_coding_runners",
                    active_coding_runners=active_coding_runners,
                    max_active_coding_runners=int(opts.coding.max_active_runners or 0),
                    worker_slot=resource,
                )
                continue
        if remote_mode:
            candidate_model_slug = base.safe_slug(candidate.infer_model or candidate.model_name or candidate.model_slug)
            budget = budgets.get(resource_slot_slug or candidate_model_slug)
            if budget is not None and not budget.launch_allowed:
                base.log_job_event(
                    "job_defer",
                    candidate.job_id,
                    reason=budget.reason,
                    infer_model=str(candidate.infer_model or candidate.model_name or ""),
                    worker_slot=resource,
                    pending_queue=budget.pending_queue,
                    source_status=budget.source_status,
                )
                continue
            if resource_slot_slug is not None and resource_slot_slug in occupied_remote_slots:
                base.log_job_event(
                    "job_defer",
                    candidate.job_id,
                    reason="remote_slot_busy",
                    infer_model=str(candidate.infer_model or candidate.model_name or ""),
                    worker_slot=resource,
                )
                continue
        if lease_manager is not None and not lease_manager.claim(
            candidate.job_id,
            lease_meta=base._lease_meta_for_item(candidate),
        ):
            base.log_job_event("job_claim_conflict", candidate.job_id, worker_slot=resource, **base._lease_meta_for_item(candidate))
            continue
        return candidate, queue_index
    return None, queue_index


def _launch_queue_items(
    *,
    opts: DispatchOptions,
    queue: Sequence[QueueItem],
    available_resources: Sequence[str],
    question_counts: Mapping[str, int],
    batch_profiler: base.BatchProfiler,
    pending_since: dict[str, float],
    launch_times: dict[str, float],
    job_metadata: dict[str, dict[str, object]],
    lease_manager: SchedulerLeaseManager | None,
    claimed_job_ids: set[str],
    skipped_missing_keys: set[CompletedKey] | None = None,
    generated_job_ids: Sequence[str] | set[str] = (),
    remote_budgets: Mapping[str, RemoteConcurrencyBudget] | None = None,
    runtime_provenance: Mapping[str, object] | None = None,
) -> None:
    if skipped_missing_keys is None:
        skipped_missing_keys = set()
    remote_mode = base._dispatch_uses_remote_inference(opts)
    resource_label = "Free slots" if remote_mode else "Idle GPUs"
    print(f"🧮 Pending={len(queue)} | {resource_label}={', '.join(available_resources)}")
    running_now = base.load_running(opts.pid_dir)
    remote_slots = base.remote_slot_map(opts.inference.models) if remote_mode else {}
    occupied_remote_slots = (
        base._running_remote_slot_slugs(
            running_now,
            opts.inference.models,
            generated_job_ids=generated_job_ids,
        )
        if remote_mode
        else set()
    )
    budgets = remote_budgets or {}
    active_coding_runners = _running_job_group_count(running_now, RunnerGroup.CODING)

    queue_index = 0
    skipped_remote_job_ids: set[str] = set()
    for resource in available_resources:
        resource_slot_slug = base._remote_resource_model_slug(resource) if remote_mode else None
        resource_model_slug = None
        if remote_mode and resource_slot_slug is not None:
            slot_spec = remote_slots.get(resource_slot_slug)
            resource_model_slug = slot_spec.model_slug if slot_spec is not None else resource_slot_slug
        item, queue_index = _claim_item_for_resource(
            opts=opts,
            queue=queue,
            resource=resource,
            remote_mode=remote_mode,
            resource_slot_slug=resource_slot_slug,
            resource_model_slug=resource_model_slug,
            queue_index=queue_index,
            skipped_remote_job_ids=skipped_remote_job_ids,
            budgets=budgets,
            occupied_remote_slots=occupied_remote_slots,
            lease_manager=lease_manager,
            active_coding_runners=active_coding_runners,
        )
        if item is None and remote_mode:
            continue
        if item is None:
            break

        job = base.JOB_CATALOGUE[item.job_name]
        dataset_slug = item.dataset_slug
        try:
            dataset_path = base.locate_dataset(dataset_slug, search=base.DATASET_ROOTS, output_root=base.DATA_OUTPUT_ROOT)
        except Exception as exc:
            if opts.skip_missing_dataset:
                print(f"⚠️  {item.job_id} 数据集不可用：{exc}. 已跳过。")
                skipped_missing_keys.add(
                    CompletedKey(
                        job=item.job_name,
                        model_slug=item.model_slug,
                        dataset_slug=dataset_slug,
                        is_cot=job.is_cot,
                    )
                )
                pending_since.pop(item.job_id, None)
                job_metadata.pop(item.job_id, None)
                if lease_manager is not None:
                    lease_manager.release((item.job_id,))
                base.log_job_event(
                    "job_skip",
                    item.job_id,
                    reason="unavailable_dataset",
                    dataset_slug=dataset_slug,
                    error=type(exc).__name__,
                )
                continue
            base.log_job_event(
                "job_error",
                item.job_id,
                reason="missing_dataset",
                dataset_slug=dataset_slug,
                error=type(exc).__name__,
            )
            if lease_manager is not None:
                lease_manager.release((item.job_id,))
            raise

        log_relpath = base.build_run_log_name(item.model_name or item.model_slug, dataset_slug, is_cot=job.is_cot)
        console_log_path = _allocate_console_log_path(opts.run_log_dir, log_relpath)
        pid_path = opts.pid_dir / f"{item.job_id}.pid"
        item.dataset_path = dataset_path
        if remote_mode and resource_slot_slug is not None:
            slot_spec = remote_slots.get(resource_slot_slug)
            if slot_spec is not None and slot_spec.base_url:
                item.infer_base_url = slot_spec.base_url

        if pid_path.exists():
            lines = pid_path.read_text().splitlines()
            if lines:
                try:
                    existing_pid = int(lines[0])
                except ValueError:
                    existing_pid = None
                else:
                    if existing_pid and existing_pid > 0:
                        print(f"ℹ️  {item.job_id} 已有运行中的 PID({existing_pid})，跳过")
                        base.log_job_event(
                            "job_skip",
                            item.job_id,
                            reason="already_running",
                            pid=existing_pid,
                        )
                        if lease_manager is not None:
                            lease_manager.release((item.job_id,))
                        continue
            pid_path.unlink(missing_ok=True)

        env = os.environ.copy()
        if runtime_provenance is not None:
            env[STRICT_RUNTIME_PROVENANCE_ENV] = canonical_json_bytes(
                runtime_provenance
            ).decode("utf-8")
        # The formal checkout may reuse the shared checkout's virtualenv. Its
        # editable-install .pth can otherwise put the shared source tree ahead
        # of this scheduler's cwd for child processes. Always make the
        # scheduler's own repository the first import root for every runner.
        repo_pythonpath = str(base.REPO_ROOT)
        inherited_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_pythonpath, inherited_pythonpath) if part
        )
        env.update(
            {
                "RWKV_SKILLS_JOB_ID": item.job_id,
                "RWKV_SKILLS_JOB_NAME": item.job_name,
                "RWKV_SKILLS_MODEL_NAME": str(item.model_name or item.model_slug),
                "RWKV_SKILLS_DATASET": str(dataset_path),
                "RWKV_SKILLS_DATASET_SLUG": dataset_slug,
                "RWKV_TASK_DESC": env.get("RWKV_TASK_DESC")
                or f"job={item.job_name}, dataset={dataset_slug}",
                "RUN_LOG_DIR": str(opts.log_dir),
                "RUN_RUN_LOG_DIR": str(opts.run_log_dir),
                "RWKV_EVAL_RUN_MODE": opts.run_mode.value,
                "RWKV_SCHEDULER_OVERWRITE": "1" if opts.run_mode is RunMode.RERUN else "0",
            }
        )
        if item.model_path is not None:
            env["RWKV_SKILLS_MODEL_PATH"] = str(item.model_path)
        if item.is_remote:
            env["RWKV_SKILLS_INFER_BASE_URL"] = str(item.infer_base_url or "")
            env["RWKV_SKILLS_INFER_MODEL"] = str(item.infer_model or item.model_name or "")
            env["CUDA_VISIBLE_DEVICES"] = ""
            if opts.inference.api_key:
                env["RWKV_SKILLS_INFER_API_KEY"] = opts.inference.api_key
            env["RWKV_SKILLS_INFER_PROTOCOL"] = str(opts.inference.protocol or "openai")
            env["RWKV_SKILLS_INFER_SEED_POLICY"] = str(opts.inference.seed_policy or "preserve")
        if opts.disable_checker:
            env["RWKV_SKILLS_DISABLE_CHECKER"] = "1"
        if opts.benchmark_config_root is not None:
            env["RWKV_BENCHMARK_CONFIG_ROOT"] = str(opts.benchmark_config_root)

        questions = question_counts.get(dataset_slug)

        batch_size = None
        item_budget = budgets.get(resource_slot_slug or "") if item.is_remote else None
        if item.is_remote and job.batch_flag:
            if item_budget is not None and item_budget.remote_batch_size is not None:
                batch_size = max(1, int(item_budget.remote_batch_size))
            elif opts.inference.remote_batch_size is not None:
                batch_size = max(1, int(opts.inference.remote_batch_size))
            if (
                opts.inference.plain_choice_batch_size is not None
                and item.job_name in {"multi_choice_plain", "multi_choice_plain_naive"}
            ):
                batch_size = max(1, int(opts.inference.plain_choice_batch_size))
        if not item.is_remote and item.model_path is not None:
            batch_size = batch_profiler.determine_batch_size(
                job=job,
                job_id=item.job_id,
                gpu=resource,
                dataset_path=dataset_path,
                model_path=item.model_path,
                model_slug=item.model_slug,
                env=env,
                dataset_questions=questions,
            )

        extra_args = (
            item.extra_args
            + _knowledge_extra_args(opts, job)
            + _coding_extra_args(opts, job)
            + _maths_extra_args(opts, job)
            + _function_calling_extra_args(opts, job)
        )
        if opts.run_mode is RunMode.RERUN and item.job_name == "param_search_select" and "--overwrite" not in extra_args:
            extra_args = extra_args + ("--overwrite",)
        infer_timeout_s = opts.inference.timeout_s
        if (
            opts.inference.plain_choice_timeout_s is not None
            and item.job_name in {"multi_choice_plain", "multi_choice_plain_naive"}
        ):
            infer_timeout_s = max(1.0, float(opts.inference.plain_choice_timeout_s))

        command = build_command(
            job,
            item,
            dataset_path,
            None if item.is_remote else f"cuda:{resource}",
            batch_size=batch_size,
            extra_args=extra_args,
            infer_api_key=opts.inference.api_key,
            infer_timeout_s=infer_timeout_s,
            infer_max_workers=(
                item_budget.infer_max_workers
                if item_budget is not None
                else opts.inference.max_workers
            ),
            infer_protocol=opts.inference.protocol,
            infer_seed_policy=opts.inference.seed_policy,
        )
        _backup_run_config(
            model_name=item.model_name or item.model_slug,
            model_path=item.model_path,
            infer_base_url=item.infer_base_url,
            infer_model=item.infer_model,
            infer_protocol=opts.inference.protocol,
            infer_seed_policy=opts.inference.seed_policy,
            dataset_slug=dataset_slug,
            dataset_path=dataset_path,
            job_name=item.job_name,
            job_id=item.job_id,
            batch_size=batch_size,
            sample_workers=opts.inference.sample_workers,
            infer_max_workers=(
                item_budget.infer_max_workers
                if item_budget is not None
                else opts.inference.max_workers
            ),
            budget_reason=(item_budget.reason if item_budget is not None else None),
            gpu=(None if item.is_remote else f"cuda:{resource}"),
            log_path=console_log_path,
        )
        print(f"🚀 Launch {item.job_id} -> {base._launch_target_label(item, resource)}")
        print(f"    Dataset: {dataset_path}")
        print(f"    Console: {console_log_path}")
        print(f"    Cmd: {' '.join(command)}")
        meta = job_metadata.setdefault(item.job_id, {})
        meta.update(
            job=item.job_name,
            dataset_slug=dataset_slug,
            dataset_path=str(dataset_path),
            model_name=item.model_name or item.model_slug,
            model_slug=item.model_slug,
            console_log_path=str(console_log_path),
        )
        if item.model_path is not None:
            meta["model_path"] = str(item.model_path)
        if item.is_remote:
            meta["infer_base_url"] = str(item.infer_base_url)
            meta["infer_model"] = str(item.infer_model or item.model_name)
            meta["remote_slot"] = resource
            if opts.inference.sample_workers is not None:
                meta["sample_workers"] = max(1, int(opts.inference.sample_workers))
            if item_budget is not None:
                meta["infer_max_workers"] = item_budget.infer_max_workers
                meta["remote_budget_reason"] = item_budget.reason
                if item_budget.remote_batch_size is not None:
                    meta["remote_batch_size"] = item_budget.remote_batch_size
        else:
            meta["gpu"] = resource

        try:
            process = base.launch_job(
                item.job_id,
                command,
                cwd=base.REPO_ROOT,
                log_path=console_log_path,
                env=env,
            )
        except Exception:
            if lease_manager is not None:
                lease_manager.release((item.job_id,))
            raise
        claimed_job_ids.add(item.job_id)
        try:
            log_reference = str(console_log_path.relative_to(opts.run_log_dir))
        except ValueError:
            log_reference = str(console_log_path)
        base.write_pid_file(opts.pid_dir, item.job_id, process.pid, resource, log_reference)
        launch_times[item.job_id] = base.time.time()
        pending_start = pending_since.pop(item.job_id, None)
        wait_s = base.time.time() - pending_start if pending_start else None
        payload: dict[str, object] = {
            "job": item.job_name,
            "dataset_slug": dataset_slug,
            "dataset_path": str(dataset_path),
            "model_name": item.model_name or item.model_slug,
            "pid": process.pid,
            "wait_s": wait_s,
        }
        if item.model_path is not None:
            payload["model_path"] = str(item.model_path)
            payload["gpu"] = f"cuda:{resource}"
        if item.is_remote:
            payload["infer_base_url"] = str(item.infer_base_url)
            payload["infer_model"] = str(item.infer_model or item.model_name)
            payload["worker_slot"] = resource
            if item_budget is not None:
                payload["infer_max_workers"] = item_budget.infer_max_workers
                payload["remote_budget_reason"] = item_budget.reason
                if item_budget.remote_batch_size is not None:
                    payload["remote_batch_size"] = item_budget.remote_batch_size
        base.log_job_event("job_launch", item.job_id, **payload)
        if remote_mode:
            if resource_slot_slug is not None:
                occupied_remote_slots.add(resource_slot_slug)
        if job.runner_group is RunnerGroup.CODING:
            active_coding_runners += 1


def action_dispatch(
    opts: DispatchOptions,
    *,
    runtime_control: SchedulerRuntimeControl | None = None,
) -> None:
    # This must remain the first operation: runtime attestation precedes
    # directory creation, queue/DB access, task creation, and runner spawn.
    runtime_provenance = require_strict_g1i_runtime_attestation(opts)
    base.ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    if opts.clean_param_swap:
        _clean_param_swap_records(opts.log_dir)

    batch_cache = opts.batch_cache_path or (opts.log_dir / "batch_cache.json")
    batch_profiler = base.BatchProfiler(batch_cache)
    job_priority = base._job_priority_map(opts.job_priority)

    base.FAILURE_MONITOR.reset()
    state = base.DispatcherState()
    session_completed: set[CompletedKey] = set()
    session_failed: set[CompletedKey] = set()
    skipped_missing_keys: set[CompletedKey] = set()
    previous_running: set[str] = set()
    claimed_job_ids: set[str] = set()
    pending_notice_printed = False
    lease_manager = base._build_lease_manager(opts)

    if runtime_control is not None:
        runtime_control.write_status(ObservedStatus.STARTING)

    while True:
        failure = base.FAILURE_MONITOR.wait_failure(timeout=0)
        if failure is not None:
            failure_meta = state.job_metadata.get(failure.job_id, {}).copy()
            failure_job = failure_meta.get("job")
            failure_dataset_slug = failure_meta.get("dataset_slug")
            failure_model_slug = failure_meta.get("model_slug")
            if (
                isinstance(failure_job, str)
                and isinstance(failure_dataset_slug, str)
                and isinstance(failure_model_slug, str)
            ):
                job_spec = base.JOB_CATALOGUE.get(failure_job)
                if job_spec is not None:
                    session_failed.add(
                        CompletedKey(
                            job=failure_job,
                            model_slug=failure_model_slug,
                            dataset_slug=failure_dataset_slug,
                            is_cot=job_spec.is_cot,
                        )
                    )
            base.handle_job_failure(
                failure,
                opts.pid_dir,
                state.job_metadata,
                state.launch_times,
                stop_running=False,
            )
            state.pending_since.pop(failure.job_id, None)
            state.cooldown_until.pop(failure.job_id, None)
            _handle_batch_failure(batch_profiler, failure, failure_meta)
            if lease_manager is not None and failure.job_id in claimed_job_ids:
                lease_manager.release((failure.job_id,))
            claimed_job_ids.discard(failure.job_id)
            if runtime_control is not None:
                runtime_control.write_status(
                    ObservedStatus.RUNNING,
                    error=f"{failure.job_id} exited with returncode={failure.returncode}",
                    progress=_build_progress_snapshot(
                        queue=(),
                        running_entries=base.load_running(opts.pid_dir),
                        completed_count=len(state.completed_versions),
                        available_gpus=(),
                    ),
                )
            print("⚠️  调度已隔离失败任务并继续。")
            continue

        completed, completed_records, running_entries, question_counts = base._read_scheduler_state(pid_dir=opts.pid_dir)
        failed_keys: set[CompletedKey] = set()
        now = base.time.time()

        completed_job_ids = base._reconcile_completed_versions(
            completed_records=completed_records,
            state=state,
            session_completed=session_completed,
            now=now,
        )
        if lease_manager is not None and completed_job_ids:
            lease_manager.release(tuple(completed_job_ids))
            claimed_job_ids.difference_update(completed_job_ids)

        # If a job stops without a new score, briefly avoid re-queueing to allow DB writes to land.
        cooldown_jobs = base._update_cooldown_jobs(
            previous_running=previous_running,
            running_entries=running_entries,
            completed_records=completed_records,
            state=state,
            now=now,
            dispatch_poll_seconds=opts.dispatch_poll_seconds,
        )
        previous_running = set(running_entries.keys())
        foreign_claimed_job_ids: set[str] = set()
        if lease_manager is not None:
            owned_running_jobs = {job_id for job_id in running_entries.keys() if job_id in claimed_job_ids}
            renewed_job_ids = lease_manager.renew(tuple(sorted(owned_running_jobs)))
            lost_job_ids = owned_running_jobs - renewed_job_ids
            if lost_job_ids:
                claimed_job_ids.difference_update(lost_job_ids)
                print(f"⚠️  已失去 lease：{', '.join(sorted(lost_job_ids))}")
                base.log_job_event(
                    "dispatcher_lease_lost",
                    "_dispatcher",
                    jobs=",".join(sorted(lost_job_ids)),
                )
            foreign_claimed_job_ids = lease_manager.active_foreign_job_ids()

        queue = base._build_pending_queue(
            opts,
            completed=base._completed_for_queue(
                run_mode=opts.run_mode,
                completed=completed,
                session_completed=session_completed,
            ),
            failed=failed_keys | skipped_missing_keys | session_failed,
            running=tuple(set(running_entries.keys()) | cooldown_jobs | foreign_claimed_job_ids),
            question_counts=question_counts,
            job_priority=job_priority,
        )
        base._mark_pending_jobs(
            queue=queue,
            state=state,
            now=now,
        )
        remote_budgets = base._resolve_remote_concurrency_budgets(opts) if base._dispatch_uses_remote_inference(opts) else {}
        generated_job_ids = (
            base._generated_running_job_ids(
                running_entries=running_entries,
                job_metadata=state.job_metadata,
            )
            if base._dispatch_uses_remote_inference(opts)
            else set()
        )
        available_resources = base._resolve_available_dispatch_resources(
            opts,
            running_entries,
            generated_job_ids=generated_job_ids,
            remote_budgets=remote_budgets,
        )
        progress = _build_progress_snapshot(
            queue=queue,
            running_entries=running_entries,
            completed_count=len(completed_records),
            available_gpus=available_resources,
        )
        desired_state = runtime_control.desired_state() if runtime_control is not None else DesiredState.RUNNING

        if desired_state is DesiredState.CANCELLED:
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.CANCELLING, progress=progress)
            if not state.cancel_requested:
                state.cancel_requested = True
                base.FAILURE_MONITOR.mark_aborting()
                base.stop_all_jobs(opts.pid_dir)
            if running_entries:
                base.time.sleep(1)
                continue
            if lease_manager is not None and claimed_job_ids:
                lease_manager.release(tuple(sorted(claimed_job_ids)))
                claimed_job_ids.clear()
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.CANCELLED, progress=progress)
            print("🛑 调度已取消")
            base.log_job_event("dispatcher_cancelled", "_dispatcher", completed=len(completed_records))
            return

        if not queue:
            running_count = len(running_entries)
            if running_count > 0:
                if runtime_control is not None:
                    status = ObservedStatus.PAUSING if desired_state is DesiredState.PAUSED else ObservedStatus.RUNNING
                    runtime_control.write_status(status, progress=progress)
                if not pending_notice_printed:
                    print(f"⏳ 所有任务已调度，等待 {running_count} 个任务完成…")
                    pending_notice_printed = True
                base.log_job_event(
                    "dispatcher_wait",
                    "_dispatcher",
                    reason="running",
                    running=running_count,
                    pending=0,
                )
                base.time.sleep(opts.dispatch_poll_seconds)
                continue
            if foreign_claimed_job_ids:
                if runtime_control is not None:
                    runtime_control.write_status(ObservedStatus.RUNNING, progress=progress)
                if not pending_notice_printed:
                    print(f"⏳ 当前节点无可启动任务，等待集群中 {len(foreign_claimed_job_ids)} 个 lease 任务完成…")
                    pending_notice_printed = True
                base.log_job_event(
                    "dispatcher_wait",
                    "_dispatcher",
                    reason="cluster_running",
                    foreign_claims=len(foreign_claimed_job_ids),
                    pending=0,
                    running=0,
                )
                base.time.sleep(opts.dispatch_poll_seconds)
                continue
            print("🎉 所有任务调度完成")
            base.log_job_event("dispatcher_done", "_dispatcher", completed=len(completed_records))
            if lease_manager is not None and claimed_job_ids:
                lease_manager.release(tuple(sorted(claimed_job_ids)))
                claimed_job_ids.clear()
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.COMPLETED, progress=progress)
            break

        pending_notice_printed = False
        if desired_state is DesiredState.PAUSED:
            status = ObservedStatus.PAUSING if running_entries else ObservedStatus.PAUSED
            if runtime_control is not None:
                runtime_control.write_status(status, progress=progress)
            base.time.sleep(opts.dispatch_poll_seconds)
            continue
        if runtime_control is not None:
            runtime_control.write_status(ObservedStatus.RUNNING, progress=progress)
        if not available_resources:
            running_count = len(running_entries)
            suffix = f"（当前运行 {running_count} 个任务）" if running_count else ""
            if base._dispatch_uses_remote_inference(opts) and remote_budgets and any(
                not budget.launch_allowed for budget in remote_budgets.values()
            ):
                print(f"⏳ 远端推理背压中，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
                wait_reason = "remote_backpressure"
            elif base._dispatch_uses_remote_inference(opts):
                print(f"⏳ 远端推理模型槽已占满，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
                wait_reason = "remote_model_slots_exhausted"
            else:
                print(f"⏳ 未检测到空闲 GPU，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
                wait_reason = "no_gpu"
            base.log_job_event(
                "dispatcher_wait",
                "_dispatcher",
                reason=wait_reason,
                pending=len(queue),
                running=running_count,
            )
            base.time.sleep(opts.dispatch_poll_seconds)
            continue

        _launch_queue_items(
            opts=opts,
            queue=queue,
            available_resources=available_resources,
            question_counts=question_counts,
            batch_profiler=batch_profiler,
            pending_since=state.pending_since,
            launch_times=state.launch_times,
            job_metadata=state.job_metadata,
            lease_manager=lease_manager,
            claimed_job_ids=claimed_job_ids,
            skipped_missing_keys=skipped_missing_keys,
            generated_job_ids=generated_job_ids,
            remote_budgets=remote_budgets,
            runtime_provenance=runtime_provenance,
        )

        base.time.sleep(1)


def _build_progress_snapshot(
    *,
    queue: Sequence[QueueItem],
    running_entries: Mapping[str, RunningEntry],
    completed_count: int,
    available_gpus: Sequence[str],
) -> SchedulerProgressSnapshot:
    return SchedulerProgressSnapshot(
        pending_jobs=len(queue),
        running_jobs=len(running_entries),
        completed_jobs=completed_count,
        failed_jobs=0,
        queue_head=tuple(item.job_id for item in queue[:8]),
        active_jobs=tuple(sorted(running_entries.keys())),
        available_gpus=tuple(available_gpus),
    )


def build_command(
    job: JobSpec,
    item: QueueItem,
    dataset_path: Path,
    device: str | None,
    *,
    batch_size: int | None = None,
    extra_args: Sequence[str] = (),
    infer_api_key: str = "",
    infer_timeout_s: float = 600.0,
    infer_max_workers: int = 32,
    infer_protocol: str = "openai",
    infer_seed_policy: str = "preserve",
) -> list[str]:
    base = [DEFAULT_PYTHON, "-m", job.module]
    args = ["--dataset", str(dataset_path)]
    if item.is_remote:
        args.extend(
            [
                "--infer-base-url",
                str(item.infer_base_url or ""),
                "--infer-model",
                str(item.infer_model or item.model_name or ""),
            ]
        )
        if infer_api_key:
            args.extend(["--infer-api-key", infer_api_key])
        args.extend(["--infer-timeout-s", str(float(infer_timeout_s))])
        args.extend(["--infer-max-workers", str(int(infer_max_workers))])
        args.extend(["--infer-protocol", str(infer_protocol or "openai")])
        args.extend(["--infer-seed-policy", str(infer_seed_policy or "preserve")])
    else:
        if item.model_path is None:
            raise ValueError("local scheduler launch requires model_path")
        args.extend(["--model-path", str(item.model_path)])
        if device:
            args.extend(["--device", device])
    if batch_size is not None and job.batch_flag:
        args.extend([job.batch_flag, str(batch_size)])
    if job.extra_args:
        args.extend(job.extra_args)
    if extra_args:
        args.extend(extra_args)
    return base + args


def _running_job_group_count(running_entries: Mapping[str, RunningEntry], group: RunnerGroup) -> int:
    count = 0
    for job_id in running_entries:
        for spec in base.JOB_CATALOGUE.values():
            if spec.runner_group is group and job_id.startswith(spec.id_prefix):
                count += 1
                break
    return count


def _candidate_exceeds_coding_limit(
    *,
    opts: QueueOptions,
    candidate: QueueItem,
    active_coding_runners: int,
) -> bool:
    if opts.coding.max_active_runners is None:
        return False
    limit = max(1, int(opts.coding.max_active_runners))
    job = base.JOB_CATALOGUE.get(candidate.job_name)
    return bool(job is not None and job.runner_group is RunnerGroup.CODING and active_coding_runners >= limit)


def _function_calling_extra_args(opts: QueueOptions, job: JobSpec) -> tuple[str, ...]:
    if job.runner_group is not RunnerGroup.FUNCTION_CALLING:
        return ()
    if job.module != "src.eval.tasks.function_calling.runner":
        return ()

    args: list[str] = []

    def _append(flag: str, value: int | None) -> None:
        if value is not None:
            args.extend([flag, str(max(1, int(value)))])

    def _append_str(flag: str, value: str | None) -> None:
        if value:
            args.extend([flag, str(value)])

    _append_str("--prompt-style", opts.functions.prompt_style)
    if job.name in base._SAMPLE_WORKER_JOB_NAMES:
        _append("--sample-workers", opts.inference.sample_workers)
    _append_str("--tool-catalog-format", opts.functions.tool_catalog_format)
    _append("--cot-max-tokens", opts.functions.cot_max_tokens)
    _append("--decision-max-tokens", opts.functions.decision_max_tokens)
    _append("--planning-max-tokens", opts.functions.planning_max_tokens)
    _append("--final-max-tokens", opts.functions.final_max_tokens)
    _append("--answer-max-tokens", opts.functions.answer_max_tokens)
    if job.name == "function_browsecomp":
        _append("--judge-max-workers", opts.functions.judge_max_workers)
    _append("--history-max-chars", opts.functions.history_max_chars)
    _append("--prompt-max-chars", opts.functions.prompt_max_chars)
    _append_str("--long-doc-mode", opts.functions.long_doc_mode)
    _append_str("--tool-router-mode", opts.functions.tool_router_mode)
    _append("--tool-router-max-tools", opts.functions.tool_router_max_tools)
    _append("--tool-router-trigger-tool-count", opts.functions.tool_router_trigger_tool_count)
    _append("--tool-router-trigger-catalog-chars", opts.functions.tool_router_trigger_catalog_chars)
    _append_str("--candidate-router-mode", opts.functions.candidate_router_mode)
    _append("--candidate-router-chunk-tools", opts.functions.candidate_router_chunk_tools)
    _append("--candidate-router-batch-size", opts.functions.candidate_router_batch_size)
    _append("--candidate-router-prompt-max-chars", opts.functions.candidate_router_prompt_max_chars)
    _append("--candidate-router-context-chars", opts.functions.candidate_router_context_chars)
    _append("--candidate-router-candidate-max-tokens", opts.functions.candidate_router_candidate_max_tokens)
    _append("--candidate-router-aggregate-max-tokens", opts.functions.candidate_router_aggregate_max_tokens)
    _append("--candidate-router-max-candidates", opts.functions.candidate_router_max_candidates)
    _append_str("--candidate-router-tool-schema-mode", opts.functions.candidate_router_tool_schema_mode)
    _append("--candidate-router-evidence-chars", opts.functions.candidate_router_evidence_chars)
    _append("--candidate-router-policy-chars", opts.functions.candidate_router_policy_chars)
    _append("--max-rounds", opts.functions.max_rounds)
    _append("--max-steps", opts.functions.max_steps)
    _append("--max-tool-errors", opts.functions.max_tool_errors)
    if opts.functions.complexfuncbench_disable_response_eval:
        args.append("--complexfuncbench-disable-response-eval")
    if opts.functions.complexfuncbench_offline_compare:
        args.append("--complexfuncbench-offline-compare")
    return tuple(args)


def _coding_extra_args(opts: QueueOptions, job: JobSpec) -> tuple[str, ...]:
    if job.runner_group is not RunnerGroup.CODING:
        return ()
    args: list[str] = []
    if opts.coding.eval_workers is not None:
        args.extend(["--eval-workers", str(max(1, int(opts.coding.eval_workers)))])
    return tuple(args)


def _maths_extra_args(opts: QueueOptions, job: JobSpec) -> tuple[str, ...]:
    if job.runner_group is not RunnerGroup.MATHS:
        return ()
    args: list[str] = []
    disable_oracle_cascade = os.environ.get("RWKV_MATH_DISABLE_ORACLE_CASCADE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not disable_oracle_cascade and job.name in {
        "free_response",
        "free_response_naive",
        "free_response_judge",
        "free_response_judge_naive",
    }:
        # Every Math prompt profile uses the explicit A/B/C contract: A is a
        # single completion; only A failures enter the two-stage B/C cascade.
        args.append("--strategy-a-single-generation")
    if job.name in {"free_response_judge", "free_response_judge_naive"} and opts.math.judge_max_workers is not None:
        args.extend(["--judge-max-workers", str(max(1, int(opts.math.judge_max_workers)))])
    if opts.math.prompt_max_chars is not None:
        args.extend(["--prompt-max-chars", str(max(1, int(opts.math.prompt_max_chars)))])
    if opts.math.long_doc_mode:
        args.extend(["--long-doc-mode", str(opts.math.long_doc_mode)])
    return tuple(args)


def _knowledge_extra_args(opts: QueueOptions, job: JobSpec) -> tuple[str, ...]:
    if job.runner_group is not RunnerGroup.KNOWLEDGE:
        return ()
    args: list[str] = []
    if opts.knowledge.prompt_max_chars is not None:
        args.extend(["--prompt-max-chars", str(max(1, int(opts.knowledge.prompt_max_chars)))])
    if opts.knowledge.long_doc_mode:
        args.extend(["--long-doc-mode", str(opts.knowledge.long_doc_mode)])
    return tuple(args)


def _clean_param_swap_records(log_dir: Path) -> None:
    target = (log_dir / "param_swap").resolve()
    if not target.exists():
        return
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    print(f"🧹 已清理参数搜索记录: {target}")


def _allocate_console_log_path(base_dir: Path, rel: Path) -> Path:
    target_dir = base_dir / rel.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{rel.name}.log"
    if not candidate.exists():
        return candidate
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    candidate = target_dir / f"{rel.name}--{timestamp}.log"
    if not candidate.exists():
        return candidate
    attempt = 1
    while True:
        numbered = target_dir / f"{rel.name}--{timestamp}-{attempt}.log"
        if not numbered.exists():
            return numbered
        attempt += 1


def _handle_batch_failure(batch_profiler: base.BatchProfiler, failure: JobFailure, metadata: Mapping[str, object]) -> None:
    if not metadata:
        return
    job_name = metadata.get("job")
    model_slug = metadata.get("model_slug")
    gpu = metadata.get("gpu")
    if not job_name or not model_slug or gpu is None:
        return
    log_path = failure.log_path
    if not log_path.exists():
        return
    if not _log_contains_oom(log_path):
        return
    reason = f"runtime oom ({failure.job_id})"
    batch_profiler.invalidate_cache(str(job_name), str(model_slug), str(gpu), reason=reason)
    print(f"⚠️  {failure.job_id} 日志包含 OOM，已清理 {job_name}/{model_slug} 在 GPU {gpu} 的批量缓存。")


def _log_contains_oom(log_path: Path, *, tail_bytes: int = 65536) -> bool:
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes), os.SEEK_SET)
            chunk = fh.read()
    except OSError:
        return False
    text = chunk.decode("utf-8", errors="ignore").lower()
    keywords = ("out of memory", "cuda oom", "cuda out of memory", "torch.outofmemoryerror")
    return any(keyword in text for keyword in keywords)


def _backup_run_config(
    *,
    model_name: str,
    model_path: Path | None,
    infer_base_url: str | None,
    infer_model: str | None,
    infer_protocol: str | None,
    infer_seed_policy: str | None,
    dataset_slug: str,
    dataset_path: Path,
    job_name: str,
    job_id: str,
    batch_size: int | None,
    sample_workers: int | None,
    infer_max_workers: int | None,
    budget_reason: str | None,
    gpu: str | None,
    log_path: Path,
) -> Path:
    benchmark, _ = base.split_benchmark_and_split(dataset_slug)
    config_path = base.config_path_for_benchmark(benchmark, model_name)
    model_dir = base.safe_slug(model_name)
    benchmark_dir = base.safe_slug(benchmark)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = base.REPO_ROOT / "config_backup" / model_dir / benchmark_dir / f"{timestamp}.toml"
    target.parent.mkdir(parents=True, exist_ok=True)

    base_text = ""
    if config_path.exists():
        base_text = config_path.read_text(encoding="utf-8")

    run_block = _render_run_block(
        benchmark=benchmark,
        dataset_slug=dataset_slug,
        model_name=model_name,
        model_path=model_path,
        infer_base_url=infer_base_url,
        infer_model=infer_model,
        infer_protocol=infer_protocol,
        infer_seed_policy=infer_seed_policy,
        config_path=config_path,
        job_name=job_name,
        job_id=job_id,
        batch_size=batch_size,
        sample_workers=sample_workers,
        infer_max_workers=infer_max_workers,
        budget_reason=budget_reason,
        gpu=gpu,
        dataset_path=dataset_path,
        log_path=log_path,
    )
    separator = "\n\n" if base_text.strip() else ""
    target.write_text(f"{base_text.rstrip()}{separator}{run_block}", encoding="utf-8")
    return target


def _render_run_block(
    *,
    benchmark: str,
    dataset_slug: str,
    model_name: str,
    model_path: Path | None,
    infer_base_url: str | None,
    infer_model: str | None,
    infer_protocol: str | None,
    infer_seed_policy: str | None,
    config_path: Path,
    job_name: str,
    job_id: str,
    batch_size: int | None,
    sample_workers: int | None,
    infer_max_workers: int | None,
    budget_reason: str | None,
    gpu: str | None,
    dataset_path: Path,
    log_path: Path,
) -> str:
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "[run]",
        f"created_at = {_toml_quote(created_at)}",
        f"benchmark = {_toml_quote(benchmark)}",
        f"dataset_slug = {_toml_quote(dataset_slug)}",
        f"model_name = {_toml_quote(model_name)}",
        f"config_path = {_toml_quote(str(config_path))}",
        f"job_name = {_toml_quote(job_name)}",
        f"job_id = {_toml_quote(job_id)}",
        f"dataset_path = {_toml_quote(str(dataset_path))}",
        f"log_path = {_toml_quote(str(log_path))}",
    ]
    if model_path is not None:
        lines.append(f"model_path = {_toml_quote(str(model_path))}")
    if infer_base_url:
        lines.append(f"infer_base_url = {_toml_quote(str(infer_base_url))}")
    if infer_model:
        lines.append(f"infer_model = {_toml_quote(str(infer_model))}")
    if infer_protocol:
        lines.append(f"infer_protocol = {_toml_quote(str(infer_protocol))}")
    if infer_seed_policy:
        lines.append(f"infer_seed_policy = {_toml_quote(str(infer_seed_policy))}")
    if gpu:
        lines.append(f"gpu = {_toml_quote(gpu)}")
    if batch_size is not None:
        lines.append(f"batch_size = {int(batch_size)}")
    if sample_workers is not None:
        lines.append(f"sample_workers = {max(1, int(sample_workers))}")
    if infer_max_workers is not None:
        lines.append(f"infer_max_workers = {int(infer_max_workers)}")
    if budget_reason:
        lines.append(f"remote_budget_reason = {_toml_quote(str(budget_reason))}")
    return "\n".join(lines) + "\n"


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


__all__ = [
    "action_dispatch",
    "require_strict_g1i_runtime_attestation",
    "_launch_queue_items",
    "_build_progress_snapshot",
    "build_command",
    "_running_job_group_count",
    "_candidate_exceeds_coding_limit",
    "_function_calling_extra_args",
    "_knowledge_extra_args",
    "_coding_extra_args",
    "_maths_extra_args",
    "_clean_param_swap_records",
    "_allocate_console_log_path",
    "_handle_batch_failure",
    "_log_contains_oom",
    "_backup_run_config",
    "_render_run_block",
    "_toml_quote",
]
