"""Merge partitioned read-only free-response replay audits and enforce gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--partitions", type=int, required=True)
    parser.add_argument(
        "--groups", default="strategy_a,strategy_b,strategy_c"
    )
    parser.add_argument("--expected-merge-script-sha256", help=argparse.SUPPRESS)
    return parser.parse_args()


def _counter_add(target: Counter[str], values: dict[str, Any]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _nested_counter_add(
    target: dict[str, Counter[str]], values: dict[str, Any]
) -> None:
    for group, counts in values.items():
        if isinstance(counts, dict):
            _counter_add(target[str(group)], counts)


def _load_parts(
    prefix: Path, groups: Iterable[str], partitions: int
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    missing: list[str] = []
    expected_paths: set[Path] = set()
    for group in groups:
        for index in range(partitions):
            path = Path(f"{prefix}_{group}_p{index}.json")
            expected_paths.add(path.resolve())
            if not path.exists():
                missing.append(str(path))
                continue
            payload, artifact_sha = _read_immutable_artifact(
                path,
                label="audit part",
            )
            document = json.loads(payload.decode("utf-8"))
            document["_artifact"] = str(path)
            document["_artifact_sha256"] = artifact_sha
            document["_artifact_bytes"] = len(payload)
            document["_expected_group"] = group
            document["_expected_partition_index"] = index
            parts.append(document)
    if missing:
        raise FileNotFoundError(f"missing audit artifacts: {missing}")
    pattern = re.compile(
        rf"^{re.escape(prefix.name)}_(strategy_[abc])_p\d+\.json$"
    )
    discovered = {
        path.resolve()
        for path in prefix.parent.glob(f"{prefix.name}_strategy_*_p*.json")
        if pattern.fullmatch(path.name)
    }
    extras = sorted(discovered.difference(expected_paths))
    if extras:
        raise RuntimeError(f"unexpected audit artifacts for prefix: {extras}")
    return parts


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _read_immutable_artifact(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, str]:
    """Read one regular, read-only artifact through one non-symlink fd."""

    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    resolved = path.expanduser().absolute()
    flags = os.O_RDONLY
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {resolved}")
        if metadata.st_mode & 0o222:
            raise RuntimeError(f"{label} is writable: {resolved}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    return payload, hashlib.sha256(payload).hexdigest()


def _freeze_merge_script(source: Path, directory: Path) -> tuple[Path, str]:
    """Publish the exact merger bytes under a content-addressed filename."""

    payload = source.resolve().read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / f"{digest}.py"
    if target.exists():
        existing, existing_sha = _read_immutable_artifact(
            target,
            label="frozen merge script",
        )
        if existing_sha != digest or existing != payload:
            raise RuntimeError(f"invalid existing frozen merge script: {target}")
        return target.resolve(), digest
    temporary = directory / f".{target.name}.tmp.{os.getpid()}.{uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, target)
        _fsync_directory(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    frozen, actual = _read_immutable_artifact(
        target,
        label="frozen merge script",
    )
    if actual != digest or frozen != payload:
        raise RuntimeError(f"failed to freeze merge script: {target}")
    return target.resolve(), digest


def _bootstrap_or_verify_merge_script(args: argparse.Namespace) -> str:
    """Re-exec the merger from immutable bytes, then bind those bytes."""

    expected = str(args.expected_merge_script_sha256 or "")
    current = Path(__file__).resolve()
    if not expected:
        frozen, digest = _freeze_merge_script(
            current,
            args.output_json.resolve().parent / ".free-response-merge-code",
        )
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(frozen),
                *sys.argv[1:],
                "--expected-merge-script-sha256",
                digest,
            ],
            check=False,
        )
        raise SystemExit(completed.returncode)
    payload, actual = _read_immutable_artifact(
        current,
        label="merge script",
    )
    del payload
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise RuntimeError(
            f"merge script SHA mismatch: {actual} != {expected}"
        )
    if current.stem != expected:
        raise RuntimeError(f"merge script is not content-addressed: {current}")
    return actual


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_path(path: Path, state: str) -> Path:
    return path.with_name(f"{path.stem}.{state}{path.suffix}")


def _acceptance_manifest_path(output_json: Path) -> Path:
    return Path(f"{output_json}.acceptance.json")


def _assert_output_paths_fresh(output_json: Path, output_md: Path) -> None:
    paths = (
        output_json,
        output_md,
        _artifact_path(output_json, "failed"),
        _artifact_path(output_md, "failed"),
        _acceptance_manifest_path(output_json),
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite merge evidence: " + ", ".join(existing)
        )


def _validate_tree_contract(
    value: Any,
    *,
    label: str,
    require_files: bool,
) -> list[dict[str, Any]]:
    contract = dict(value or {})
    records = contract.get("files") or []
    if not isinstance(records, list) or (require_files and not records):
        raise RuntimeError(f"missing {label} file manifest")
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid {label} file record")
        path = str(record.get("path") or "")
        digest = str(record.get("sha256") or "")
        size = record.get("bytes")
        if not path or size is None or int(size) < 0 or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise RuntimeError(f"invalid {label} file record: {path}")
        normalized.append(
            {"path": path, "bytes": int(size), "sha256": digest}
        )
    normalized.sort(key=lambda record: str(record["path"]))
    if records != normalized:
        raise RuntimeError(f"non-canonical {label} file manifest")
    expected_digest = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    if (
        int(contract.get("file_count") or 0) != len(normalized)
        or int(contract.get("total_bytes") or 0)
        != sum(int(record["bytes"]) for record in normalized)
        or str(contract.get("sha256") or "") != expected_digest
    ):
        raise RuntimeError(f"{label} tree digest mismatch")
    return normalized


def _validate_dependency_contract(environment: dict[str, Any]) -> None:
    if environment.get("schema_version") != (
        "free-response-global-audit-dependencies.v1"
    ):
        raise RuntimeError("unsupported dependency manifest schema")
    for key in (
        "python",
        "python_executable",
        "python_cache_tag",
        "platform",
        "machine",
    ):
        if not str(environment.get(key) or ""):
            raise RuntimeError(f"missing dependency runtime identity: {key}")
    src_records = _validate_tree_contract(
        environment.get("local_src_contract"),
        label="local src contract",
        require_files=True,
    )
    if any(not str(record["path"]).startswith("src/") for record in src_records):
        raise RuntimeError("local src contract contains a non-src path")
    metadata_records = _validate_tree_contract(
        environment.get("project_metadata_contract"),
        label="project metadata contract",
        require_files=True,
    )
    project_records = _validate_tree_contract(
        environment.get("project_contract"),
        label="project contract",
        require_files=True,
    )
    if project_records != sorted(
        [*src_records, *metadata_records],
        key=lambda record: str(record["path"]),
    ):
        raise RuntimeError("project contract does not equal src plus metadata")
    frozen_root = Path(str(environment.get("frozen_project_root") or ""))
    project_sha = str(dict(environment["project_contract"])["sha256"])
    if not frozen_root.is_absolute() or frozen_root.name != f"project-{project_sha}":
        raise RuntimeError("dependency manifest project root is not content-addressed")
    packages = dict(environment.get("packages") or {})
    for name in (
        "math-verify",
        "sympy",
        "latex2sympy2-extended",
        "antlr4-python3-runtime",
        "psycopg",
    ):
        package = dict(packages.get(name) or {})
        if not str(package.get("version") or ""):
            raise RuntimeError(f"missing critical dependency tree: {name}")
        _validate_tree_contract(
            package,
            label=f"dependency {name}",
            require_files=True,
        )


def _validate_production_parts(
    parts: list[dict[str, Any]],
    *,
    groups: list[str],
    partitions: int,
) -> dict[str, Any]:
    """Reject legacy, partial, stale, or cross-snapshot artifacts."""

    if groups != ["strategy_a", "strategy_b", "strategy_c"]:
        raise RuntimeError(
            "production merge requires exactly strategy_a,strategy_b,strategy_c"
        )
    if partitions < 1:
        raise RuntimeError("production merge requires positive partitions")
    if len(parts) != len(groups) * partitions:
        raise RuntimeError("partition cardinality mismatch")
    artifact_records: list[dict[str, Any]] = []
    for part in parts:
        artifact_sha = str(part.get("_artifact_sha256") or "")
        artifact_bytes = part.get("_artifact_bytes")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", artifact_sha)
            or artifact_bytes is None
            or int(artifact_bytes) <= 0
        ):
            raise RuntimeError("missing immutable audit-part provenance")
        artifact_records.append(
            {
                "group": str(part.get("_expected_group") or ""),
                "partition_index": int(
                    part.get("_expected_partition_index", -1)
                ),
                "path": str(
                    Path(str(part.get("_artifact") or "")).expanduser().absolute()
                ),
                "bytes": int(artifact_bytes),
                "sha256": artifact_sha,
            }
        )
    artifact_records.sort(
        key=lambda value: (str(value["group"]), int(value["partition_index"]))
    )
    artifact_set_sha = hashlib.sha256(
        _canonical_json_bytes(artifact_records)
    ).hexdigest()
    first = parts[0]
    if first.get("schema_version") != "free-response-global-audit-part.v3":
        raise RuntimeError("unsupported/legacy audit part schema")
    audit_mode = dict(first.get("audit_mode") or {})
    required_mode = {
        "full_scan_a": True,
        "full_real_scorer": True,
        "max_structural_rows": None,
        "database_snapshot_imported": True,
        "metadata_snapshot_digest_verified": True,
        "question_source": "current_production_snapshot",
        "independent_order_probe": True,
        "stable_row_order": "task_id,completions_id,eval_id",
        "frozen_code_verified": True,
        "frozen_src_contract_verified": True,
        "dependency_manifest_verified": True,
        "atomic_part_artifact": True,
    }
    mode_mismatches = {
        key: {"required": required, "actual": audit_mode.get(key)}
        for key, required in required_mode.items()
        if audit_mode.get(key) != required
    }
    if mode_mismatches:
        raise RuntimeError(
            f"legacy/non-production audit mode is disabled: {mode_mismatches}"
        )
    if int(audit_mode.get("order_consistency_probe_rows") or 0) <= 0:
        raise RuntimeError("module-order consistency probing is disabled")

    metadata_summary = dict(first.get("metadata_snapshot") or {})
    if metadata_summary.get("schema_version") != (
        "free-response-global-audit-metadata.v2"
    ):
        raise RuntimeError("unsupported metadata snapshot schema")
    metadata_digest = str(metadata_summary.get("digest") or "")
    identity = dict(metadata_summary.get("database_identity") or {})
    snapshot_id = str(identity.get("exported_snapshot_id") or "")
    dataset_digests = dict(metadata_summary.get("dataset_digests") or {})
    if not re.fullmatch(r"[0-9a-f]{64}", metadata_digest):
        raise RuntimeError("missing/invalid metadata snapshot digest")
    if not snapshot_id:
        raise RuntimeError("missing PostgreSQL exported snapshot id")
    if not dataset_digests:
        raise RuntimeError("missing dataset snapshot digests")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(metadata_summary.get("task_families_digest") or ""),
    ):
        raise RuntimeError("missing/invalid strategy task-family digest")
    for key, value in dataset_digests.items():
        digest = dict(value) if isinstance(value, dict) else {}
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest.get("file_sha256") or "")):
            raise RuntimeError(f"invalid dataset file digest: {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest.get("records_sha256") or "")):
            raise RuntimeError(f"invalid dataset record digest: {key}")
        if int(digest.get("record_count") or 0) <= 0:
            raise RuntimeError(f"invalid dataset record count: {key}")

    module_shas = {
        "audit": str(first.get("audit_script_sha256") or ""),
        "baseline": str(first.get("baseline_module_sha256") or ""),
        "candidate": str(first.get("candidate_module_sha256") or ""),
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in module_shas.values()
    ):
        raise RuntimeError("missing/invalid evaluator module SHA")
    dependency_environment = dict(first.get("dependency_environment") or {})
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(dependency_environment.get("sha256") or ""),
    ):
        raise RuntimeError("missing dependency-environment SHA")
    actual_dependency_sha = hashlib.sha256(
        _canonical_json_bytes(
            {
                key: value
                for key, value in dependency_environment.items()
                if key != "sha256"
            }
        )
    ).hexdigest()
    if dependency_environment.get("sha256") != actual_dependency_sha:
        raise RuntimeError("dependency-environment digest mismatch")
    _validate_dependency_contract(dependency_environment)
    dependency_file_sha = str(
        first.get("dependency_manifest_file_sha256") or ""
    )
    if not re.fullmatch(r"[0-9a-f]{64}", dependency_file_sha):
        raise RuntimeError("missing/invalid dependency manifest file SHA")
    if hashlib.sha256(_canonical_json_bytes(dependency_environment)).hexdigest() != (
        dependency_file_sha
    ):
        raise RuntimeError("dependency manifest file SHA mismatch")

    task_inventory = dict(first.get("task_inventory") or {})
    task_inventory_digest = str(first.get("task_inventory_digest") or "")
    actual_inventory_digest = hashlib.sha256(
        _canonical_json_bytes(task_inventory)
    ).hexdigest()
    if not task_inventory or task_inventory_digest != actual_inventory_digest:
        raise RuntimeError("task inventory digest mismatch")
    if metadata_summary.get("task_inventory_digest") != task_inventory_digest:
        raise RuntimeError("task inventory is not bound to metadata snapshot summary")
    raw_task_families = dict(first.get("task_families") or {})
    task_families: dict[str, list[int]] = {}
    for raw_task_id, raw_family in raw_task_families.items():
        task_id = str(raw_task_id)
        if (
            not task_id.isdecimal()
            or str(int(task_id)) != task_id
            or not isinstance(raw_family, list)
            or not raw_family
            or any(type(member) is not int for member in raw_family)
        ):
            raise RuntimeError(f"invalid strategy task family encoding: {task_id}")
        family = list(raw_family)
        if family != sorted(set(family)):
            raise RuntimeError(f"non-canonical strategy task family: {task_id}")
        task_families[task_id] = family
    actual_family_digest = hashlib.sha256(
        _canonical_json_bytes(task_families)
    ).hexdigest()
    if metadata_summary.get("task_families_digest") != actual_family_digest:
        raise RuntimeError("strategy task-family digest mismatch")
    if set(task_families) != set(task_inventory):
        raise RuntimeError("strategy task-family inventory mismatch")
    for task_id, family in task_families.items():
        if int(task_id) not in family or {
            str(value) for value in family
        }.difference(task_inventory):
            raise RuntimeError(f"invalid strategy task family: {task_id}")
        for member in family:
            if task_families.get(str(member)) != family:
                raise RuntimeError(
                    f"asymmetric strategy task family: {task_id}->{member}"
                )
    inventory_rows = sum(
        int(dict(value).get("rows") or 0)
        for value in task_inventory.values()
        if isinstance(value, dict)
    )
    inventory_nonempty_tasks = sum(
        int(dict(value).get("rows") or 0) > 0
        for value in task_inventory.values()
        if isinstance(value, dict)
    )
    if int(first.get("database_rows") or -1) != inventory_rows:
        raise RuntimeError("database row inventory total mismatch")
    if int(first.get("tasks") or -1) != inventory_nonempty_tasks:
        raise RuntimeError("non-empty task inventory count mismatch")
    derived_strategy_totals: Counter[str] = Counter()
    for value in task_inventory.values():
        if not isinstance(value, dict):
            raise RuntimeError("invalid task inventory entry")
        derived_strategy_totals[str(value.get("group") or "")] += int(
            value.get("rows") or 0
        )
    recorded_strategy_totals = dict(first.get("strategy_totals") or {})
    for group in ("strategy_a", "strategy_b", "strategy_c"):
        if int(recorded_strategy_totals.get(group) or 0) != int(
            derived_strategy_totals[group]
        ):
            raise RuntimeError(f"strategy row inventory mismatch: {group}")

    invariant_values = {
        "schema_version": first.get("schema_version"),
        "database": first.get("database"),
        "database_rows": first.get("database_rows"),
        "strategy_totals": first.get("strategy_totals"),
        "baseline_module_sha256": module_shas["baseline"],
        "candidate_module_sha256": module_shas["candidate"],
        "audit_script_sha256": module_shas["audit"],
        "dependency_manifest_file_sha256": dependency_file_sha,
        "dependency_environment": dependency_environment,
        "audit_mode": audit_mode,
        "metadata_snapshot": metadata_summary,
        "audit_scope": first.get("audit_scope"),
        "historical_generation_provenance": first.get(
            "historical_generation_provenance"
        ),
        "task_inventory": task_inventory,
        "task_inventory_digest": task_inventory_digest,
        "task_families": task_families,
    }
    seen_tasks: set[str] = set()
    for part in parts:
        for key, expected in invariant_values.items():
            if part.get(key) != expected:
                raise RuntimeError(
                    f"cross-part invariant mismatch for {key}: {part.get('_artifact')}"
                )
        expected_group = str(part["_expected_group"])
        expected_index = int(part["_expected_partition_index"])
        requested_groups = part.get("requested_groups")
        partition = dict(part.get("partition") or {})
        if requested_groups != [expected_group]:
            raise RuntimeError("part group identity mismatch")
        if (
            int(partition.get("count", -1)) != partitions
            or int(partition.get("index", -1)) != expected_index
        ):
            raise RuntimeError("part partition identity mismatch")
        expected_tasks = {
            str(task_id)
            for task_id, value in task_inventory.items()
            if isinstance(value, dict)
            and str(value.get("group")) == expected_group
            and int(task_id) % partitions == expected_index
        }
        selected_tasks = {
            str(value) for value in partition.get("selected_task_ids") or []
        }
        if selected_tasks != expected_tasks:
            raise RuntimeError(
                f"selected task set mismatch for {expected_group}/p{expected_index}"
            )
        duplicates = seen_tasks.intersection(selected_tasks)
        if duplicates:
            raise RuntimeError(f"tasks scanned by multiple parts: {sorted(duplicates)}")
        seen_tasks.update(selected_tasks)
        expected_counts = {
            task_id: int(dict(task_inventory[task_id]).get("rows") or 0)
            for task_id in expected_tasks
        }
        part_expected_counts = {
            str(key): int(value)
            for key, value in dict(
                partition.get("expected_task_counts") or {}
            ).items()
        }
        scanned_counts = {
            str(key): int(value)
            for key, value in dict(
                partition.get("scanned_task_counts") or {}
            ).items()
        }
        scanned_counts_with_zeroes = {
            task_id: scanned_counts.get(task_id, 0)
            for task_id in expected_tasks
        }
        if part_expected_counts != expected_counts:
            raise RuntimeError("part expected task-count inventory mismatch")
        if scanned_counts_with_zeroes != expected_counts:
            raise RuntimeError("per-task scanned row count mismatch")
        if set(scanned_counts).difference(expected_tasks):
            raise RuntimeError("part scanned unexpected task ids")
        if sum(scanned_counts.values()) != int(
            part.get("structural_rows_scanned") or 0
        ):
            raise RuntimeError("part scanned total disagrees with per-task counts")
        nonnegative_scalar_fields = (
            "structural_rows_scanned",
            "full_baseline_scores",
            "full_candidate_scores",
            "scoring_errors",
            "indeterminate_rows",
            "blocking_timeout_count",
            "judge_input_affected_rows",
            "replay_affected_rows",
        )
        if any(int(part.get(field) or 0) < 0 for field in nonnegative_scalar_fields):
            raise RuntimeError("negative audit counters are invalid")
        transitions = {
            str(key): int(value)
            for key, value in dict(part.get("row_transitions") or {}).items()
        }
        if set(transitions).difference({"0->0", "0->1", "1->0", "1->1"}) or any(
            value < 0 for value in transitions.values()
        ):
            raise RuntimeError("invalid row transition counters")
        if int(part.get("scoring_errors") or 0) == 0 and sum(
            transitions.values()
        ) != int(part.get("structural_rows_scanned") or 0):
            raise RuntimeError("successful row transitions do not cover the full scan")
        order_evidence = dict(part.get("module_order_consistency") or {})
        order_conflicts = [
            value
            for value in (order_evidence.get("conflicts") or [])
            if isinstance(value, dict)
        ]
        if (
            order_evidence.get("probe_processes")
            != "two_independent_processes_per_row"
            or order_evidence.get("orders")
            != ["candidate_then_baseline", "baseline_then_candidate"]
            or int(order_evidence.get("conflict_count") or 0)
            != len(order_conflicts)
        ):
            raise RuntimeError("invalid independent module-order evidence")
        replay_by_task = {
            str(key): int(value)
            for key, value in dict(
                part.get("replay_affected_by_task") or {}
            ).items()
            if int(value) > 0
        }
        replay_task_ids = {
            str(value) for value in part.get("replay_affected_task_ids") or []
        }
        if replay_task_ids != set(replay_by_task):
            raise RuntimeError("replay affected task manifest mismatch")
        if replay_task_ids.difference(task_inventory):
            raise RuntimeError("replay manifest contains tasks outside inventory")
        if int(part.get("judge_input_affected_rows") or 0) > int(
            part.get("replay_affected_rows") or 0
        ):
            raise RuntimeError("Judge affected rows are missing from replay scope")
        changes = [
            value
            for value in (part.get("changes") or [])
            if isinstance(value, dict)
        ]
        if int(part.get("scoring_errors") or 0) == 0 and any(
            change.get("error") for change in changes
        ):
            raise RuntimeError("error evidence conflicts with zero scoring errors")
        expected_judge_rows = sum(
            value.get("judge_input_affected") is True for value in changes
        )
        expected_replay_rows = sum(
            value.get("replay_affected") is True for value in changes
        )
        if expected_judge_rows != int(part.get("judge_input_affected_rows") or 0):
            raise RuntimeError("Judge affected row evidence mismatch")
        if expected_replay_rows != int(part.get("replay_affected_rows") or 0):
            raise RuntimeError("replay affected row evidence mismatch")
        expected_replay_by_task: Counter[str] = Counter()
        for change in changes:
            if change.get("replay_affected") is not True:
                continue
            source_task = str(int(change["task_id"]))
            affected = (
                task_families[source_task]
                if change.get("group") == "strategy_a"
                else [int(source_task)]
            )
            expected_replay_by_task.update(str(value) for value in affected)
        if replay_by_task != dict(expected_replay_by_task):
            raise RuntimeError("replay affected task expansion mismatch")

    if seen_tasks != set(task_inventory):
        raise RuntimeError("global task inventory has missing/unscanned tasks")
    return {
        "database_snapshot_id": snapshot_id,
        "metadata_snapshot_digest": metadata_digest,
        "dataset_digests": dataset_digests,
        "module_shas": module_shas,
        "dependency_environment_sha256": dependency_environment["sha256"],
        "dependency_manifest_file_sha256": dependency_file_sha,
        "task_inventory_digest": task_inventory_digest,
        "task_count": len(task_inventory),
        "input_artifacts": artifact_records,
        "input_artifacts_sha256": artifact_set_sha,
    }


def _merge_fingerprints(
    parts: list[dict[str, Any]], key: str
) -> tuple[dict[str, str], list[dict[str, str]], int]:
    merged: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    duplicates = 0
    for part in parts:
        values = part.get(key)
        if not isinstance(values, dict):
            continue
        for fingerprint, transition in values.items():
            fingerprint = str(fingerprint)
            transition = str(transition)
            previous = merged.get(fingerprint)
            if previous is None:
                merged[fingerprint] = transition
            elif previous == transition:
                duplicates += 1
            else:
                conflicts.append(
                    {
                        "fingerprint": fingerprint,
                        "previous": previous,
                        "current": transition,
                        "artifact": str(part.get("_artifact") or ""),
                    }
                )
    return merged, conflicts, duplicates


def _canonical_counts(
    fingerprints: dict[str, str]
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    totals: Counter[str] = Counter()
    by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    for value in fingerprints.values():
        if "|" in value:
            group, transition = value.rsplit("|", 1)
        else:
            group, transition = "primary_c", value
        totals[transition] += 1
        by_strategy[group][transition] += 1
    return totals, by_strategy


def _table_row(values: Iterable[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def main() -> None:  # noqa: C901, PLR0915
    args = _parse_args()
    merge_script_sha = _bootstrap_or_verify_merge_script(args)
    _assert_output_paths_fresh(args.output_json, args.output_md)
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    parts = _load_parts(args.input_prefix, groups, args.partitions)
    production_provenance = _validate_production_parts(
        parts,
        groups=groups,
        partitions=args.partitions,
    )

    invariant_keys = (
        "database",
        "database_rows",
        "tasks",
        "strategy_totals",
        "primary_c_tasks",
        "baseline_module_sha256",
        "candidate_module_sha256",
        "audit_script_sha256",
        "dependency_manifest_file_sha256",
        "math_fast_integer_match_env",
        "math_fast_integer_match_enabled",
        "sql_answer_cue_regex",
        "audit_mode",
    )
    invariant_conflicts: list[dict[str, Any]] = []
    for key in invariant_keys:
        expected = parts[0].get(key)
        for part in parts[1:]:
            if part.get(key) != expected:
                invariant_conflicts.append(
                    {
                        "key": key,
                        "expected": expected,
                        "actual": part.get(key),
                        "artifact": part["_artifact"],
                    }
                )

    scalar_sum_keys = (
        "structural_rows_scanned",
        "changed_verification_windows",
        "proof_equivalent_rows",
        "judgement_rows",
        "stored_noncomparable_rows",
        "real_scorer_rows",
        "full_candidate_scores",
        "full_baseline_scores",
        "scoring_errors",
        "indeterminate_rows",
        "deterministic_surface_changed_rows",
        "judge_input_affected_rows",
        "replay_affected_rows",
        "stored_reference_drift_rows",
    )
    sums = {
        key: sum(int(part.get(key) or 0) for part in parts)
        for key in scalar_sum_keys
    }
    counter_keys = (
        "proof_equivalent_reasons",
        "proof_equivalent_rows_by_strategy",
        "real_scorer_reasons",
        "timeout_retries",
        "row_transitions",
        "stored_final_transitions",
    )
    counters: dict[str, Counter[str]] = {
        key: Counter() for key in counter_keys
    }
    row_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    stored_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    timeout_by_implementation: dict[str, Counter[str]] = defaultdict(Counter)
    replay_affected_by_task: Counter[str] = Counter()
    replay_affected_reasons: Counter[str] = Counter()
    for part in parts:
        for key in counter_keys:
            values = part.get(key)
            if isinstance(values, dict):
                _counter_add(counters[key], values)
        _nested_counter_add(
            row_by_strategy,
            part.get("row_transitions_by_strategy") or {},
        )
        _nested_counter_add(
            stored_by_strategy,
            part.get("stored_final_transitions_by_strategy") or {},
        )
        _nested_counter_add(
            timeout_by_implementation,
            part.get("timeout_retries_by_implementation") or {},
        )
        _counter_add(
            replay_affected_by_task,
            part.get("replay_affected_by_task") or {},
        )
        _counter_add(
            replay_affected_reasons,
            part.get("replay_affected_reasons") or {},
        )

    canonical, canonical_conflicts, canonical_duplicates = _merge_fingerprints(
        parts, "canonical_fingerprints"
    )
    primary_c, primary_c_conflicts, primary_c_duplicates = _merge_fingerprints(
        parts, "primary_c_canonical_fingerprints"
    )
    canonical_transitions, canonical_by_strategy = _canonical_counts(canonical)
    primary_c_transitions, _ = _canonical_counts(primary_c)
    canonical_by_strategy["primary_c"] = primary_c_transitions

    changes = [
        dict(change)
        for part in parts
        for change in (part.get("changes") or [])
        if isinstance(change, dict)
    ]
    one_to_zero = [
        change for change in changes if change.get("transition") == "1->0"
    ]
    unexplained = [
        change
        for change in one_to_zero
        if change.get("explanation") in {None, "", "UNEXPLAINED"}
    ]
    explanations = Counter(
        str(change.get("explanation") or "UNEXPLAINED")
        for change in one_to_zero
    )
    cell_deltas = [
        dict(cell)
        for part in parts
        for cell in (part.get("cell_deltas") or [])
        if isinstance(cell, dict)
    ]
    cell_deltas.sort(
        key=lambda item: (
            -abs(float(item.get("candidate_minus_baseline_pp") or 0.0)),
            int(item.get("task_id") or 0),
        )
    )

    main_scanned_by_group: Counter[str] = Counter()
    a_complement_scanned = 0
    a_superset_violations: list[dict[str, Any]] = []
    a_proof_exhaustive = True
    for part in parts:
        requested = part.get("requested_groups") or []
        if len(requested) == 1:
            main_scanned_by_group[str(requested[0])] += int(
                part.get("structural_rows_scanned") or 0
            )
        proof = part.get("a_sql_prefilter_superset_proof") or {}
        if "strategy_a" in requested:
            a_complement_scanned += int(
                proof.get("complement_rows_scanned") or 0
            )
            a_proof_exhaustive = a_proof_exhaustive and bool(
                proof.get("exhaustive")
            )
            a_superset_violations.extend(proof.get("violations") or [])

    strategy_totals = dict(parts[0].get("strategy_totals") or {})
    audit_mode = dict(parts[0].get("audit_mode") or {})
    full_scan_a = bool(audit_mode.get("full_scan_a"))
    full_real_scorer = bool(audit_mode.get("full_real_scorer"))
    scan_proof = {
        "strategy_a": {
            "expected_rows": int(strategy_totals.get("strategy_a") or 0),
            "selection_mode": (
                "all_rows" if full_scan_a else "sql_answer_cue_prefilter"
            ),
            "prefilter_rows": main_scanned_by_group["strategy_a"],
            "complement_rows": a_complement_scanned,
            "union_rows": (
                main_scanned_by_group["strategy_a"] + a_complement_scanned
            ),
            "exhaustive": a_proof_exhaustive,
            "changed_windows_outside_prefilter": len(a_superset_violations),
        },
        "strategy_b": {
            "expected_rows": int(strategy_totals.get("strategy_b") or 0),
            "scanned_rows": main_scanned_by_group["strategy_b"],
        },
        "strategy_c": {
            "expected_rows": int(strategy_totals.get("strategy_c") or 0),
            "scanned_rows": main_scanned_by_group["strategy_c"],
        },
    }
    if full_scan_a:
        a_gate = (
            scan_proof["strategy_a"]["prefilter_rows"]
            == scan_proof["strategy_a"]["expected_rows"]
        )
    else:
        a_gate = (
            scan_proof["strategy_a"]["exhaustive"]
            and scan_proof["strategy_a"]["union_rows"]
            == scan_proof["strategy_a"]["expected_rows"]
            and not a_superset_violations
        )
    b_gate = (
        scan_proof["strategy_b"]["scanned_rows"]
        == scan_proof["strategy_b"]["expected_rows"]
    )
    c_gate = (
        scan_proof["strategy_c"]["scanned_rows"]
        == scan_proof["strategy_c"]["expected_rows"]
    )
    blocking_timeouts = sum(
        int(part.get("blocking_timeout_count") or 0) for part in parts
    )
    indeterminate_rows = sums["indeterminate_rows"]
    order_probe_expected = sum(
        min(
            int(dict(part.get("audit_mode") or {}).get(
                "order_consistency_probe_rows"
            ) or 0),
            int(part.get("structural_rows_scanned") or 0),
        )
        for part in parts
    )
    order_probe_actual = sum(
        int(
            dict(part.get("module_order_consistency") or {}).get(
                "probed_rows"
            )
            or 0
        )
        for part in parts
    )
    order_probe_conflicts = [
        dict(conflict)
        for part in parts
        for conflict in (
            dict(part.get("module_order_consistency") or {}).get(
                "conflicts"
            )
            or []
        )
        if isinstance(conflict, dict)
    ]
    order_probe_timeout_events: Counter[str] = Counter()
    for part in parts:
        _counter_add(
            order_probe_timeout_events,
            dict(part.get("module_order_consistency") or {}).get(
                "timeout_events"
            )
            or {},
        )
    gate = {
        "production_snapshot_and_inventory_valid": True,
        "invariants_consistent": not invariant_conflicts,
        "canonical_fingerprints_consistent": not (
            canonical_conflicts or primary_c_conflicts
        ),
        "a_sql_prefilter_is_proven_superset": a_gate,
        "strategy_b_full_scan": b_gate,
        "strategy_c_full_scan": c_gate,
        "no_scoring_errors": sums["scoring_errors"] == 0,
        "no_timeout_events": counters["timeout_retries"].get(
            "initial_timeout", 0
        )
        == 0,
        "no_blocking_timeouts": blocking_timeouts == 0,
        "no_indeterminate_rows": indeterminate_rows == 0,
        "no_baseline_one_to_zero": (
            counters["row_transitions"].get("1->0", 0) == 0
            and not one_to_zero
        ),
        "no_judge_input_affected_rows": sums["judge_input_affected_rows"] == 0,
        "no_replay_affected_rows": sums["replay_affected_rows"] == 0,
        "module_order_probe_complete": order_probe_actual
        == order_probe_expected,
        "module_order_consistent": not order_probe_conflicts,
        "module_order_probe_has_no_timeouts": (
            order_probe_timeout_events.get("initial_timeout", 0) == 0
        ),
        "no_unexplained_one_to_zero": not unexplained,
    }
    if full_real_scorer:
        gate["every_row_scored_by_baseline"] = (
            sums["full_baseline_scores"] == sums["structural_rows_scanned"]
        )
        gate["every_row_scored_by_candidate"] = (
            sums["full_candidate_scores"] == sums["structural_rows_scanned"]
        )
    gate["passed"] = all(gate.values())

    key_examples = [
        change
        for change in changes
        if change.get("transition") == "0->1"
    ][:5]
    output = {
        "database": parts[0].get("database"),
        "database_rows": parts[0].get("database_rows"),
        "tasks": parts[0].get("tasks"),
        "strategy_totals": strategy_totals,
        "primary_c_tasks": parts[0].get("primary_c_tasks"),
        "baseline_module_sha256": parts[0].get("baseline_module_sha256"),
        "candidate_module_sha256": parts[0].get("candidate_module_sha256"),
        "audit_script_sha256": parts[0].get("audit_script_sha256"),
        "merge_script_sha256": merge_script_sha,
        "dependency_manifest_file_sha256": parts[0].get(
            "dependency_manifest_file_sha256"
        ),
        "dependency_environment": parts[0].get("dependency_environment"),
        "audit_mode": audit_mode,
        "audit_scope": parts[0].get("audit_scope"),
        "historical_generation_provenance": parts[0].get(
            "historical_generation_provenance"
        ),
        "production_provenance": production_provenance,
        "module_order_consistency": {
            "expected_probed_rows": order_probe_expected,
            "actual_probed_rows": order_probe_actual,
            "conflict_count": len(order_probe_conflicts),
            "conflicts": order_probe_conflicts,
            "timeout_events": dict(order_probe_timeout_events),
        },
        "partitions": args.partitions,
        "groups": groups,
        "input_artifacts": production_provenance["input_artifacts"],
        "input_artifacts_sha256": production_provenance[
            "input_artifacts_sha256"
        ],
        "partition_count": len(parts),
        "totals": sums,
        "total_rows_examined_including_a_complement": (
            sums["structural_rows_scanned"] + a_complement_scanned
        ),
        "proof_equivalent_reasons": dict(
            counters["proof_equivalent_reasons"]
        ),
        "proof_equivalent_rows_by_strategy": dict(
            counters["proof_equivalent_rows_by_strategy"]
        ),
        "real_scorer_reasons": dict(counters["real_scorer_reasons"]),
        "timeout_retries": dict(counters["timeout_retries"]),
        "timeout_retries_by_implementation": {
            implementation: dict(stats)
            for implementation, stats in sorted(
                timeout_by_implementation.items()
            )
        },
        "blocking_timeout_count": blocking_timeouts,
        "indeterminate_rows": indeterminate_rows,
        "replay_affected_rows": sums["replay_affected_rows"],
        "replay_affected_by_task": dict(replay_affected_by_task),
        "replay_affected_task_ids": sorted(
            (int(value) for value in replay_affected_by_task),
        ),
        "replay_affected_reasons": dict(replay_affected_reasons),
        "row_transitions": dict(counters["row_transitions"]),
        "stored_final_transitions": dict(
            counters["stored_final_transitions"]
        ),
        "row_transitions_by_strategy": {
            key: dict(value) for key, value in sorted(row_by_strategy.items())
        },
        "stored_final_transitions_by_strategy": {
            key: dict(value)
            for key, value in sorted(stored_by_strategy.items())
        },
        "canonical_changed_payloads": len(canonical),
        "canonical_duplicate_fingerprints": canonical_duplicates,
        "canonical_transitions": dict(canonical_transitions),
        "canonical_transitions_by_strategy": {
            key: dict(value)
            for key, value in sorted(canonical_by_strategy.items())
        },
        "primary_c_canonical_payloads": len(primary_c),
        "primary_c_duplicate_fingerprints": primary_c_duplicates,
        "scan_proof": scan_proof,
        "a_superset_violations": a_superset_violations,
        "invariant_conflicts": invariant_conflicts,
        "canonical_conflicts": canonical_conflicts,
        "primary_c_canonical_conflicts": primary_c_conflicts,
        "one_to_zero": len(one_to_zero),
        "unexplained_one_to_zero": len(unexplained),
        "one_to_zero_explanations": dict(explanations),
        "one_to_zero_rows": one_to_zero,
        "key_examples": key_examples,
        "cell_deltas": cell_deltas,
        "changes": changes,
        "gate": gate,
    }
    lines = [
        "# Free-response extractor global replay audit",
        "",
        f"- Database: `{output['database']}`",
        f"- Candidate SHA256: `{output['candidate_module_sha256']}`",
        f"- Audit mode: `{json.dumps(audit_mode, sort_keys=True)}`",
        f"- Rows in inventory: {output['database_rows']:,}",
        f"- Rows examined (including A complement): "
        f"{output['total_rows_examined_including_a_complement']:,}",
        f"- Canonical changed payloads: {len(canonical):,}",
        f"- Deduplicated canonical repeats: {canonical_duplicates:,}",
        f"- Gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "## Scan proof",
        "",
        _table_row(("Strategy", "Expected", "Main", "A complement", "Complete")),
        _table_row(("---", "---:", "---:", "---:", "---")),
        _table_row(
            (
                "A",
                scan_proof["strategy_a"]["expected_rows"],
                scan_proof["strategy_a"]["prefilter_rows"],
                scan_proof["strategy_a"]["complement_rows"],
                a_gate,
            )
        ),
        _table_row(
            (
                "B",
                scan_proof["strategy_b"]["expected_rows"],
                scan_proof["strategy_b"]["scanned_rows"],
                "-",
                b_gate,
            )
        ),
        _table_row(
            (
                "C",
                scan_proof["strategy_c"]["expected_rows"],
                scan_proof["strategy_c"]["scanned_rows"],
                "-",
                c_gate,
            )
        ),
        "",
        "## Transitions",
        "",
        _table_row(("Transition", "Rows", "Canonical")),
        _table_row(("---", "---:", "---:")),
    ]
    for transition in ("0->0", "0->1", "1->1", "1->0"):
        lines.append(
            _table_row(
                (
                    transition,
                    counters["row_transitions"].get(transition, 0),
                    canonical_transitions.get(transition, 0),
                )
            )
        )
    lines.extend(
        [
            "",
            "## Blocking checks",
            "",
            *[f"- {key}: {value}" for key, value in gate.items()],
            "",
            "## Every baseline 1->0 explanation",
            "",
        ]
    )
    if not one_to_zero:
        lines.append("None.")
    else:
        lines.extend(
            [
                _table_row(
                    (
                        "Task",
                        "Completion",
                        "Strategy",
                        "Benchmark",
                        "Explanation",
                    )
                ),
                _table_row(("---:", "---:", "---", "---", "---")),
            ]
        )
        for change in one_to_zero:
            lines.append(
                _table_row(
                    (
                        change.get("task_id"),
                        change.get("completion_id"),
                        change.get("group"),
                        change.get("benchmark"),
                        change.get("explanation") or "UNEXPLAINED",
                    )
                )
            )
    json_payload = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    markdown_payload = ("\n".join(lines) + "\n").encode("utf-8")
    _, end_merge_script_sha = _read_immutable_artifact(
        Path(__file__),
        label="merge script",
    )
    if end_merge_script_sha != merge_script_sha:
        raise RuntimeError("frozen merge script changed during merge")
    if gate["passed"]:
        json_target = args.output_json
        markdown_target = args.output_md
    else:
        json_target = _artifact_path(args.output_json, "failed")
        markdown_target = _artifact_path(args.output_md, "failed")
    _atomic_publish_bytes(json_target, json_payload)
    _atomic_publish_bytes(markdown_target, markdown_payload)
    if gate["passed"]:
        acceptance = {
            "schema_version": "free-response-global-audit-acceptance.v1",
            "accepted": True,
            "gate_passed": True,
            "merge_script_sha256": merge_script_sha,
            "json": {
                "path": str(json_target.resolve()),
                "sha256": hashlib.sha256(json_payload).hexdigest(),
            },
            "markdown": {
                "path": str(markdown_target.resolve()),
                "sha256": hashlib.sha256(markdown_payload).hexdigest(),
            },
            "production_provenance": production_provenance,
        }
        _atomic_publish_bytes(
            _acceptance_manifest_path(args.output_json),
            json.dumps(
                acceptance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
    print(json.dumps({"gate": gate, "output": str(json_target)}))
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
