"""Launch partitioned audits under one exported PostgreSQL snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple
from uuid import uuid4

STRATEGIES = ("strategy_a", "strategy_b", "strategy_c")
INVENTORY_DIAGNOSTIC_TAIL_BYTES = 32 * 1024
_SENSITIVE_ENV_KEY_RE = re.compile(
    r"(?i)(?:password|passwd|token|secret|api[_-]?key|credential)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:pg_)?(?:password|passwd|token|secret|api[_-]?key|credential)"
    r"\b\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_CREDENTIAL_RE = re.compile(r"(://[^:/\s]+:)[^@\s]+(@)")

# Importing psycopg before the frozen dependency manifest exists would execute
# mutable site-packages bytes that the launcher has not yet authenticated.
# Keep the names patchable for unit tests, but load production psycopg only
# after the dependency artifact has been published.
psycopg: Any = None
dict_row: Any = None


def _load_psycopg() -> None:
    global dict_row  # noqa: PLW0603
    global psycopg  # noqa: PLW0603

    if psycopg is not None:
        return
    psycopg = importlib.import_module("psycopg")
    dict_row = importlib.import_module("psycopg.rows").dict_row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--candidate-module", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument(
        "--env",
        type=Path,
        default=Path("/home/rwkv/chase/rwkv-skills/.env"),
    )
    parser.add_argument("--database", default="chase_rwkv_skills")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Maximum number of audit shard processes resident at once. "
            "All shards still share one exported PostgreSQL snapshot."
        ),
    )
    parser.add_argument(
        "--groups",
        default=",".join(STRATEGIES),
    )
    parser.add_argument("--max-structural-rows-per-part", type=int)
    parser.add_argument("--prove-a-superset", action="store_true")
    parser.add_argument("--max-a-superset-proof-rows-per-part", type=int)
    return parser.parse_args()


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _connection_string(env: dict[str, str], database: str) -> str:
    parts = [
        f"host={env.get('PG_HOST', '127.0.0.1')}",
        f"port={env.get('PG_PORT', '5432')}",
        f"user={env.get('PG_USER', 'postgres')}",
        f"dbname={database}",
        f"sslmode={env.get('PG_SSLMODE', 'prefer')}",
    ]
    if env.get("PG_PASSWORD"):
        parts.append(f"password={env['PG_PASSWORD']}")
    return " ".join(parts)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _snapshot_digest(document: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in document.items()
        if key != "snapshot_digest"
    }
    # JSON object keys are strings on disk.  Normalize through the exact JSON
    # data model before sorting so integer task IDs do not hash in numeric order
    # before publication and lexicographic order after a read-back.
    json_payload = json.loads(
        json.dumps(payload, ensure_ascii=False, default=str)
    )
    return hashlib.sha256(_canonical_json_bytes(json_payload)).hexdigest()


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_file_tail(path: Path, *, limit: int) -> str:
    if limit < 1:
        raise ValueError("diagnostic tail limit must be positive")
    flags = os.O_RDONLY
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        size = os.fstat(descriptor).st_size
        truncated = size > limit
        if truncated:
            os.lseek(descriptor, size - limit, os.SEEK_SET)
        payload = os.read(descriptor, limit)
    finally:
        os.close(descriptor)
    text = payload.decode("utf-8", errors="replace")
    return ("[earlier stderr truncated]\n" if truncated else "") + text


def _redact_diagnostic(text: str, configured_env: dict[str, str]) -> str:
    redacted = text
    secrets = sorted(
        {
            value
            for key, value in configured_env.items()
            if len(value) >= 4 and _SENSITIVE_ENV_KEY_RE.search(key)
        },
        key=len,
        reverse=True,
    )
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    redacted = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1<redacted>", redacted)
    return _URL_CREDENTIAL_RE.sub(r"\1<redacted>\2", redacted)


def _atomic_publish_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_inventory_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    configured_env: dict[str, str],
    diagnostic_path: Path,
) -> None:
    """Run inventory without inheriting a caller's terminal or journal pipe."""

    if diagnostic_path.exists():
        raise FileExistsError(
            f"refusing existing inventory diagnostic: {diagnostic_path}"
        )
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        diagnostic_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    launch_error: OSError | None = None
    returncode: int | None = None
    try:
        with os.fdopen(descriptor, "wb") as stderr:
            try:
                completed = subprocess.run(  # noqa: S603
                    command,
                    check=False,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
                returncode = int(completed.returncode)
            except OSError as exc:
                launch_error = exc
    finally:
        diagnostic = _redact_diagnostic(
            _bounded_file_tail(
                diagnostic_path,
                limit=INVENTORY_DIAGNOSTIC_TAIL_BYTES,
            ),
            configured_env,
        )
        encoded = diagnostic.encode("utf-8")
        if len(encoded) > INVENTORY_DIAGNOSTIC_TAIL_BYTES:
            diagnostic = encoded[-INVENTORY_DIAGNOSTIC_TAIL_BYTES :].decode(
                "utf-8",
                errors="replace",
            )
        _atomic_publish_text(diagnostic_path, diagnostic)

    diagnostic_label = diagnostic.strip() or "<empty stderr>"
    if launch_error is not None:
        raise RuntimeError(
            "inventory subprocess could not start "
            f"({type(launch_error).__name__}); "
            f"sanitized diagnostic: {diagnostic_path}\n{diagnostic_label}"
        ) from launch_error
    if returncode != 0:
        raise RuntimeError(
            f"inventory subprocess failed with exit code {returncode}; "
            f"sanitized diagnostic: {diagnostic_path}\n{diagnostic_label}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_file(source: Path, snapshot_root: Path) -> tuple[Path, str]:
    """Publish exactly one read of *source* under its content digest."""

    payload = source.resolve().read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = snapshot_root / f"{digest}{source.suffix}"
    if target.exists():
        if _sha256_file(target) != digest or target.stat().st_mode & 0o222:
            raise RuntimeError(f"invalid existing frozen code artifact: {target}")
        return target.resolve(), digest
    temporary = snapshot_root / f".{target.name}.tmp.{os.getpid()}.{uuid4().hex}"
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
        _fsync_directory(snapshot_root)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256_file(target) != digest or target.stat().st_mode & 0o222:
        raise RuntimeError(f"failed to freeze code artifact: {target}")
    return target.resolve(), digest


def _find_project_root(source: Path) -> Path:
    """Resolve the repository root without depending on the caller's cwd."""

    resolved = source.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src"
        ).is_dir():
            return candidate
    raise RuntimeError(f"cannot locate project root from audit script: {source}")


def _freeze_project_contract(
    project_root: Path,
    snapshot_root: Path,
) -> tuple[Path, str, list[dict[str, Any]]]:
    """Freeze every local Python module imported through ``src``.

    Files are read once into a private staging tree.  The tree name is the
    digest of the exact bytes written, not of a later re-read of the mutable
    worktree.  Workers only receive this read-only tree on ``PYTHONPATH``.
    """

    project_root = project_root.resolve()
    src_root = project_root / "src"
    src_entries = sorted(src_root.rglob("*"))
    symlinks = [path for path in src_entries if path.is_symlink()]
    if symlinks:
        raise RuntimeError(
            "project source contract refuses symlinks: "
            + ", ".join(str(path) for path in symlinks[:10])
        )
    unsupported = [
        path
        for path in src_entries
        if not path.is_dir() and not stat.S_ISREG(path.stat().st_mode)
    ]
    if unsupported:
        raise RuntimeError(
            "project source contract refuses special files: "
            + ", ".join(str(path) for path in unsupported[:10])
        )
    sources = [
        path
        for path in src_entries
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    sources.extend(
        path
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.is_file()
    )
    if not sources:
        raise RuntimeError(f"project source contract is empty: {project_root}")
    snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = snapshot_root / f".project.tmp.{os.getpid()}.{uuid4().hex}"
    staging.mkdir(mode=0o700)
    records: list[dict[str, Any]] = []
    try:
        for source in sources:
            relative = source.resolve().relative_to(project_root).as_posix()
            payload = source.resolve().read_bytes()
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(0o444)
            records.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        records.sort(key=lambda value: str(value["path"]))
        tree_digest = hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
        target_root = snapshot_root / f"project-{tree_digest}"
        if target_root.exists():
            raise FileExistsError(
                f"refusing existing project contract snapshot: {target_root}"
            )
        for directory in sorted(
            (value for value in staging.rglob("*") if value.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        staging.chmod(0o555)
        staging.rename(target_root)
        _fsync_directory(snapshot_root)
        return target_root.resolve(), tree_digest, records
    finally:
        if staging.exists():
            # An error can occur after modes were locked but before rename.
            staging.chmod(0o700)
            for directory in (
                value for value in staging.rglob("*") if value.is_dir()
            ):
                directory.chmod(0o700)
            for path in sorted(
                staging.rglob("*"),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def _child_environment(
    frozen_project_root: Path,
    configured: dict[str, str],
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(configured)
    environment["PYTHONPATH"] = str(frozen_project_root.resolve())
    environment["RWKV_AUDIT_FROZEN_PROJECT_ROOT"] = str(
        frozen_project_root.resolve()
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _verify_artifact_matches_project_contract(
    source: Path,
    digest: str,
    *,
    project_root: Path,
    records: list[dict[str, Any]],
    label: str,
) -> None:
    try:
        relative = source.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return
    if not relative.startswith("src/"):
        return
    expected = {
        str(value["path"]): str(value["sha256"])
        for value in records
    }.get(relative)
    if expected is None or expected != digest:
        raise RuntimeError(
            f"{label} changed while the frozen project contract was captured"
        )


def _assert_fresh_prefix(
    output_prefix: Path,
    status_path: Path,
    metadata_cache: Path,
) -> None:
    stale = sorted(output_prefix.parent.glob(f"{output_prefix.name}_*"))
    for path in (status_path, metadata_cache):
        if path.exists() and path not in stale:
            stale.append(path)
    if stale:
        raise FileExistsError(
            "refusing stale audit prefix; choose a new output prefix: "
            + ", ".join(str(path) for path in stale)
        )


def _terminate_all(processes: list[subprocess.Popen[bytes]]) -> None:
    def send_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        try:
            if process.poll() is not None:
                return
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            # The process may exit between poll() and signalling.  Continue so
            # one race cannot prevent the remaining workers from being reaped.
            return

    if os.name != "posix":
        for process in processes:
            send_process(process, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                send_process(process, signal.SIGKILL)
        for process in processes:
            try:
                if process.poll() is None:
                    process.wait()
            except (ChildProcessError, ProcessLookupError):
                continue
        return

    # Every worker is created with start_new_session=True, hence PID == PGID.
    # Track the group independently from the leader: a worker can exit after
    # spawning a descendant that ignores SIGTERM.  Looking only at poll()
    # would leak that descendant forever.
    group_ids = sorted({int(process.pid) for process in processes})

    def group_exists(group_id: int) -> bool:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def signal_group(group_id: int, sig: signal.Signals) -> None:
        try:
            os.killpg(group_id, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return

    for group_id in group_ids:
        signal_group(group_id, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        if not any(group_exists(group_id) for group_id in group_ids):
            break
        time.sleep(0.05)
    for group_id in group_ids:
        if group_exists(group_id):
            signal_group(group_id, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, ChildProcessError, ProcessLookupError):
            continue


class _WorkerSpec(NamedTuple):
    group: str
    index: int
    command: list[str]
    stdout_path: Path
    stderr_path: Path


class _WorkerHandle(NamedTuple):
    spec: _WorkerSpec
    process: subprocess.Popen[bytes]


def _start_worker(
    spec: _WorkerSpec,
    *,
    cwd: Path,
    env: dict[str, str],
) -> _WorkerHandle:
    """Start one shard without retaining its output in launcher memory."""

    with spec.stdout_path.open("wb") as stdout, spec.stderr_path.open(
        "wb"
    ) as stderr:
        process = subprocess.Popen(  # noqa: S603
            spec.command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            cwd=cwd,
            env=env,
        )
    return _WorkerHandle(spec=spec, process=process)


def _worker_status_records(
    specs: list[_WorkerSpec],
    handles: list[_WorkerHandle],
    returncodes: dict[tuple[str, int], int | None],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    handles_by_key = {
        (handle.spec.group, handle.spec.index): handle for handle in handles
    }
    records: list[dict[str, Any]] = []
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for spec in specs:
        key = (spec.group, spec.index)
        handle = handles_by_key.get(key)
        if handle is None:
            state = "pending"
            pid = None
            returncode = None
        else:
            pid = handle.process.pid
            returncode = returncodes.get(key)
            if returncode is None:
                state = "running"
            elif returncode == 0:
                state = "completed"
            else:
                state = "failed"
        counts[state] += 1
        records.append(
            {
                "group": spec.group,
                "index": spec.index,
                "state": state,
                "pid": pid,
                "returncode": returncode,
                "stdout": str(spec.stdout_path),
                "stderr": str(spec.stderr_path),
            }
        )
    return records, counts


def _run_bounded_workers(
    specs: list[_WorkerSpec],
    *,
    max_workers: int,
    cwd: Path,
    env: dict[str, str],
    publish_status: Callable[[list[dict[str, Any]], dict[str, int]], None],
    poll_interval_seconds: float = 5.0,
) -> list[_WorkerHandle]:
    """Run every shard while bounding aggregate resident worker memory.

    The caller owns the exported PostgreSQL snapshot and does not return from
    this function until every shard has exited.  Completed shards stay on
    disk, while a failure terminates and reaps all still-running process
    groups.  No shard is silently skipped.
    """

    if max_workers < 1:
        raise ValueError("max-workers must be positive")
    if poll_interval_seconds < 0:
        raise ValueError("poll interval cannot be negative")

    pending = list(specs)
    handles: list[_WorkerHandle] = []
    returncodes: dict[tuple[str, int], int | None] = {}
    try:
        while pending or any(value is None for value in returncodes.values()):
            running = sum(value is None for value in returncodes.values())
            while pending and running < max_workers:
                spec = pending.pop(0)
                handle = _start_worker(spec, cwd=cwd, env=env)
                handles.append(handle)
                returncodes[(spec.group, spec.index)] = None
                running += 1

            for handle in handles:
                key = (handle.spec.group, handle.spec.index)
                if returncodes[key] is None:
                    returncodes[key] = handle.process.poll()

            records, counts = _worker_status_records(
                specs, handles, returncodes
            )
            publish_status(records, counts)

            failed = [
                handle
                for handle in handles
                if returncodes[(handle.spec.group, handle.spec.index)]
                not in {None, 0}
            ]
            if failed:
                labels = ", ".join(
                    f"{handle.spec.group}/p{handle.spec.index}:"
                    f"{returncodes[(handle.spec.group, handle.spec.index)]}"
                    for handle in failed
                )
                raise RuntimeError(
                    "audit worker failed; all remaining workers were "
                    f"terminated ({labels})"
                )

            if not pending and counts["running"] == 0:
                break
            # Refill immediately after fast workers finish; otherwise avoid a
            # busy loop while the resident set is at its configured bound.
            if pending and counts["running"] < max_workers:
                continue
            time.sleep(poll_interval_seconds)
    except BaseException:
        _terminate_all([handle.process for handle in handles])
        raise
    return handles


def _metadata_snapshot(path: Path, snapshot_id: str) -> tuple[dict[str, Any], str]:
    if path.stat().st_mode & 0o222:
        raise RuntimeError("metadata snapshot is not read-only")
    document = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(document.get("snapshot_digest") or "")
    actual = _snapshot_digest(document)
    if not claimed or claimed != actual:
        raise RuntimeError(f"metadata snapshot digest mismatch: {claimed} != {actual}")
    identity = dict(document.get("database_identity") or {})
    if identity.get("exported_snapshot_id") != snapshot_id:
        raise RuntimeError("metadata snapshot did not import the exporter snapshot")
    return document, claimed


def main() -> None:  # noqa: C901
    args = _parse_args()
    if args.partitions < 1:
        raise ValueError("partitions must be positive")
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("groups must be a non-empty unique list")
    unsupported = set(groups).difference(STRATEGIES)
    if unsupported:
        raise ValueError(f"unsupported strategy groups: {sorted(unsupported)}")

    audit_source = args.audit_script.resolve()
    baseline_source = args.baseline_module.resolve()
    candidate_source = args.candidate_module.resolve()
    env_path = args.env.resolve()
    output_prefix = args.output_prefix.resolve()
    status_path = args.status.resolve()
    metadata_cache = (args.metadata_cache.resolve() if args.metadata_cache else Path(
        f"{args.output_prefix}_metadata.json"
    ).resolve())
    dataset_snapshot_root = Path(f"{output_prefix}_datasets")
    code_snapshot_root = Path(f"{output_prefix}_code")
    _assert_fresh_prefix(output_prefix, status_path, metadata_cache)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = _find_project_root(audit_source)
    configured_env = _load_env(env_path)
    frozen_project_root, source_contract_sha, source_contract_files = (
        _freeze_project_contract(project_root, code_snapshot_root)
    )
    artifact_snapshot_root = code_snapshot_root / "artifacts"
    frozen_audit, audit_sha = _freeze_file(audit_source, artifact_snapshot_root)
    frozen_baseline, baseline_sha = _freeze_file(
        baseline_source, artifact_snapshot_root
    )
    frozen_candidate, candidate_sha = _freeze_file(
        candidate_source, artifact_snapshot_root
    )
    _verify_artifact_matches_project_contract(
        baseline_source,
        baseline_sha,
        project_root=project_root,
        records=source_contract_files,
        label="baseline module",
    )
    _verify_artifact_matches_project_contract(
        candidate_source,
        candidate_sha,
        project_root=project_root,
        records=source_contract_files,
        label="candidate module",
    )
    artifact_snapshot_root.chmod(0o555)
    child_env = _child_environment(frozen_project_root, configured_env)
    dependency_build = output_prefix.parent / (
        f".{output_prefix.name}.dependency.{os.getpid()}.{uuid4().hex}.json"
    )
    dependency_command = [
        sys.executable,
        str(frozen_audit),
        "--baseline-module",
        str(frozen_baseline),
        "--candidate-module",
        str(frozen_candidate),
        "--output",
        "/dev/null",
        "--expected-audit-script-sha256",
        audit_sha,
        "--expected-baseline-module-sha256",
        baseline_sha,
        "--expected-candidate-module-sha256",
        candidate_sha,
        "--emit-dependency-manifest",
        str(dependency_build),
    ]
    try:
        subprocess.run(  # noqa: S603
            dependency_command,
            check=True,
            cwd=frozen_project_root,
            env=child_env,
        )
        dependency_snapshot_root = code_snapshot_root / "dependencies"
        frozen_dependency, dependency_file_sha = _freeze_file(
            dependency_build, dependency_snapshot_root
        )
        dependency_snapshot_root.chmod(0o555)
    finally:
        if dependency_build.exists():
            dependency_build.unlink()
    code_snapshot_root.chmod(0o555)

    provenance_args = [
        "--expected-audit-script-sha256",
        audit_sha,
        "--expected-baseline-module-sha256",
        baseline_sha,
        "--expected-candidate-module-sha256",
        candidate_sha,
        "--dependency-manifest",
        str(frozen_dependency),
        "--expected-dependency-manifest-sha256",
        dependency_file_sha,
    ]

    _load_psycopg()
    exporter = psycopg.connect(
        _connection_string(configured_env, args.database),
        row_factory=dict_row,
    )
    started = time.time()
    try:
        exporter.execute(
            "set transaction isolation level repeatable read, read only"
        )
        snapshot_row = exporter.execute(
            "select pg_export_snapshot() as snapshot_id"
        ).fetchone()
        snapshot_id = str(snapshot_row["snapshot_id"])

        inventory_command = [
            sys.executable,
            str(frozen_audit),
            "--env",
            str(env_path),
            "--database",
            args.database,
            "--baseline-module",
            str(frozen_baseline),
            "--candidate-module",
            str(frozen_candidate),
            "--output",
            "/dev/null",
            "--metadata-cache",
            str(metadata_cache),
            "--dataset-snapshot-root",
            str(dataset_snapshot_root),
            "--dataset-source-root",
            str((project_root / "data").resolve()),
            "--refresh-metadata-snapshot",
            "--database-snapshot-id",
            snapshot_id,
            "--inventory-only",
            *provenance_args,
        ]
        inventory_diagnostic = Path(
            f"{output_prefix}_inventory.stderr.txt"
        )
        _run_inventory_command(
            inventory_command,
            cwd=frozen_project_root,
            env=child_env,
            configured_env=configured_env,
            diagnostic_path=inventory_diagnostic,
        )
        metadata_document, metadata_digest = _metadata_snapshot(
            metadata_cache,
            snapshot_id,
        )

        worker_specs: list[_WorkerSpec] = []
        for group in groups:
            for index in range(args.partitions):
                stem = f"{output_prefix}_{group}_p{index}"
                output = Path(f"{stem}.json")
                stdout_path = Path(f"{stem}.out")
                stderr_path = Path(f"{stem}.err")
                command = [
                    sys.executable,
                    str(frozen_audit),
                    "--env",
                    str(env_path),
                    "--database",
                    args.database,
                    "--baseline-module",
                    str(frozen_baseline),
                    "--candidate-module",
                    str(frozen_candidate),
                    "--output",
                    str(output),
                    "--groups",
                    group,
                    "--partitions",
                    str(args.partitions),
                    "--partition-index",
                    str(index),
                    "--metadata-cache",
                    str(metadata_cache),
                    "--dataset-source-root",
                    str((project_root / "data").resolve()),
                    "--metadata-snapshot-digest",
                    metadata_digest,
                    "--database-snapshot-id",
                    snapshot_id,
                    "--progress-every",
                    "1000",
                    "--full-scan-a",
                    "--full-real-scorer",
                    *provenance_args,
                ]
                if args.max_structural_rows_per_part is not None:
                    command.extend(
                        [
                            "--max-structural-rows",
                            str(args.max_structural_rows_per_part),
                        ]
                    )
                if group == "strategy_a" and args.prove_a_superset:
                    command.append("--prove-a-superset")
                    if args.max_a_superset_proof_rows_per_part is not None:
                        command.extend(
                            [
                                "--max-a-superset-proof-rows",
                                str(args.max_a_superset_proof_rows_per_part),
                            ]
                        )
                worker_specs.append(
                    _WorkerSpec(
                        group=group,
                        index=index,
                        command=command,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                )

        def publish_status(
            part_records: list[dict[str, Any]],
            part_counts: dict[str, int],
        ) -> None:
            status = {
                "started_epoch": started,
                "updated_epoch": time.time(),
                "database": args.database,
                "database_snapshot_id": snapshot_id,
                "metadata_snapshot_digest": metadata_digest,
                "dataset_digests": {
                    key: {
                        "file_sha256": value.get("file_sha256"),
                        "records_sha256": value.get("records_sha256"),
                    }
                    for key, value in dict(
                        metadata_document.get("dataset_sources") or {}
                    ).items()
                    if isinstance(value, dict)
                },
                "code_provenance": {
                    "audit_script_sha256": audit_sha,
                    "baseline_module_sha256": baseline_sha,
                    "candidate_module_sha256": candidate_sha,
                    "dependency_manifest_file_sha256": dependency_file_sha,
                    "source_contract_sha256": source_contract_sha,
                    "source_contract_files": len(source_contract_files),
                    "frozen_project_root": str(frozen_project_root),
                    "snapshot_root": str(code_snapshot_root.resolve()),
                },
                "audit_mode": {
                    "full_scan_a": True,
                    "full_real_scorer": True,
                    "max_workers": args.max_workers,
                },
                "part_counts": part_counts,
                "parts": part_records,
            }
            _atomic_write_json(status_path, status)

        _run_bounded_workers(
            worker_specs,
            max_workers=args.max_workers,
            cwd=frozen_project_root,
            env=child_env,
            publish_status=publish_status,
        )
    finally:
        try:
            exporter.rollback()
        finally:
            exporter.close()


if __name__ == "__main__":
    main()
