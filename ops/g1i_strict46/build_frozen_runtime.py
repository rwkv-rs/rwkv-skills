#!/usr/bin/env python3
"""Publish a root-owned content-addressed strict-46 scheduler runtime.

Run this tool as root after a PASS approval and lock exist.  The output is a
read-only tree owned by root and readable/executable by the supplied runtime
group.  Normal evaluation users are intentionally unable to publish a trusted
runtime themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from uuid import uuid4

_BOOTSTRAP_REPO = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_REPO) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO))

from ops.g1i_strict46 import require_global_protocol_gate as gate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_BOOTSTRAP_REPO)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--runtime-gid", type=int, required=True)
    parser.add_argument(
        "--python-runtime-root",
        type=Path,
        required=True,
        help="root-owned, read-only, symlink-free Python environment tree",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        required=True,
        help="executable inside --python-runtime-root (use venv --copies)",
    )
    return parser.parse_args()


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    payload = gate._read_stable_bytes(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_once(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o550 if executable else 0o440)


def _copy_source(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise gate.ProtocolGateError(f"frozen source must not be a symlink: {source}")
    executable = bool(source.stat().st_mode & stat.S_IXUSR)
    _write_once(destination, gate._read_stable_bytes(source), executable=executable)


def _dataset_paths(repo: Path) -> dict[str, Path]:
    from src.eval.scheduler.datasets import find_dataset_file, refresh_dataset_index

    root = (repo / "data").resolve(strict=True)
    refresh_dataset_index((root,))
    resolved: dict[str, Path] = {}
    for dataset in sorted(gate.EXPECTED_DATASETS):
        path = find_dataset_file(dataset, (root,))
        if path is None:
            raise gate.ProtocolGateError(f"strict-46 dataset is missing: {dataset}")
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise gate.ProtocolGateError(f"strict-46 dataset is a symlink: {expanded}")
        candidate = expanded.resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise gate.ProtocolGateError(
                f"strict-46 dataset is outside repository data root: {dataset}"
            ) from exc
        resolved[dataset] = candidate
    return resolved


def _manifest_candidates(dataset_path: Path) -> tuple[Path, ...]:
    candidates = (
        Path(f"{dataset_path}.manifest.json"),
        dataset_path.with_suffix(".manifest.json"),
    )
    return tuple(dict.fromkeys(path for path in candidates if path.is_file()))


def _lock_and_approval(repo: Path, lock: Path, approval: Path) -> tuple[dict, dict]:
    verified_lock = gate._verify_lock(repo, lock)
    gate._verify_global_approval(
        repo,
        approval,
        locked_descriptor=verified_lock.get("global_approval"),
        locked_runtime_evidence_sha256=verified_lock.get(
            "runtime_attestation_evidence_sha256"
        ),
    )
    approval_payload = json.loads(gate._read_stable_bytes(approval).decode("utf-8"))
    return verified_lock, approval_payload


def _relative_source(repo: Path, path: Path, *, label: str) -> str:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise gate.ProtocolGateError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    try:
        relative = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise gate.ProtocolGateError(f"{label} is outside the release source") from exc
    return gate._reject_release_path(relative, label=label)


def main() -> int:
    args = _parse_args()
    if os.name != "posix" or os.geteuid() != 0:
        print("frozen runtime publication requires root", file=sys.stderr)
        return 42
    repo = args.repo.expanduser().resolve(strict=True)
    approval = args.approval.expanduser()
    lock = args.lock.expanduser()
    if not approval.is_absolute():
        approval = repo / approval
    if not lock.is_absolute():
        lock = repo / lock
    output_parent = args.output_parent.expanduser().resolve()
    if output_parent.is_symlink():
        print("output parent must not be a symlink", file=sys.stderr)
        return 42
    output_parent.mkdir(parents=True, exist_ok=True)
    try:
        gate._require_trusted_ancestor_chain(output_parent, trusted_uid=0)
    except gate.ProtocolGateError as exc:
        print(f"output parent is not a trusted root-owned path: {exc}", file=sys.stderr)
        return 42
    staging = output_parent / f".strict46-staging-{os.getpid()}-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        _verified_lock, approval_payload = _lock_and_approval(
            repo,
            lock,
            approval,
        )
        protocol_tree = approval_payload["protocol_contract"]["protocol_tree"]
        protocol_files = dict(protocol_tree["files"])
        dataset_sources = _dataset_paths(repo)
        source_data_root = (repo / "data").resolve(strict=True)
        dataset_manifests = {
            manifest
            for source in dataset_sources.values()
            for manifest in _manifest_candidates(source)
        }
        publication_sources = set(protocol_files)
        publication_sources.update(
            _relative_source(repo, source, label=f"strict-46 dataset {dataset}")
            for dataset, source in dataset_sources.items()
        )
        publication_sources.update(
            _relative_source(repo, manifest, label="strict-46 dataset manifest")
            for manifest in dataset_manifests
        )
        publication_sources.add(
            _relative_source(repo, approval, label="global approval")
        )
        publication_sources.add(
            _relative_source(repo, lock, label="protocol lock")
        )
        # This validation happens before any release byte is copied.  It is
        # intentionally sensitive to a real top-level repo/.env even though
        # .env is absent from the allowlist: silent omission made it too easy
        # for a future broad copy to reintroduce credentials.
        gate.validate_release_source(repo, published_paths=publication_sources)

        for relative, expected_sha in sorted(protocol_files.items()):
            source = repo / relative
            if gate._sha256_file(source) != expected_sha:
                raise gate.ProtocolGateError(
                    f"approved protocol source changed while freezing: {relative}"
                )
            _copy_source(source, staging / relative)

        frozen_approval = staging / gate.APPROVAL_DIRECTORY / approval.name
        frozen_lock = staging / "ops/g1i_strict46/protocol_gate.lock.json"
        _copy_source(approval, frozen_approval)
        _copy_source(lock, frozen_lock)

        datasets: dict[str, dict[str, object]] = {}
        support_files: list[dict[str, object]] = []
        for dataset, source in dataset_sources.items():
            relative = source.relative_to(source_data_root)
            destination = staging / "data" / relative
            _copy_source(source, destination)
            datasets[dataset] = _descriptor(destination, staging)
            for manifest in _manifest_candidates(source):
                manifest_relative = manifest.resolve().relative_to(source_data_root)
                frozen_manifest = staging / "data" / manifest_relative
                if not frozen_manifest.exists():
                    _copy_source(manifest, frozen_manifest)
                    support_files.append(_descriptor(frozen_manifest, staging))

        unsigned_manifest: dict[str, object] = {
            "schema_version": gate.FROZEN_RUNTIME_SCHEMA,
            "release_policy": gate._release_policy(),
            "protocol_tree_sha256": protocol_tree["tree_sha256"],
            "protocol_files": protocol_files,
            "approval": _descriptor(frozen_approval, staging),
            "protocol_lock": _descriptor(frozen_lock, staging),
            "datasets": datasets,
            "support_files": sorted(support_files, key=lambda value: str(value["path"])),
            "python_runtime": gate.build_python_runtime_contract(
                args.python_runtime_root,
                args.python_executable,
            ),
        }
        manifest = dict(unsigned_manifest)
        manifest["manifest_sha256"] = gate._canonical_json_sha256(unsigned_manifest)
        _write_once(
            staging / gate.FROZEN_RUNTIME_MANIFEST,
            gate._canonical_json_bytes(manifest),
        )

        # Recheck every copied byte before ownership/mode sealing.
        for relative, expected_sha in sorted(protocol_files.items()):
            if gate._sha256_file(staging / relative) != expected_sha:
                raise gate.ProtocolGateError(
                    f"frozen protocol copy changed before publish: {relative}"
                )

        for path in sorted(staging.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            os.chown(path, 0, args.runtime_gid)
            if path.is_dir():
                path.chmod(0o550)
            elif path.stat().st_mode & stat.S_IXUSR:
                path.chmod(0o550)
            else:
                path.chmod(0o440)
        os.chown(staging, 0, args.runtime_gid)
        staging.chmod(0o550)

        output = output_parent / str(manifest["manifest_sha256"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen runtime: {output}")
        os.replace(staging, output)
        directory_fd = os.open(output_parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, gate.ProtocolGateError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        print(f"frozen runtime publication failed: {exc}", file=sys.stderr)
        return 42
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
