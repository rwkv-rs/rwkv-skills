#!/usr/bin/env python3
"""Fail-closed protocol gate for the strict-46 G1i launch/recovery lanes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


LOCK_SCHEMA = "rwkv.g1i-strict46-protocol-lock.v2"
APPROVAL_SCHEMA = "rwkv.free-response-global-protocol-approval.v2"
MERGE_ACCEPTANCE_SCHEMA = "free-response-global-audit-acceptance.v1"
APPROVAL_DIRECTORY = Path("ops/g1i_strict46/approvals")
JUDGE_PROTOCOL_VERSION = "rwkv.free_response.llm_judge.v1"
JUDGE_RESPONSE_CONTRACT = "trimmed_exact_literal_true_false.v1"
JUDGE_PROTOCOL_KEYS = (
    "protocol_version",
    "model",
    "temperature",
    "prompt_template_sha256",
    "max_completion_tokens",
    "response_contract",
    "stream",
    "qwen3_enable_thinking",
    "max_workers",
    "max_retries",
    "recovery_rounds",
)
PROTOCOL_TREE_SCHEMA = "rwkv.g1i-strict46-protocol-tree.v1"
FROZEN_RUNTIME_SCHEMA = "rwkv.g1i-strict46-frozen-runtime.v2"
FROZEN_RUNTIME_MANIFEST = "strict46-frozen-runtime.json"
PYTHON_RUNTIME_SCHEMA = "rwkv.g1i-strict46-python-runtime.v1"
RUNTIME_EVIDENCE_SCHEMA = "rwkv.g1i-strict46-runtime-evidence.v1"
RELEASE_POLICY_SCHEMA = "rwkv.g1i-strict46-release-policy.v1"
RELEASE_ALLOWED_ROOT_FILES = {"pyproject.toml", "uv.lock"}
RELEASE_ALLOWED_ROOTS = {
    "src",
    "configs",
    "ops/g1i_strict46",
    "scripts/oneoff",
    "data",
}
RELEASE_SECRET_EXACT_NAMES = {
    ".env",
    ".pgpass",
    ".netrc",
    "credentials",
    "credentials.json",
    "credential.json",
    "token.json",
    "tokens.json",
    "secret.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
RELEASE_SECRET_SUFFIXES = (
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".token",
    ".secret",
    ".credentials",
)
RELEASE_SECRET_COMPONENT_RE = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|credential(?:s)?|private[_-]?key|secret(?:s)?|token(?:s)?)(?:$|[._-])",
    re.IGNORECASE,
)
EXPECTED_JOBS = {
    "multi_choice_plain_naive",
    "free_response_naive",
    "free_response_judge_naive",
    "code_human_eval_naive",
    "code_mbpp_naive",
    "code_livecodebench_plain_naive",
    "instruction_following_naive",
}
KNOWLEDGE = {
    "arc_easy",
    "mmlu",
    "openbookqa",
    "cmmlu",
    "commonsense_qa",
    "ceval",
    "truthfulqa_mc1",
    "mmlu_pro",
    "hellaswag",
    "mmlu_redux",
    "winogrande",
    "agieval_mcq",
    "mmlu_sr_question_and_answer",
    "bbh_mcq",
    "kmmlu",
    "gpqa_main",
    "gpqa_extended",
    "medqa",
    "gpqa_diamond",
    "medmcqa",
    "arc_challenge",
}
MATH = {
    "aime24",
    "aime25",
    "amc23",
    "answer_judge",
    "beyond_aime",
    "brumo25",
    "comp_math_24_25",
    "gaokao2023en",
    "gsm8k",
    "hmmt_feb25",
    "math_500",
    "math_odyssey",
    "minerva_math",
    "olympiadbench",
    "simpleqa",
    "svamp",
}
CODING = {
    "human_eval",
    "human_eval_cn",
    "human_eval_fix",
    "human_eval_plus",
    "mbpp",
    "mbpp_plus",
    "livecodebench",
}
INSTRUCTION = {"ifeval", "ifbench"}
EXPECTED_DATASETS = KNOWLEDGE | MATH | CODING | INSTRUCTION
EXPECTED_JUDGE_BENCHMARKS = {
    "amc23",
    "comp_math_24_25",
    "gaokao2023en",
    "minerva_math",
}
EXPECTED_MODELS = {
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
}
EXPECTED_RUNTIME_HOSTS = {
    "rwkv7-g1i-1.5b-20260805-ctx16384": "157",
    "rwkv7-g1i-2.9b-20260805-ctx16384": "157",
    "rwkv7-g1i-7.2b-20260805-ctx16384": "8222",
    "rwkv7-g1i-13.3b-20260805-ctx16384": "8222",
}
STRICT_CONFIG_STEMS = (
    EXPECTED_DATASETS - {"gpqa_main", "gpqa_extended", "gpqa_diamond"}
) | {"gpqa"}
PROTOCOL_SOURCE_PATHS = {
    "src/eval/datasets/snapshot.py",
    "src/eval/field_common.py",
    "src/eval/benchmark_config.py",
    "src/eval/evaluating/task_persistence.py",
    "src/eval/results/schema.py",
    "src/eval/prompt_builders.py",
    "src/infer/sampling.py",
    "src/infer/backend.py",
    "src/eval/datasets/data_loader/multiple_choice.py",
    "src/eval/datasets/data_loader/free_answer.py",
    "src/eval/datasets/data_loader/code_generation.py",
    "src/eval/datasets/data_loader/instruction_following.py",
    "src/eval/tasks/knowledge/runner.py",
    "src/eval/tasks/knowledge/pipeline.py",
    "src/eval/tasks/maths/runner.py",
    "src/eval/tasks/maths/pipeline.py",
    "src/eval/tasks/maths/common.py",
    "src/eval/tasks/coding/runner.py",
    "src/eval/tasks/coding/pipeline.py",
    "src/eval/tasks/instruction_following/runner.py",
    "src/eval/tasks/instruction_following/pipeline.py",
    "src/eval/metrics/multi_choice.py",
    "src/eval/metrics/free_response.py",
    "src/eval/metrics/code_generation/evaluate.py",
    "src/eval/metrics/code_generation/livecodebench/evaluation.py",
    "src/eval/metrics/instruction_following/metrics.py",
}
OPS_PATHS = {
    "ops/g1i_strict46/require_global_protocol_gate.py",
    "ops/g1i_strict46/build_frozen_runtime.py",
    "ops/g1i_strict46/runtime_attestation.py",
    "ops/g1i_strict46/runtime_state.py",
    "ops/g1i_strict46/publish_global_protocol_approval.py",
    "ops/g1i_strict46/run_model.sh",
    "ops/g1i_strict46/run_audit_missing.py",
    "ops/g1i_strict46/audit_current.py",
    "ops/g1i_strict46/ensure_model_complete.sh",
    "ops/g1i_strict46/handoff_idle_guard.sh",
    "ops/g1i_strict46/handoff_request.py",
    "ops/g1i_strict46/wait_157_1p5.sh",
    "ops/g1i_strict46/wait_8222_13p3.sh",
    "ops/g1i_strict46/provision_root_runtime.sh",
    "ops/g1i_strict46/ROOT_PROVISIONING.md",
    "ops/g1i_strict46/templates/ssh_config.157-to-8222.in",
    "ops/g1i_strict46/templates/known_hosts.157-to-8222.in",
    "ops/g1i_strict46/templates/strict46_db_grants.sql.in",
}


class ProtocolGateError(RuntimeError):
    pass


def _release_policy() -> dict[str, object]:
    """Return the exact, non-secret publication policy bound by the manifest."""

    return {
        "schema_version": RELEASE_POLICY_SCHEMA,
        "allowed_root_files": sorted(RELEASE_ALLOWED_ROOT_FILES),
        "allowed_roots": sorted(RELEASE_ALLOWED_ROOTS),
        "secret_exact_names": sorted(RELEASE_SECRET_EXACT_NAMES),
        "secret_suffixes": list(RELEASE_SECRET_SUFFIXES),
        "symlinks_allowed": False,
    }


def _reject_release_path(relative: str | Path, *, label: str) -> str:
    """Validate one publication path against traversal and secret-name attacks.

    The rules intentionally target credential-bearing *data* names.  Python's
    standard-library ``token.py`` and ``secrets.py`` are source modules rather
    than credential files and therefore remain valid.
    """

    candidate = Path(str(relative))
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ProtocolGateError(f"{label} has an unsafe relative path: {relative}")
    rendered = candidate.as_posix()
    for part in candidate.parts:
        lowered = part.casefold()
        if lowered in RELEASE_SECRET_EXACT_NAMES or lowered.startswith(".env."):
            raise ProtocolGateError(f"{label} has a secret-like path: {rendered}")
        if lowered.endswith(RELEASE_SECRET_SUFFIXES):
            raise ProtocolGateError(f"{label} has a secret-like path: {rendered}")
        if lowered not in {"token.py", "secrets.py"} and RELEASE_SECRET_COMPONENT_RE.search(
            lowered
        ):
            raise ProtocolGateError(f"{label} has a secret-like path: {rendered}")
        stem, suffix = os.path.splitext(lowered)
        if suffix in {".json", ".yaml", ".yml", ".toml", ".ini"} and any(
            marker in stem for marker in ("credential", "private_key", "secret", "token")
        ):
            raise ProtocolGateError(f"{label} has a secret-like path: {rendered}")
    return rendered


def validate_release_source(
    repo: Path,
    *,
    published_paths: set[str] | frozenset[str],
) -> None:
    """Fail closed unless every publication source is explicit and secret-free.

    Only exact paths in ``published_paths`` may enter the frozen tree.  The
    source roots are also scanned for symlinks and credential-like files so a
    top-level ``repo/.env`` or a hidden key cannot be silently ignored by the
    allowlist and later reintroduced by a copy change.
    """

    root = repo.expanduser().resolve(strict=True)
    normalized: set[str] = set()
    for raw in published_paths:
        relative = _reject_release_path(raw, label="publication allowlist")
        if relative not in RELEASE_ALLOWED_ROOT_FILES and not any(
            relative == allowed or relative.startswith(f"{allowed}/")
            for allowed in RELEASE_ALLOWED_ROOTS
        ):
            raise ProtocolGateError(
                f"publication path is outside the release allowlist roots: {relative}"
            )
        source = root / relative
        if source.is_symlink():
            raise ProtocolGateError(f"publication source is a symlink: {relative}")
        if not source.is_file():
            raise ProtocolGateError(f"publication source is not a regular file: {relative}")
        normalized.add(relative)
    if normalized != set(published_paths):
        raise ProtocolGateError("publication allowlist is not canonical")

    scan_roots = [root / name for name in sorted(RELEASE_ALLOWED_ROOTS)]
    # Root-level entries are not recursively copied, but a symlink or a
    # case-varied credential file there is too dangerous to silently ignore.
    for candidate in root.iterdir():
        relative = candidate.name
        if candidate.is_symlink():
            raise ProtocolGateError(f"publication source is a symlink: {relative}")
        _reject_release_path(relative, label="repository release source")
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        if scan_root.is_symlink():
            raise ProtocolGateError(
                f"publication source root is a symlink: {scan_root.relative_to(root)}"
            )
        for current, directory_names, file_names in os.walk(
            scan_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            for name in tuple(directory_names) + tuple(file_names):
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                status = os.lstat(path)
                if stat.S_ISLNK(status.st_mode):
                    raise ProtocolGateError(f"publication source is a symlink: {relative}")
                _reject_release_path(relative, label="repository release source")


def _runtime_attestation_module(repo: Path):
    """Load only the protocol-bound verifier from the selected runtime tree."""

    path = repo / "ops/g1i_strict46/runtime_attestation.py"
    spec = importlib.util.spec_from_file_location(
        "rwkv_strict46_runtime_attestation",
        path,
    )
    if spec is None or spec.loader is None:
        raise ProtocolGateError("cannot load the runtime attestation verifier")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - defensive import boundary
        raise ProtocolGateError(
            f"cannot import the runtime attestation verifier: {exc}"
        ) from exc
    return module


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_stable_bytes(path)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_bytes(path: Path) -> bytes:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ProtocolGateError(f"gate evidence must not be a symlink: {unresolved}")
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ProtocolGateError(f"gate evidence is missing: {unresolved}") from exc
    if not resolved.is_file():
        raise ProtocolGateError(f"gate evidence is not a regular file: {resolved}")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = resolved.stat()
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(after) != _stat_identity(path_after):
        raise ProtocolGateError(f"gate evidence changed while it was read: {resolved}")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise ProtocolGateError(f"short read while hashing gate evidence: {resolved}")
    return payload


def protocol_inventory_paths(repo: Path) -> tuple[str, ...]:
    """Return the complete executable/config tree approved for strict-46.

    A short hand-maintained list is unsafe: scheduler mappings, a transitive
    metric helper, or a newly-created model-specific TOML can change semantics
    while an old approval continues to pass.  The inventory deliberately binds
    every Python source plus every strict benchmark config and launch script.
    """

    paths: set[Path] = set()
    # Python is not the only import-time input.  Package data (JSON, grammar,
    # prompt, tokenizer and instruction resources) can change scoring without
    # changing a .py file, so bind every regular source-tree resource.  Build
    # artefacts are excluded because they are never a protocol input.
    for path in (repo / "src").rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        paths.add(path)
    for root in (repo / "configs", repo / "configs" / "g1h"):
        template = root / "_templates.toml"
        if template.is_file():
            paths.add(template)
        for stem in STRICT_CONFIG_STEMS:
            candidate = root / f"{stem}.toml"
            if candidate.is_file():
                paths.add(candidate)
            for model in EXPECTED_MODELS:
                model_candidate = root / model / f"{stem}.toml"
                if model_candidate.is_file():
                    paths.add(model_candidate)
    # Strict runtime operations are an exact allowlist.  Glob-importing every
    # shell script previously admitted legacy user-systemd chains into an
    # otherwise approved runtime merely because a developer left the file in
    # this directory.
    for relative in OPS_PATHS:
        candidate = repo / relative
        if candidate.is_file():
            paths.add(candidate)
    for name in (
        "audit_free_response_extractor_global.py",
        "run_free_response_global_audit_parts.py",
        "merge_free_response_global_audit.py",
        "merge_free_response_global_audit_parts.py",
    ):
        candidate = repo / "scripts" / "oneoff" / name
        if candidate.is_file():
            paths.add(candidate)
    for name in ("pyproject.toml", "uv.lock"):
        candidate = repo / name
        if candidate.is_file():
            paths.add(candidate)
    rendered: list[str] = []
    for path in paths:
        if path.is_symlink():
            raise ProtocolGateError(f"protocol inventory contains a symlink: {path}")
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
        _reject_release_path(relative, label="protocol inventory")
        rendered.append(relative)
    if not rendered:
        raise ProtocolGateError("protocol inventory is empty")
    return tuple(sorted(rendered))


def _protocol_tree_contract(repo: Path) -> dict[str, object]:
    paths = protocol_inventory_paths(repo)
    first = {relative: _sha256_file(repo / relative) for relative in paths}
    second = {relative: _sha256_file(repo / relative) for relative in paths}
    if first != second or paths != protocol_inventory_paths(repo):
        raise ProtocolGateError("protocol tree changed during inventory")
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_TREE_SCHEMA,
        "files": first,
    }
    payload["tree_sha256"] = _canonical_json_sha256(payload)
    return payload


def required_protocol_paths(repo: Path) -> tuple[str, ...]:
    config_paths = {f"configs/g1h/{stem}.toml" for stem in STRICT_CONFIG_STEMS} | {
        "configs/g1h/_templates.toml"
    }
    paths = PROTOCOL_SOURCE_PATHS | OPS_PATHS | config_paths
    missing = [path for path in sorted(paths) if not (repo / path).is_file()]
    if missing:
        raise ProtocolGateError(
            "required protocol files are missing: " + ", ".join(missing)
        )
    return tuple(sorted(paths))


def _path_descriptor(repo: Path, path: Path) -> dict[str, str]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ProtocolGateError(f"gate descriptor target must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    try:
        rendered = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": _sha256_file(resolved)}


def _resolve_descriptor_path(repo: Path, raw_path: Any) -> Path:
    value = Path(str(raw_path or ""))
    if not value.is_absolute():
        value = repo / value
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ProtocolGateError(f"gate descriptor target must not be a symlink: {expanded}")
    return expanded.resolve()


def current_lock_payload(
    repo: Path,
    *,
    approval_path: Path | None = None,
) -> dict[str, object]:
    files = {
        relative: _sha256_file(repo / relative)
        for relative in required_protocol_paths(repo)
    }
    if files != {
        relative: _sha256_file(repo / relative)
        for relative in required_protocol_paths(repo)
    }:
        raise ProtocolGateError("protocol source/config changed during lock inventory")
    runtime_evidence_sha256: str | None = None
    if approval_path is not None:
        try:
            approval_document = json.loads(
                _read_stable_bytes(approval_path).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolGateError(
                "cannot bind runtime evidence from invalid approval JSON"
            ) from exc
        evidence = (
            approval_document.get("runtime_attestation_evidence")
            if isinstance(approval_document, dict)
            else None
        )
        candidate_sha = evidence.get("evidence_sha256") if isinstance(evidence, dict) else None
        if not _is_sha256(candidate_sha):
            raise ProtocolGateError("approval has no valid runtime evidence digest")
        runtime_evidence_sha256 = str(candidate_sha)
    payload: dict[str, object] = {
        "schema_version": LOCK_SCHEMA,
        "strict_scope": {
            "benchmark_count": 46,
            "knowledge_mode": "NoCoT",
            "math_mode": "CoT",
            "coding_mode": "NoCoT",
            "instruction_following_mode": "NoCoT",
            "prompt_profile": "naive",
            "swe_enabled": False,
        },
        "files": files,
        # A source/config lock is necessary but never sufficient.  A usable
        # lock must also pin one immutable PASS-only global approval artifact.
        "global_approval": (
            _path_descriptor(repo, approval_path) if approval_path is not None else None
        ),
        "runtime_attestation_evidence_sha256": runtime_evidence_sha256,
    }
    payload["lock_sha256"] = _canonical_json_sha256(payload)
    return payload


def _load_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_stable_bytes(path).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolGateError(f"protocol lock is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolGateError(
            f"protocol lock is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtocolGateError("protocol lock root must be an object")
    return payload


def _verify_lock(repo: Path, lock_path: Path) -> dict[str, Any]:
    expected = _load_lock(lock_path)
    if expected.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolGateError("protocol lock schema mismatch")
    expected_without_digest = dict(expected)
    recorded_digest = expected_without_digest.pop("lock_sha256", None)
    computed_digest = _canonical_json_sha256(expected_without_digest)
    if recorded_digest != computed_digest:
        raise ProtocolGateError("protocol lock self-digest mismatch")

    current = current_lock_payload(repo)
    expected_files = expected.get("files")
    current_files = current["files"]
    if not isinstance(expected_files, dict):
        raise ProtocolGateError("protocol lock has no files mapping")
    if set(expected_files) != set(current_files):
        missing = sorted(set(current_files) - set(expected_files))
        unexpected = sorted(set(expected_files) - set(current_files))
        raise ProtocolGateError(
            f"protocol lock path set mismatch: missing={missing}, unexpected={unexpected}"
        )
    drift = [
        path
        for path in sorted(current_files)
        if expected_files.get(path) != current_files.get(path)
    ]
    if drift:
        raise ProtocolGateError(
            "protocol source/config hash drift: " + ", ".join(drift)
        )
    if expected.get("strict_scope") != current.get("strict_scope"):
        raise ProtocolGateError("strict scope lock mismatch")
    if expected.get("global_approval") is not None and not _is_sha256(
        expected.get("runtime_attestation_evidence_sha256")
    ):
        raise ProtocolGateError("protocol lock does not bind runtime attestation evidence")
    return expected


def _shell_section(path: Path, start: str, end: str) -> list[str]:
    tokens = [
        token
        for token in shlex.split(path.read_text(encoding="utf-8"))
        if token.strip()
    ]
    try:
        return tokens[tokens.index(start) + 1 : tokens.index(end)]
    except ValueError as exc:
        raise ProtocolGateError(
            f"missing shell scope delimiter in {path}: {exc}"
        ) from exc


def _verify_strict_scope(repo: Path) -> None:
    run_model = repo / "ops/g1i_strict46/run_model.sh"
    jobs = set(_shell_section(run_model, "--only-jobs", "--only-datasets"))
    datasets = _shell_section(run_model, "--only-datasets", "--infer-base-url")
    if jobs != EXPECTED_JOBS:
        raise ProtocolGateError(f"strict job scope mismatch: {sorted(jobs)}")
    if (
        len(datasets) != 46
        or len(set(datasets)) != 46
        or set(datasets) != EXPECTED_DATASETS
    ):
        raise ProtocolGateError(
            "strict dataset scope is not exactly the approved 46 benchmarks"
        )
    if any(item.lower().startswith("swe_") for item in (*jobs, *datasets)):
        raise ProtocolGateError("SWE is forbidden in the strict-46 queue")
    run_text = run_model.read_text(encoding="utf-8")
    if 'RWKV_BENCHMARK_CONFIG_ROOT="$repo/configs/g1h"' not in run_text:
        raise ProtocolGateError("strict config root is not pinned to configs/g1h")
    configured_root = os.environ.get("RWKV_BENCHMARK_CONFIG_ROOT", "").strip()
    if (
        configured_root
        and Path(configured_root).expanduser().resolve()
        != (repo / "configs/g1h").resolve()
    ):
        raise ProtocolGateError(
            f"RWKV_BENCHMARK_CONFIG_ROOT points outside strict root: {configured_root}"
        )


def _require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise ProtocolGateError(f"protocol markers missing from {path}: {missing}")


def _verify_protocol_invariants(repo: Path) -> None:
    for field in ("knowledge", "maths", "coding", "instruction_following"):
        _require_markers(
            repo / f"src/eval/tasks/{field}/runner.py",
            (
                "build_dataset_snapshot",
                "build_protocol_bundle",
                "dataset_snapshot=",
                "protocol_bundle=",
            ),
        )
    _require_markers(
        repo / "src/eval/metrics/free_response.py",
        (
            "_terminal_complete_result_candidate",
            "_has_blank_recovery_stage",
            "llm_judge_protocol",
            "llm_judge_protocol_fingerprint",
            "llm_judge_protocol_stats_reasons",
        ),
    )
    _require_markers(
        repo / "src/eval/tasks/knowledge/runner.py",
        ("choice_sampling_protocol", "target_token_format"),
    )
    _require_markers(
        repo / "src/eval/evaluating/task_persistence.py",
        ("bind_resume_identity", "bound_sampling_config"),
    )

    waiter_157 = (repo / "ops/g1i_strict46/wait_157_1p5.sh").read_text(encoding="utf-8")
    if "CUDA_VISIBLE_DEVICES=3" not in waiter_157 or "rwkv7-g1i-1.5b" not in waiter_157:
        raise ProtocolGateError("1.5B handoff is not pinned to released 157 GPU3")
    waiter_8222 = (repo / "ops/g1i_strict46/wait_8222_13p3.sh").read_text(
        encoding="utf-8"
    )
    if (
        "CUDA_VISIBLE_DEVICES=2" not in waiter_8222
        or "rwkv7-g1i-13.3b" not in waiter_8222
    ):
        raise ProtocolGateError("13.3B handoff is not pinned to 8222 GPU2")
    forbidden = ("CUDA_VISIBLE_DEVICES=3", "port=18073", "GPU3", "gpu3")
    if any(marker in waiter_8222 for marker in forbidden):
        raise ProtocolGateError("8222 GPU3/18073 is reserved and must never be touched")


def validate_single_model_response(payload: Any, expected_model: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ProtocolGateError("/v1/models response has no data list")
    model_ids = [
        item.get("id")
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if len(model_ids) != 1:
        raise ProtocolGateError(f"expected exactly one model id, got {model_ids!r}")
    if model_ids[0] != expected_model:
        raise ProtocolGateError(
            f"endpoint model mismatch: expected={expected_model!r}, actual={model_ids[0]!r}"
        )
    return model_ids[0]


def verify_inference_endpoint(
    infer_base_url: str,
    expected_model: str,
    *,
    api_key: str,
    timeout_s: float = 10.0,
) -> str:
    """Require the dispatch endpoint to expose exactly the approved model.

    Recovery used to pass ``--infer-base-url`` to this gate while the gate
    silently ignored it.  That allowed a scheduler to be approved for one
    model and dispatch to a stale endpoint serving another.  Build the models
    URL ourselves and validate the response immediately before dispatch.
    """

    raw_url = str(infer_base_url or "").strip()
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProtocolGateError(f"invalid inference base URL: {raw_url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProtocolGateError("inference base URL must not contain credentials/query/fragment")
    base_path = parsed.path.rstrip("/")
    models_path = f"{base_path}/models" if base_path else "/models"
    models_url = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(models_url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ProtocolGateError(
            f"failed to verify inference endpoint {models_url}: {exc}"
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolGateError(
            f"inference endpoint returned invalid JSON: {models_url}"
        ) from exc
    return validate_single_model_response(payload, expected_model)


def _run_remote_runtime_attestation(
    *,
    repo: Path,
    approval_path: Path,
    model: str,
    runtime_artifact: dict[str, Any],
    forward_artifact: dict[str, Any],
    infer_api_key: str,
    timeout_s: float = 900.0,
) -> None:
    """Run the frozen verifier on 8222 over the approved SSH trust route."""

    destination = runtime_artifact["endpoint"]
    destination_url = (
        f"{destination['scheme']}://{destination['host']}:{destination['port']}"
        f"{destination['api_prefix']}"
    )
    prefix = list(forward_artifact["verification_argv_prefix"])
    remote_gate = repo / "ops/g1i_strict46/require_global_protocol_gate.py"
    remote_lock = repo / "ops/g1i_strict46/protocol_gate.lock.json"
    remote_approval = repo / APPROVAL_DIRECTORY / approval_path.name
    remote_command = [
        "/usr/bin/python3",
        "-I",
        str(remote_gate),
        "--repo",
        str(repo),
        "--lock",
        str(remote_lock),
        "--approval",
        str(remote_approval),
        "--frozen-runtime",
        str(repo),
        "--phase",
        "attest",
        "--model",
        model,
        "--infer-base-url",
        destination_url,
        "--infer-api-key",
        infer_api_key,
        "--attest-runtime-host-local",
    ]
    # OpenSSH concatenates arguments after the host and executes them through
    # the remote login shell.  Pass one shell-quoted command string so an API
    # key or path containing whitespace/metacharacters cannot become syntax.
    command = [*prefix, shlex.join(remote_command)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolGateError(
            f"remote runtime attestation transport failed for {model}: {exc}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()[-2000:]
        raise ProtocolGateError(
            f"remote runtime attestation failed for {model}: {diagnostic}"
        )


def _verify_inference_runtime_route(
    *,
    repo: Path,
    approval: dict[str, Any] | None,
    approval_path: Path,
    model: str,
    infer_base_url: str,
    infer_api_key: str,
    local_only: bool = False,
    proc_root: Path = Path("/proc"),
    trusted_uid: int = 0,
    security_anchor: Path = Path("/"),
    remote_runner=None,
) -> dict[str, Any]:
    """Verify local vLLM or the complete 157-to-8222 runtime chain."""

    if not isinstance(approval, dict):
        raise ProtocolGateError("runtime attestation approval is unavailable")
    evidence = approval.get("runtime_attestation_evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("models"), dict):
        raise ProtocolGateError("runtime attestation evidence is unavailable")
    entry = evidence["models"].get(model)
    if not isinstance(entry, dict):
        raise ProtocolGateError(f"runtime attestation is missing for {model}")
    runtime_artifact = entry.get("runtime_attestation")
    route = entry.get("route")
    if not isinstance(runtime_artifact, dict) or not isinstance(route, dict):
        raise ProtocolGateError(f"runtime attestation route is invalid for {model}")
    verifier = _runtime_attestation_module(repo)
    api_key = str(infer_api_key or "")
    try:
        if local_only:
            result = verifier.verify_runtime_attestation_payload(
                runtime_artifact,
                model,
                infer_base_url,
                api_key=api_key,
                proc_root=proc_root,
                trusted_uid=trusted_uid,
                security_anchor=security_anchor,
            )
            return {"runtime": result}
        scheduler_endpoint = verifier.endpoint_contract_from_url(infer_base_url)
        if scheduler_endpoint != route.get("scheduler_endpoint"):
            raise ProtocolGateError(
                f"scheduler endpoint is not approved for runtime {model}"
            )
        if route.get("kind") == "local":
            result = verifier.verify_runtime_attestation_payload(
                runtime_artifact,
                model,
                infer_base_url,
                api_key=api_key,
                proc_root=proc_root,
                trusted_uid=trusted_uid,
                security_anchor=security_anchor,
            )
            return {"runtime": result}
        if route.get("kind") != "ssh_forward":
            raise ProtocolGateError(f"unapproved runtime route for {model}")
        forward_artifact = route.get("forward_attestation")
        if not isinstance(forward_artifact, dict):
            raise ProtocolGateError(f"forward attestation is missing for {model}")
        forward_result = verifier.verify_forward_attestation_payload(
            forward_artifact,
            infer_base_url,
            proc_root=proc_root,
            trusted_uid=trusted_uid,
            security_anchor=security_anchor,
        )
        runner = remote_runner or _run_remote_runtime_attestation
        runner(
            repo=repo,
            approval_path=approval_path,
            model=model,
            runtime_artifact=runtime_artifact,
            forward_artifact=forward_artifact,
            infer_api_key=api_key,
        )
        return {"forward": forward_result, "remote_runtime_verified": True}
    except verifier.RuntimeAttestationError as exc:
        raise ProtocolGateError(f"runtime attestation failed for {model}: {exc}") from exc


def _strict_catalogue_sha256() -> str:
    return _canonical_json_sha256(
        {
            "jobs": sorted(EXPECTED_JOBS),
            "datasets": sorted(EXPECTED_DATASETS),
            "models": sorted(EXPECTED_MODELS),
            "modes": {
                "knowledge": "NoCoT",
                "math": "CoT",
                "coding": "NoCoT",
                "instruction_following": "NoCoT",
            },
            "prompt_profile": "naive",
            "swe_enabled": False,
        }
    )


def _read_hashed_json(
    repo: Path,
    descriptor: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(descriptor, dict):
        raise ProtocolGateError(f"{label} descriptor is missing")
    path = _resolve_descriptor_path(repo, descriptor.get("path"))
    expected_sha = descriptor.get("sha256")
    if not path.is_file() or path.is_symlink():
        raise ProtocolGateError(f"{label} is missing or is a symlink: {path}")
    payload_bytes = _read_stable_bytes(path)
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if not _is_sha256(expected_sha) or expected_sha != actual_sha:
        raise ProtocolGateError(
            f"{label} SHA mismatch: expected={expected_sha!r}, actual={actual_sha}"
        )
    try:
        document = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolGateError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ProtocolGateError(f"{label} root must be an object")
    return document, path


def _verify_judge_protocol_evidence(evidence: Any) -> None:
    if not isinstance(evidence, dict) or evidence.get("locked") is not True:
        raise ProtocolGateError("Judge protocol evidence is missing or unlocked")
    if evidence.get("source") != "production_score_metrics":
        raise ProtocolGateError("Judge protocol evidence has an unapproved source")
    expected_evidence_sha = evidence.get("evidence_sha256")
    digest_payload = dict(evidence)
    digest_payload.pop("evidence_sha256", None)
    if not _is_sha256(expected_evidence_sha) or (
        expected_evidence_sha != _canonical_json_sha256(digest_payload)
    ):
        raise ProtocolGateError("Judge protocol evidence digest mismatch")

    rows = evidence.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ProtocolGateError("Judge protocol evidence contains no persisted rows")
    task_ids: set[int] = set()
    score_ids: set[int] = set()
    benchmarks: set[str] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ProtocolGateError("Judge protocol evidence row is invalid")
        try:
            task_id = int(raw_row["task_id"])
            score_id = int(raw_row["score_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolGateError(
                "Judge evidence has invalid task/score identity"
            ) from exc
        benchmark = str(raw_row.get("benchmark") or "")
        model = str(raw_row.get("model") or "")
        if task_id <= 0 or score_id <= 0 or benchmark not in MATH:
            raise ProtocolGateError("Judge evidence row is outside strict Math scope")
        if model not in EXPECTED_MODELS:
            raise ProtocolGateError("Judge evidence row uses an unapproved model")
        protocol = raw_row.get("protocol")
        if not isinstance(protocol, dict):
            raise ProtocolGateError("Judge evidence row has no protocol mapping")
        if set(protocol) != set(JUDGE_PROTOCOL_KEYS) | {"protocol_fingerprint_sha256"}:
            raise ProtocolGateError(
                "Judge protocol fields do not match the locked schema"
            )
        canonical_protocol = {key: protocol.get(key) for key in JUDGE_PROTOCOL_KEYS}
        fingerprint = protocol.get("protocol_fingerprint_sha256")
        if not _is_sha256(fingerprint) or fingerprint != _canonical_json_sha256(
            canonical_protocol
        ):
            raise ProtocolGateError("Judge protocol fingerprint mismatch")
        if canonical_protocol["protocol_version"] != JUDGE_PROTOCOL_VERSION:
            raise ProtocolGateError("Judge protocol version mismatch")
        if canonical_protocol["response_contract"] != JUDGE_RESPONSE_CONTRACT:
            raise ProtocolGateError("Judge response contract mismatch")
        if canonical_protocol["temperature"] != 0.0:
            raise ProtocolGateError(
                "Judge protocol is not deterministic temperature-zero"
            )
        if canonical_protocol["stream"] is not False:
            raise ProtocolGateError("Judge protocol unexpectedly enables streaming")
        if not str(canonical_protocol["model"] or ""):
            raise ProtocolGateError("Judge protocol model is empty")
        if not _is_sha256(canonical_protocol["prompt_template_sha256"]):
            raise ProtocolGateError("Judge prompt template hash is invalid")
        max_completion_tokens = canonical_protocol["max_completion_tokens"]
        if max_completion_tokens is not None and (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0
        ):
            raise ProtocolGateError("Judge max_completion_tokens is invalid")
        for key in ("max_workers", "max_retries", "recovery_rounds"):
            value = canonical_protocol[key]
            minimum = 1 if key == "max_workers" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ProtocolGateError(f"Judge protocol has invalid {key}")
        expected_thinking = (
            False if "qwen3" in str(canonical_protocol["model"]).lower() else None
        )
        if canonical_protocol["qwen3_enable_thinking"] is not expected_thinking:
            raise ProtocolGateError("Judge thinking-mode contract mismatch")
        if task_id in task_ids or score_id in score_ids:
            raise ProtocolGateError("Judge evidence contains duplicate task/score identity")
        task_ids.add(task_id)
        score_ids.add(score_id)
        benchmarks.add(benchmark)

    if sorted(task_ids) != sorted(
        int(value) for value in evidence.get("task_ids") or []
    ):
        raise ProtocolGateError("Judge evidence task coverage mismatch")
    if sorted(score_ids) != sorted(
        int(value) for value in evidence.get("score_ids") or []
    ):
        raise ProtocolGateError("Judge evidence score coverage mismatch")
    if sorted(benchmarks) != sorted(
        str(value) for value in evidence.get("benchmarks") or []
    ):
        raise ProtocolGateError("Judge evidence benchmark coverage mismatch")
    if benchmarks != EXPECTED_JUDGE_BENCHMARKS:
        raise ProtocolGateError(
            "Judge evidence does not cover the exact strict-46 Judge benchmark set"
        )


def _verify_runtime_attestation_evidence(
    repo: Path,
    evidence: Any,
) -> dict[str, Any]:
    """Validate the content-addressed live-runtime contract for all four models."""

    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version",
        "models",
        "evidence_sha256",
    }:
        raise ProtocolGateError("runtime attestation evidence schema is incomplete")
    if evidence.get("schema_version") != RUNTIME_EVIDENCE_SCHEMA:
        raise ProtocolGateError("runtime attestation evidence version mismatch")
    evidence_sha = evidence.get("evidence_sha256")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    if not _is_sha256(evidence_sha) or evidence_sha != _canonical_json_sha256(
        unsigned
    ):
        raise ProtocolGateError("runtime attestation evidence digest mismatch")
    models = evidence.get("models")
    if not isinstance(models, dict) or set(models) != EXPECTED_MODELS:
        raise ProtocolGateError(
            "runtime attestation evidence must cover exactly four G1i models"
        )
    verifier = _runtime_attestation_module(repo)
    for model, raw_entry in sorted(models.items()):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "runtime_attestation",
            "route",
        }:
            raise ProtocolGateError(f"runtime evidence entry is invalid: {model}")
        try:
            runtime_artifact = verifier._validate_artifact(
                raw_entry["runtime_attestation"]
            )
        except verifier.RuntimeAttestationError as exc:
            raise ProtocolGateError(
                f"runtime attestation artifact is invalid for {model}: {exc}"
            ) from exc
        if runtime_artifact["model"]["name"] != model:
            raise ProtocolGateError(
                f"runtime attestation model binding mismatch: {model}"
            )
        expected_host = EXPECTED_RUNTIME_HOSTS[model]
        if runtime_artifact["host_label"] != expected_host:
            raise ProtocolGateError(
                f"runtime attestation host mismatch for {model}: {expected_host} required"
            )
        route = raw_entry["route"]
        if not isinstance(route, dict) or route.get("kind") not in {
            "local",
            "ssh_forward",
        }:
            raise ProtocolGateError(f"runtime route is invalid for {model}")
        if route["kind"] == "local":
            if set(route) != {"kind", "scheduler_endpoint"} or expected_host != "157":
                raise ProtocolGateError(f"local runtime route is invalid for {model}")
            try:
                scheduler_endpoint = verifier._endpoint_contract(
                    route["scheduler_endpoint"]
                )
            except verifier.RuntimeAttestationError as exc:
                raise ProtocolGateError(
                    f"scheduler endpoint is invalid for {model}: {exc}"
                ) from exc
            if scheduler_endpoint != runtime_artifact["endpoint"]:
                raise ProtocolGateError(
                    f"local scheduler/runtime endpoint mismatch for {model}"
                )
            continue
        if set(route) != {
            "kind",
            "scheduler_endpoint",
            "forward_attestation",
        } or expected_host != "8222":
            raise ProtocolGateError(f"SSH-forward runtime route is invalid for {model}")
        try:
            scheduler_endpoint = verifier._endpoint_contract(
                route["scheduler_endpoint"]
            )
            forward_artifact = verifier._validate_forward_artifact(
                route["forward_attestation"]
            )
        except verifier.RuntimeAttestationError as exc:
            raise ProtocolGateError(
                f"forward attestation artifact is invalid for {model}: {exc}"
            ) from exc
        if forward_artifact["endpoint"] != scheduler_endpoint:
            raise ProtocolGateError(
                f"forward listener/scheduler endpoint mismatch for {model}"
            )
        if forward_artifact["destination"]["endpoint"] != runtime_artifact["endpoint"]:
            raise ProtocolGateError(
                f"forward destination/runtime endpoint mismatch for {model}"
            )
    return evidence


def _verify_merge_acceptance(
    repo: Path,
    descriptor: Any,
) -> dict[str, Any]:
    acceptance, _path = _read_hashed_json(
        repo,
        descriptor,
        label="global audit merge acceptance",
    )
    if acceptance.get("schema_version") != MERGE_ACCEPTANCE_SCHEMA:
        raise ProtocolGateError("global audit merge acceptance schema mismatch")
    if (
        acceptance.get("accepted") is not True
        or acceptance.get("gate_passed") is not True
    ):
        raise ProtocolGateError("global audit merge acceptance is not PASS")

    current_merge_script_sha = _sha256_file(
        repo / "scripts/oneoff/merge_free_response_global_audit.py"
    )
    if acceptance.get("merge_script_sha256") != current_merge_script_sha:
        raise ProtocolGateError("global audit merge script SHA is stale")

    result, _result_path = _read_hashed_json(
        repo,
        acceptance.get("json"),
        label="global audit merged result",
    )
    if (
        not isinstance(result.get("gate"), dict)
        or result["gate"].get("passed") is not True
    ):
        raise ProtocolGateError("global audit merged result gate did not pass")
    if any(value is not True for value in result["gate"].values()):
        raise ProtocolGateError("global audit merged result contains a failed sub-gate")
    if set(result.get("groups") or []) != {"strategy_a", "strategy_b", "strategy_c"}:
        raise ProtocolGateError("global audit did not cover all strategy groups")
    if result.get("merge_script_sha256") != current_merge_script_sha:
        raise ProtocolGateError("global audit result merge script SHA is stale")

    provenance = acceptance.get("production_provenance")
    if not isinstance(provenance, dict) or provenance != result.get(
        "production_provenance"
    ):
        raise ProtocolGateError(
            "merge acceptance provenance does not match merged result"
        )
    module_shas = provenance.get("module_shas")
    if not isinstance(module_shas, dict):
        raise ProtocolGateError("global acceptance has no module SHA evidence")
    current_candidate = _sha256_file(repo / "src/eval/metrics/free_response.py")
    current_audit = _sha256_file(
        repo / "scripts/oneoff/audit_free_response_extractor_global.py"
    )
    if module_shas.get("candidate") != current_candidate:
        raise ProtocolGateError("global acceptance candidate SHA is stale")
    if module_shas.get("audit") != current_audit:
        raise ProtocolGateError("global acceptance audit SHA is stale")
    if not _is_sha256(module_shas.get("baseline")):
        raise ProtocolGateError("global acceptance baseline SHA is invalid")
    for key in (
        "metadata_snapshot_digest",
        "dependency_environment_sha256",
        "dependency_manifest_file_sha256",
        "task_inventory_digest",
    ):
        if not _is_sha256(provenance.get(key)):
            raise ProtocolGateError(f"global acceptance has invalid {key}")
    dataset_digests = provenance.get("dataset_digests")
    if not isinstance(dataset_digests, dict) or not dataset_digests:
        raise ProtocolGateError("global acceptance has no dataset snapshot evidence")
    for dataset, raw_digest in dataset_digests.items():
        digest = raw_digest if isinstance(raw_digest, dict) else {}
        if not _is_sha256(digest.get("file_sha256")) or not _is_sha256(
            digest.get("records_sha256")
        ):
            raise ProtocolGateError(f"invalid dataset snapshot digest: {dataset}")
        if int(digest.get("record_count") or 0) <= 0:
            raise ProtocolGateError(f"invalid dataset snapshot record count: {dataset}")

    input_artifacts = provenance.get("input_artifacts")
    if not isinstance(input_artifacts, list) or not input_artifacts:
        raise ProtocolGateError("global acceptance has no immutable input artifacts")
    if input_artifacts != result.get("input_artifacts"):
        raise ProtocolGateError("merged result input artifacts disagree with provenance")
    input_artifacts_sha = provenance.get("input_artifacts_sha256")
    if (
        not _is_sha256(input_artifacts_sha)
        or input_artifacts_sha != _canonical_json_sha256(input_artifacts)
        or input_artifacts_sha != result.get("input_artifacts_sha256")
    ):
        raise ProtocolGateError("global audit input artifact set digest mismatch")
    artifact_keys: set[tuple[str, int]] = set()
    artifact_order: list[tuple[str, int]] = []
    partitions_by_group: dict[str, set[int]] = {}
    for raw_artifact in input_artifacts:
        if not isinstance(raw_artifact, dict) or set(raw_artifact) != {
            "group",
            "partition_index",
            "path",
            "bytes",
            "sha256",
        }:
            raise ProtocolGateError("global audit input artifact descriptor is invalid")
        group = str(raw_artifact.get("group") or "")
        try:
            partition_index = int(raw_artifact["partition_index"])
            expected_bytes = int(raw_artifact["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolGateError("global audit input artifact identity is invalid") from exc
        if group not in {"strategy_a", "strategy_b", "strategy_c"}:
            raise ProtocolGateError("global audit input artifact group is invalid")
        if partition_index < 0 or expected_bytes <= 0:
            raise ProtocolGateError("global audit input artifact size/index is invalid")
        artifact_key = (group, partition_index)
        if artifact_key in artifact_keys:
            raise ProtocolGateError("global audit input artifact identity is duplicated")
        artifact_keys.add(artifact_key)
        artifact_order.append(artifact_key)
        partitions_by_group.setdefault(group, set()).add(partition_index)

        artifact_path = _resolve_descriptor_path(repo, raw_artifact.get("path"))
        artifact_bytes = _read_stable_bytes(artifact_path)
        if artifact_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ProtocolGateError("global audit input artifact is writable")
        if len(artifact_bytes) != expected_bytes:
            raise ProtocolGateError("global audit input artifact byte count mismatch")
        if (
            not _is_sha256(raw_artifact.get("sha256"))
            or hashlib.sha256(artifact_bytes).hexdigest() != raw_artifact["sha256"]
        ):
            raise ProtocolGateError("global audit input artifact SHA mismatch")
    if artifact_order != sorted(artifact_order):
        raise ProtocolGateError("global audit input artifacts are not canonically ordered")
    partition_sets = list(partitions_by_group.values())
    if set(partitions_by_group) != {"strategy_a", "strategy_b", "strategy_c"} or any(
        partitions != partition_sets[0] for partitions in partition_sets[1:]
    ) or partition_sets[0] != set(range(max(partition_sets[0]) + 1)):
        raise ProtocolGateError("global audit input artifact partition coverage mismatch")

    audit_mode = result.get("audit_mode")
    if not isinstance(audit_mode, dict):
        raise ProtocolGateError("global audit mode evidence is missing")
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
        "dependency_manifest_verified": True,
        "atomic_part_artifact": True,
    }
    if any(audit_mode.get(key) != value for key, value in required_mode.items()):
        raise ProtocolGateError(
            "global acceptance was not produced by production audit mode"
        )
    totals = result.get("totals")
    if not isinstance(totals, dict):
        raise ProtocolGateError("global audit totals are missing")
    for key in (
        "scoring_errors",
        "indeterminate_rows",
        "judge_input_affected_rows",
        "replay_affected_rows",
        "blocking_timeout_count",
    ):
        if int(totals.get(key) or 0) != 0:
            raise ProtocolGateError(f"global audit has blocking {key}")
    return provenance


def _verify_global_approval(
    repo: Path,
    approval_path: Path,
    *,
    locked_descriptor: Any,
    locked_runtime_evidence_sha256: Any = None,
    trust_published_evidence: bool = False,
) -> dict[str, Any]:
    expanded_approval = approval_path.expanduser()
    if expanded_approval.is_symlink():
        raise ProtocolGateError("global approval artifact must not be a symlink")
    resolved = expanded_approval.resolve()
    approval_root = (repo / APPROVAL_DIRECTORY).resolve()
    if resolved.parent != approval_root:
        raise ProtocolGateError(
            f"global approval must be content-addressed under {approval_root}"
        )
    if not resolved.is_file() or resolved.is_symlink():
        raise ProtocolGateError("global approval artifact is missing or is a symlink")
    if resolved.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ProtocolGateError("global approval artifact is writable")
    locked_path = _resolve_descriptor_path(
        repo,
        locked_descriptor.get("path") if isinstance(locked_descriptor, dict) else None,
    )
    if locked_path != resolved:
        raise ProtocolGateError("protocol lock points to a different global approval")
    approval_bytes = _read_stable_bytes(resolved)
    approval_file_sha = hashlib.sha256(approval_bytes).hexdigest()
    if not isinstance(locked_descriptor, dict) or (
        locked_descriptor.get("sha256") != approval_file_sha
    ):
        raise ProtocolGateError("protocol lock does not bind the global approval SHA")

    try:
        approval = json.loads(approval_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolGateError(f"global approval is invalid JSON: {exc}") from exc
    if (
        not isinstance(approval, dict)
        or approval.get("schema_version") != APPROVAL_SCHEMA
    ):
        raise ProtocolGateError("global approval schema mismatch")
    if approval.get("accepted") is not True or approval.get("decision") != "PASS":
        raise ProtocolGateError("global approval is not an explicit PASS")
    approval_sha = approval.get("approval_sha256")
    digest_payload = dict(approval)
    digest_payload.pop("approval_sha256", None)
    if not _is_sha256(approval_sha) or approval_sha != _canonical_json_sha256(
        digest_payload
    ):
        raise ProtocolGateError("global approval self-digest mismatch")
    if resolved.name != f"{approval_sha}.acceptance.json":
        raise ProtocolGateError("global approval filename is not content-addressed")

    merge_descriptor = approval.get("merge_acceptance")
    if trust_published_evidence:
        # The root-only frozen-runtime publisher already performed the full
        # merge/input-artifact audit before copying these immutable approval
        # bytes.  Reopening the old mutable checkout here would both defeat
        # the snapshot boundary and make a valid frozen tree depend on files
        # outside it.  The descriptor remains hash-bound by approval_sha256.
        if not isinstance(merge_descriptor, dict) or set(merge_descriptor) != {
            "path",
            "sha256",
        }:
            raise ProtocolGateError("published merge acceptance descriptor is invalid")
        if not str(merge_descriptor.get("path") or "") or not _is_sha256(
            merge_descriptor.get("sha256")
        ):
            raise ProtocolGateError("published merge acceptance binding is invalid")
        provenance = None
    else:
        provenance = _verify_merge_acceptance(repo, merge_descriptor)
    contract = approval.get("protocol_contract")
    if not isinstance(contract, dict):
        raise ProtocolGateError("global approval protocol contract is missing")
    runtime_evidence = _verify_runtime_attestation_evidence(
        repo,
        approval.get("runtime_attestation_evidence"),
    )
    runtime_evidence_sha = runtime_evidence["evidence_sha256"]
    expected_contract = {
        "candidate_module_sha256": _sha256_file(
            repo / "src/eval/metrics/free_response.py"
        ),
        "dataset_snapshot_module_sha256": _sha256_file(
            repo / "src/eval/datasets/snapshot.py"
        ),
        "resume_identity_module_sha256": _sha256_file(
            repo / "src/eval/evaluating/task_persistence.py"
        ),
        "strict_catalogue_sha256": _strict_catalogue_sha256(),
        "protocol_tree": _protocol_tree_contract(repo),
        "strict_judge_catalogue_sha256": _canonical_json_sha256(
            sorted(EXPECTED_JUDGE_BENCHMARKS)
        ),
        "runtime_attestation_evidence_sha256": runtime_evidence_sha,
    }
    if contract != expected_contract:
        raise ProtocolGateError(
            "global approval protocol contract is stale or incomplete"
        )
    if provenance is not None and provenance["module_shas"]["candidate"] != contract["candidate_module_sha256"]:
        raise ProtocolGateError("global approval candidate bindings disagree")
    _verify_judge_protocol_evidence(approval.get("judge_protocol_evidence"))
    if locked_runtime_evidence_sha256 is not None and (
        locked_runtime_evidence_sha256 != runtime_evidence_sha
    ):
        raise ProtocolGateError("protocol lock runtime evidence binding mismatch")
    return approval


def build_global_approval_payload(
    repo: Path,
    *,
    merge_acceptance_path: Path,
    judge_protocol_evidence: dict[str, Any],
    runtime_attestation_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build, but do not publish, a PASS-only approval document.

    The caller must publish the canonical bytes under the returned
    ``approval_sha256`` and make the artifact read-only.  Preflight validation
    deliberately happens before any output file is created.
    """

    merge_descriptor = _path_descriptor(repo, merge_acceptance_path)
    provenance = _verify_merge_acceptance(repo, merge_descriptor)
    _verify_judge_protocol_evidence(judge_protocol_evidence)
    runtime_attestation_evidence = _verify_runtime_attestation_evidence(
        repo,
        runtime_attestation_evidence,
    )
    contract = {
        "candidate_module_sha256": _sha256_file(
            repo / "src/eval/metrics/free_response.py"
        ),
        "dataset_snapshot_module_sha256": _sha256_file(
            repo / "src/eval/datasets/snapshot.py"
        ),
        "resume_identity_module_sha256": _sha256_file(
            repo / "src/eval/evaluating/task_persistence.py"
        ),
        "strict_catalogue_sha256": _strict_catalogue_sha256(),
        "protocol_tree": _protocol_tree_contract(repo),
        "strict_judge_catalogue_sha256": _canonical_json_sha256(
            sorted(EXPECTED_JUDGE_BENCHMARKS)
        ),
        "runtime_attestation_evidence_sha256": runtime_attestation_evidence[
            "evidence_sha256"
        ],
    }
    if provenance["module_shas"]["candidate"] != contract["candidate_module_sha256"]:
        raise ProtocolGateError(
            "merge acceptance does not approve the current candidate"
        )
    approval: dict[str, Any] = {
        "schema_version": APPROVAL_SCHEMA,
        "accepted": True,
        "decision": "PASS",
        "merge_acceptance": merge_descriptor,
        "protocol_contract": contract,
        "judge_protocol_evidence": judge_protocol_evidence,
        "runtime_attestation_evidence": runtime_attestation_evidence,
    }
    approval["approval_sha256"] = _canonical_json_sha256(approval)
    return approval


def _require_trusted_read_only(
    path: Path,
    *,
    label: str,
    trusted_uid: int,
) -> os.stat_result:
    if path.is_symlink():
        raise ProtocolGateError(f"{label} must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ProtocolGateError(f"{label} is missing: {path}") from exc
    if os.name == "posix" and int(metadata.st_uid) != int(trusted_uid):
        raise ProtocolGateError(
            f"{label} is not owned by trusted uid {trusted_uid}: {path}"
        )
    forbidden_write_bits = stat.S_IWGRP | stat.S_IWOTH
    if trusted_uid != 0:
        forbidden_write_bits |= stat.S_IWUSR
    if metadata.st_mode & forbidden_write_bits:
        raise ProtocolGateError(f"{label} is writable: {path}")
    return metadata


def _require_trusted_ancestor_chain(path: Path, *, trusted_uid: int) -> None:
    current = path
    while True:
        _require_trusted_read_only(
            current,
            label="frozen runtime ancestor",
            trusted_uid=trusted_uid,
        )
        if current == current.parent:
            return
        current = current.parent


def _frozen_child(root: Path, raw_path: Any, *, label: str) -> Path:
    relative = Path(str(raw_path or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ProtocolGateError(f"{label} must be a safe frozen-runtime relative path")
    candidate = root / relative
    if candidate.is_symlink():
        raise ProtocolGateError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProtocolGateError(f"{label} escapes frozen runtime: {candidate}") from exc
    return resolved


def _verify_frozen_parent_chain(
    root: Path,
    path: Path,
    *,
    label: str,
    trusted_uid: int,
) -> None:
    current = path.parent
    while True:
        _require_trusted_read_only(
            current,
            label=f"{label} parent directory",
            trusted_uid=trusted_uid,
        )
        if current == root:
            return
        try:
            current.relative_to(root)
        except ValueError as exc:
            raise ProtocolGateError(f"{label} parent escapes frozen runtime") from exc
        current = current.parent


def _verify_frozen_descriptor(
    root: Path,
    descriptor: Any,
    *,
    label: str,
    trusted_uid: int,
) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ProtocolGateError(f"{label} descriptor is invalid")
    path = _frozen_child(root, descriptor.get("path"), label=label)
    _verify_frozen_parent_chain(
        root,
        path,
        label=label,
        trusted_uid=trusted_uid,
    )
    metadata = _require_trusted_read_only(
        path,
        label=label,
        trusted_uid=trusted_uid,
    )
    payload = _read_stable_bytes(path)
    try:
        expected_size = int(descriptor.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ProtocolGateError(f"{label} byte count is invalid") from exc
    if expected_size != len(payload) or expected_size != int(metadata.st_size):
        raise ProtocolGateError(f"{label} byte count mismatch")
    if (
        not _is_sha256(descriptor.get("sha256"))
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise ProtocolGateError(f"{label} SHA mismatch")
    return path


def build_python_runtime_contract(
    runtime_root: Path,
    python_executable: Path,
    *,
    trusted_uid: int = 0,
) -> dict[str, object]:
    """Bind the complete Python environment used by the scheduler.

    Freezing only project sources still permits dependency or interpreter
    replacement after the gate.  The production runtime therefore lives in a
    separately sealed, trusted tree (normally a ``venv --copies`` under
    ``/opt``) and every regular byte is content-addressed.  Symlinks are
    rejected deliberately: following them would re-introduce an unbound input.
    """

    expanded_root = runtime_root.expanduser()
    if expanded_root.is_symlink():
        raise ProtocolGateError("Python runtime root must not be a symlink")
    root = expanded_root.resolve(strict=True)
    _require_trusted_ancestor_chain(root.parent, trusted_uid=trusted_uid)
    _require_trusted_read_only(
        root,
        label="Python runtime root",
        trusted_uid=trusted_uid,
    )
    executable_raw = python_executable.expanduser()
    if executable_raw.is_symlink():
        raise ProtocolGateError("Python runtime executable must not be a symlink")
    executable = executable_raw.resolve(strict=True)
    try:
        executable_relative = executable.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProtocolGateError("Python executable is outside the runtime root") from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ProtocolGateError("Python runtime executable is not executable")

    files: dict[str, dict[str, object]] = {}
    directories: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ProtocolGateError(f"Python runtime contains a symlink: {relative}")
        _reject_release_path(relative, label="Python runtime")
        if path.is_dir():
            _require_trusted_read_only(
                path,
                label=f"Python runtime directory {relative}",
                trusted_uid=trusted_uid,
            )
            directories.append(relative)
            continue
        if not path.is_file():
            raise ProtocolGateError(
                f"Python runtime contains a non-regular entry: {relative}"
            )
        metadata = _require_trusted_read_only(
            path,
            label=f"Python runtime file {relative}",
            trusted_uid=trusted_uid,
        )
        payload = _read_stable_bytes(path)
        files[relative] = {
            "size_bytes": int(metadata.st_size),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if executable_relative not in files:
        raise ProtocolGateError("Python runtime inventory omitted its executable")
    unsigned: dict[str, object] = {
        "schema_version": PYTHON_RUNTIME_SCHEMA,
        "root": str(root),
        "executable": executable_relative,
        "directories": directories,
        "files": files,
    }
    contract = dict(unsigned)
    contract["tree_sha256"] = _canonical_json_sha256(unsigned)
    return contract


def verify_python_runtime_contract(
    contract: Any,
    *,
    trusted_uid: int = 0,
) -> Path:
    if not isinstance(contract, dict) or contract.get("schema_version") != PYTHON_RUNTIME_SCHEMA:
        raise ProtocolGateError("frozen Python runtime contract schema mismatch")
    recorded_sha = contract.get("tree_sha256")
    unsigned = dict(contract)
    unsigned.pop("tree_sha256", None)
    if not _is_sha256(recorded_sha) or recorded_sha != _canonical_json_sha256(unsigned):
        raise ProtocolGateError("frozen Python runtime contract digest mismatch")
    root_raw = Path(str(contract.get("root") or ""))
    executable_raw = Path(str(contract.get("executable") or ""))
    if not root_raw.is_absolute() or executable_raw.is_absolute() or ".." in executable_raw.parts:
        raise ProtocolGateError("frozen Python runtime paths are invalid")
    # Rebuild from independently-read bytes.  Equality checks directory and
    # file inventories as well as hashes, so adding an import shadow file is a
    # gate failure rather than an unobserved runtime change.
    rebuilt = build_python_runtime_contract(
        root_raw,
        root_raw / executable_raw,
        trusted_uid=trusted_uid,
    )
    if rebuilt != contract:
        raise ProtocolGateError("frozen Python runtime changed after publication")
    return root_raw.resolve(strict=True) / executable_raw


def verify_frozen_runtime(
    *,
    source_repo: Path,
    frozen_root: Path,
    approval_path: Path,
    lock_path: Path,
    trusted_uid: int = 0,
) -> dict[str, Path | str]:
    """Verify the root-owned content snapshot used for scheduler imports.

    This is deliberately stronger than a final hash check on a mutable repo:
    the scheduler must execute from this exact tree.  Production callers may
    not override ``trusted_uid``; the argument exists so unit tests can create
    an isolated fixture without root privileges.
    """

    expanded_root = frozen_root.expanduser()
    if expanded_root.is_symlink():
        raise ProtocolGateError("frozen runtime root must not be a symlink")
    root = expanded_root.resolve(strict=True)
    _require_trusted_ancestor_chain(root.parent, trusted_uid=trusted_uid)
    _require_trusted_read_only(
        root,
        label="frozen runtime root",
        trusted_uid=trusted_uid,
    )
    manifest_path = root / FROZEN_RUNTIME_MANIFEST
    _require_trusted_read_only(
        manifest_path,
        label="frozen runtime manifest",
        trusted_uid=trusted_uid,
    )
    manifest_bytes = _read_stable_bytes(manifest_path)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolGateError(f"frozen runtime manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != FROZEN_RUNTIME_SCHEMA:
        raise ProtocolGateError("frozen runtime manifest schema mismatch")
    if manifest.get("release_policy") != _release_policy():
        raise ProtocolGateError("frozen runtime release policy mismatch")
    recorded_manifest_sha = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if (
        not _is_sha256(recorded_manifest_sha)
        or recorded_manifest_sha != _canonical_json_sha256(unsigned_manifest)
    ):
        raise ProtocolGateError("frozen runtime manifest self-digest mismatch")
    if root.name != recorded_manifest_sha:
        raise ProtocolGateError("frozen runtime directory is not content-addressed")

    approval_bytes = _read_stable_bytes(approval_path)
    lock_bytes = _read_stable_bytes(lock_path)
    try:
        approval = json.loads(approval_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolGateError("source approval is invalid while freezing") from exc
    contract = approval.get("protocol_contract") if isinstance(approval, dict) else None
    protocol_tree = contract.get("protocol_tree") if isinstance(contract, dict) else None
    if not isinstance(protocol_tree, dict):
        raise ProtocolGateError("source approval has no protocol tree")
    if manifest.get("protocol_tree_sha256") != protocol_tree.get("tree_sha256"):
        raise ProtocolGateError("frozen runtime protocol tree binding mismatch")

    frozen_approval = _verify_frozen_descriptor(
        root,
        manifest.get("approval"),
        label="frozen global approval",
        trusted_uid=trusted_uid,
    )
    frozen_lock = _verify_frozen_descriptor(
        root,
        manifest.get("protocol_lock"),
        label="frozen protocol lock",
        trusted_uid=trusted_uid,
    )
    if _read_stable_bytes(frozen_approval) != approval_bytes:
        raise ProtocolGateError("frozen global approval differs from approved source")
    if _read_stable_bytes(frozen_lock) != lock_bytes:
        raise ProtocolGateError("frozen protocol lock differs from approved source")

    files = manifest.get("protocol_files")
    approved_files = protocol_tree.get("files")
    if not isinstance(files, dict) or files != approved_files:
        raise ProtocolGateError("frozen runtime protocol file inventory mismatch")
    for relative, expected_sha in sorted(files.items()):
        _reject_release_path(relative, label="frozen protocol file")
        path = _frozen_child(root, relative, label=f"frozen protocol file {relative}")
        _verify_frozen_parent_chain(
            root,
            path,
            label=f"frozen protocol file {relative}",
            trusted_uid=trusted_uid,
        )
        _require_trusted_read_only(
            path,
            label=f"frozen protocol file {relative}",
            trusted_uid=trusted_uid,
        )
        if _sha256_file(path) != expected_sha:
            raise ProtocolGateError(f"frozen protocol file SHA mismatch: {relative}")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != EXPECTED_DATASETS:
        raise ProtocolGateError("frozen runtime dataset inventory is not strict-46")
    for dataset, descriptor in sorted(datasets.items()):
        if not isinstance(descriptor, dict):
            raise ProtocolGateError(f"frozen dataset descriptor is invalid: {dataset}")
        _reject_release_path(
            str(descriptor.get("path") or ""),
            label=f"frozen dataset {dataset}",
        )
        dataset_path = _verify_frozen_descriptor(
            root,
            descriptor,
            label=f"frozen dataset {dataset}",
            trusted_uid=trusted_uid,
        )
        try:
            dataset_path.relative_to(root / "data")
        except ValueError as exc:
            raise ProtocolGateError(
                f"frozen dataset is outside data snapshot: {dataset}"
            ) from exc

    support_files = manifest.get("support_files")
    if not isinstance(support_files, list):
        raise ProtocolGateError("frozen runtime support_files must be a list")
    support_paths: list[str] = []
    for index, descriptor in enumerate(support_files):
        if not isinstance(descriptor, dict):
            raise ProtocolGateError(f"frozen support file {index} descriptor is invalid")
        _reject_release_path(
            str(descriptor.get("path") or ""),
            label=f"frozen support file {index}",
        )
        support_path = _verify_frozen_descriptor(
            root,
            descriptor,
            label=f"frozen support file {index}",
            trusted_uid=trusted_uid,
        )
        try:
            support_relative = support_path.relative_to(root / "data")
        except ValueError as exc:
            raise ProtocolGateError(
                f"frozen support file is outside data snapshot: {index}"
            ) from exc
        if not support_relative.name.endswith(".manifest.json"):
            raise ProtocolGateError(
                f"frozen support file is not a dataset manifest: {index}"
            )
        support_paths.append(support_relative.as_posix())
    if support_paths != sorted(set(support_paths)):
        raise ProtocolGateError("frozen support file inventory is not canonical")

    python_executable = verify_python_runtime_contract(
        manifest.get("python_runtime"),
        trusted_uid=trusted_uid,
    )

    # A second full pass closes copy/verify races before the caller re-execs
    # the root-owned run_model.sh from this tree.
    if files != {
        relative: _sha256_file(_frozen_child(root, relative, label=relative))
        for relative in sorted(files)
    }:
        raise ProtocolGateError("frozen runtime changed during final verification")
    return {
        "root": root,
        "approval": frozen_approval,
        "lock": frozen_lock,
        "manifest_sha256": str(recorded_manifest_sha),
        "python_executable": python_executable,
    }


def require_gate(
    *,
    repo: Path,
    lock_path: Path,
    phase: str,
    model: str | None = None,
    approval_path: Path | None = None,
    infer_base_url: str | None = None,
    infer_api_key: str | None = None,
    frozen_runtime: Path | None = None,
    require_current_python: bool = False,
    attest_runtime_host_local: bool = False,
    runtime_route_verifier=None,
) -> None:
    if model is not None and model not in EXPECTED_MODELS:
        raise ProtocolGateError(f"model is outside the approved G1i lane: {model}")
    _verify_strict_scope(repo)
    _verify_protocol_invariants(repo)
    lock = _verify_lock(repo, lock_path)
    resolved_approval = approval_path
    if resolved_approval is None:
        configured = os.environ.get("RWKV_GLOBAL_PROTOCOL_APPROVAL", "").strip()
        if configured:
            resolved_approval = Path(configured)
    if resolved_approval is None:
        raise ProtocolGateError(
            "RWKV_GLOBAL_PROTOCOL_APPROVAL is unset; a PASS-only global audit "
            "approval is mandatory"
        )
    locked_runtime_evidence_sha = lock.get("runtime_attestation_evidence_sha256")
    if not _is_sha256(locked_runtime_evidence_sha):
        raise ProtocolGateError(
            "protocol lock has no runtime attestation evidence binding"
        )
    resolved_frozen = frozen_runtime
    if resolved_frozen is None:
        configured_frozen = os.environ.get("RWKV_STRICT_FROZEN_RUNTIME", "").strip()
        if configured_frozen:
            resolved_frozen = Path(configured_frozen)
    published_frozen = False
    runtime: dict[str, Path | str] | None = None
    if resolved_frozen is not None:
        runtime = verify_frozen_runtime(
            source_repo=repo,
            frozen_root=resolved_frozen,
            approval_path=resolved_approval,
            lock_path=lock_path,
        )
        frozen_root = Path(str(runtime["root"])).resolve(strict=True)
        if repo.resolve(strict=True) != frozen_root:
            raise ProtocolGateError(
                "gate repository is not the verified frozen runtime"
            )
        published_frozen = True
        if require_current_python:
            expected_python = Path(str(runtime["python_executable"])).resolve(strict=True)
            actual_python = Path(sys.executable).resolve(strict=True)
            if actual_python != expected_python:
                raise ProtocolGateError(
                    "scheduler interpreter is not the sealed Python runtime: "
                    f"expected={expected_python}, actual={actual_python}"
                )
    elif phase in {"dispatch", "recovery", "attest"}:
        raise ProtocolGateError(
            "RWKV_STRICT_FROZEN_RUNTIME is required for dispatch/recovery/attest; "
            "mutable-repository execution is forbidden"
        )

    _verify_global_approval(
        repo,
        resolved_approval,
        locked_descriptor=lock.get("global_approval"),
        locked_runtime_evidence_sha256=locked_runtime_evidence_sha,
        trust_published_evidence=published_frozen,
    )
    runtime_required = phase in {"dispatch", "recovery", "attest"}
    if attest_runtime_host_local and phase != "attest":
        raise ProtocolGateError(
            "--attest-runtime-host-local is restricted to the attest phase"
        )
    if runtime_required and (model is None or not infer_base_url):
        raise ProtocolGateError(
            f"runtime attestation requires model and inference endpoint for {phase}"
        )
    if infer_base_url:
        if model is None:
            raise ProtocolGateError(
                "--infer-base-url requires an exact approved --model"
            )
        verify_inference_endpoint(
            infer_base_url,
            model,
            api_key=(
                infer_api_key
                if infer_api_key is not None
                else os.environ.get("RWKV_INFER_API_KEY", "")
            ),
        )

    # The endpoint probe is an external operation. Re-read both immutable
    # contracts afterwards so a concurrent source/config/approval replacement
    # cannot be hidden inside that probe window.
    final_lock = _verify_lock(repo, lock_path)
    if final_lock.get("lock_sha256") != lock.get("lock_sha256"):
        raise ProtocolGateError("protocol lock changed during endpoint verification")
    final_approval = _verify_global_approval(
        repo,
        resolved_approval,
        locked_descriptor=final_lock.get("global_approval"),
        locked_runtime_evidence_sha256=final_lock.get(
            "runtime_attestation_evidence_sha256"
        ),
        trust_published_evidence=published_frozen,
    )
    if runtime_required:
        verifier = runtime_route_verifier or _verify_inference_runtime_route
        verifier(
            repo=repo,
            approval=final_approval,
            approval_path=resolved_approval,
            model=str(model),
            infer_base_url=str(infer_base_url),
            infer_api_key=(
                infer_api_key
                if infer_api_key is not None
                else os.environ.get("RWKV_INFER_API_KEY", "")
            ),
            local_only=attest_runtime_host_local,
        )
    print(f"global protocol gate passed: phase={phase} model={model or '-'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument(
        "--phase",
        choices=("audit", "stop", "launch", "dispatch", "recovery", "attest"),
        default="dispatch",
    )
    parser.add_argument("--model")
    parser.add_argument("--infer-base-url")
    parser.add_argument("--infer-api-key")
    parser.add_argument("--frozen-runtime", type=Path)
    parser.add_argument("--require-current-python", action="store_true")
    parser.add_argument("--attest-runtime-host-local", action="store_true")
    parser.add_argument(
        "--print-frozen-python",
        action="store_true",
        help="verify the complete frozen runtime and print its sealed Python executable",
    )
    parser.add_argument("--print-current-lock", action="store_true")
    parser.add_argument("--check-model-response", metavar="EXPECTED_MODEL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.expanduser().resolve()
    lock_path = (
        args.lock.expanduser().resolve()
        if args.lock is not None
        else repo / "ops/g1i_strict46/protocol_gate.lock.json"
    )
    try:
        if args.check_model_response:
            payload = json.load(sys.stdin)
            print(validate_single_model_response(payload, args.check_model_response))
            return 0
        if args.print_frozen_python:
            if args.frozen_runtime is None:
                raise ProtocolGateError("--print-frozen-python requires --frozen-runtime")
            approval = args.approval
            if approval is None:
                configured = os.environ.get("RWKV_GLOBAL_PROTOCOL_APPROVAL", "").strip()
                approval = Path(configured) if configured else None
            if approval is None:
                raise ProtocolGateError(
                    "--print-frozen-python requires --approval or "
                    "RWKV_GLOBAL_PROTOCOL_APPROVAL"
                )
            verified_lock = _verify_lock(repo, lock_path)
            runtime = verify_frozen_runtime(
                source_repo=repo,
                frozen_root=args.frozen_runtime,
                approval_path=approval,
                lock_path=lock_path,
            )
            frozen_root = Path(str(runtime["root"])).resolve(strict=True)
            if repo.resolve(strict=True) != frozen_root:
                raise ProtocolGateError(
                    "gate repository is not the verified frozen runtime"
                )
            _verify_global_approval(
                repo,
                approval,
                locked_descriptor=verified_lock.get("global_approval"),
                locked_runtime_evidence_sha256=verified_lock.get(
                    "runtime_attestation_evidence_sha256"
                ),
                trust_published_evidence=True,
            )
            print(runtime["python_executable"])
            return 0
        if args.print_current_lock:
            if args.approval is None:
                raise ProtocolGateError(
                    "--print-current-lock requires an already accepted content-addressed "
                    "--approval artifact"
                )
            # Validate the approval independently before allowing a deployable
            # lock to be emitted.  A normal source lock cannot self-approve.
            approval = args.approval.expanduser().resolve()
            descriptor = _path_descriptor(repo, approval)
            _verify_global_approval(
                repo,
                approval,
                locked_descriptor=descriptor,
            )
            print(
                json.dumps(
                    current_lock_payload(repo, approval_path=approval),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        require_gate(
            repo=repo,
            lock_path=lock_path,
            phase=args.phase,
            model=args.model,
            approval_path=args.approval,
            infer_base_url=args.infer_base_url,
            infer_api_key=args.infer_api_key,
            frozen_runtime=args.frozen_runtime,
            require_current_python=args.require_current_python,
            attest_runtime_host_local=args.attest_runtime_host_local,
        )
    except (OSError, ValueError, ProtocolGateError) as exc:
        print(f"global protocol gate failed: {exc}", file=sys.stderr)
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
