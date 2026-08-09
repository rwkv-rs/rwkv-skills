"""Read-only, structure-filtered full-history free-response replay audit.

The database can contain millions of historical replay rows.  This audit first
uses SQL to keep only rows whose *verification input* can be changed by the
candidate extractor:

* Strategy A rows containing a final-answer cue/box/result verb.  Python then
  performs the exact incomplete-tail and verification-window comparison.
* Strategy B/C rows with a structural stage 2, because the candidate makes
  that recovery stage authoritative and removes stage-1 reasoning from the
  verification input.

Rows are skipped only when equal outcomes can be proved from the two modules'
own judgement text, MCQ evidence, or identical math windows plus identical raw
exact-match status.  Every other row is replayed through both real scorers.
PostgreSQL is opened read-only and the only output is the requested JSON
artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib
import importlib.util
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

STRATEGIES = ("strategy_a", "strategy_b", "strategy_c")
TIMEOUT_REASONS = frozenset(
    {
        "reference_parse_timeout",
        "prediction_parse_timeout",
        "math_verify_timeout",
    }
)
TIMEOUT_RETRY_SECONDS = (15, 60)
DEFAULT_ORDER_CONSISTENCY_PROBE_ROWS = 8
METADATA_SNAPSHOT_SCHEMA_VERSION = "free-response-global-audit-metadata.v2"
DEPENDENCY_MANIFEST_SCHEMA_VERSION = "free-response-global-audit-dependencies.v1"
PART_ARTIFACT_SCHEMA_VERSION = "free-response-global-audit-part.v3"
CRITICAL_DISTRIBUTIONS = frozenset(
    {
        "math-verify",
        "sympy",
        "latex2sympy2-extended",
        "antlr4-python3-runtime",
        "psycopg",
    }
)
ANSWER_CUE_SQL_RE = (
    r"(boxed|final[[:space:]]+answer|the[[:space:]]+answer|"
    r"answer[[:space:]]*(is|:|=|should[[:space:]]+be|would[[:space:]]+be|equals?)|"
    r"(gives?|yields?|equals?|evaluates?[[:space:]]+to|comes?[[:space:]]+to)|"
    r"(corresponds?[[:space:]]+to|maps?[[:space:]]+to|equivalent[[:space:]]+to)"
    r"[[:space:]]+(the[[:space:]]+)?((answer|response)[[:space:]]+)?"
    r"(is[[:space:]]+)?(option|choice)[[:space:]]+[A-Z])"
)

# Production imports are intentionally deferred until ``main`` installs the
# frozen ``src`` importer.  Importing these at module import time would execute
# mutable worktree bytes before provenance validation.
JsonlFreeAnswerLoader: Any = None
resolve_reference_answer: Any = None
make_dataset_slug: Any = None
DATASET_ROOTS: list[Path] = []
find_dataset_file: Any = None
refresh_dataset_index: Any = None
psycopg: Any = None
dict_row: Any = None
SQL: Any = None
Literal: Any = None


def _load_psycopg() -> None:
    """Load DB dependencies in bootstrap discovery or after verification.

    The manifest-emission process is a non-result-producing provenance probe;
    inventory and worker processes call this only after reading the frozen
    manifest.  Sharing this loader keeps both phases on the exact same driver
    and optional backend selection path.
    """

    global Literal  # noqa: PLW0603
    global SQL  # noqa: PLW0603
    global dict_row  # noqa: PLW0603
    global psycopg  # noqa: PLW0603

    if psycopg is not None:
        return
    psycopg = importlib.import_module("psycopg")
    dict_row = importlib.import_module("psycopg.rows").dict_row
    sql_module = importlib.import_module("psycopg.sql")
    SQL = sql_module.SQL
    Literal = sql_module.Literal


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


def _load_verified_module(
    name: str,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> Any:
    """Execute the exact frozen bytes that were verified, with no re-read."""

    payload = _read_frozen_bytes(path, expected_sha256, label=label)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path.resolve()))
    if spec is None:
        raise RuntimeError(f"cannot create module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path.resolve())
    module.__frozen_source_sha256__ = expected_sha256
    sys.modules[name] = module
    try:
        exec(compile(payload, str(path.resolve()), "exec"), module.__dict__)  # noqa: S102
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(paths: list[Path], *, root: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted({value.resolve() for value in paths if value.is_file()}):
        label = (
            path.relative_to(root.resolve()).as_posix()
            if root is not None and path.is_relative_to(root.resolve())
            else str(path)
        )
        records.append(
            {
                "path": label,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "total_bytes": sum(int(value["bytes"]) for value in records),
        "sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest(),
        "files": records,
    }


def _distribution_digest(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "file_count": 0, "total_bytes": 0, "sha256": None}
    paths = [
        Path(distribution.locate_file(value))
        for value in (distribution.files or ())
        if not str(value).endswith((".pyc", ".pyo"))
    ]
    digest = _tree_digest(paths)
    digest["version"] = distribution.version
    return digest


def _frozen_project_root() -> Path:
    raw = os.environ.get("RWKV_AUDIT_FROZEN_PROJECT_ROOT")
    if not raw:
        raise RuntimeError("RWKV_AUDIT_FROZEN_PROJECT_ROOT is required")
    root = Path(raw).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError(
            f"audit cwd is not the frozen project root: {Path.cwd()} != {root}"
        )
    if not root.name.startswith("project-") or root.stat().st_mode & 0o222:
        raise RuntimeError(f"invalid frozen project root: {root}")
    return root


def _loaded_distribution_names() -> set[str]:
    """Resolve third-party distributions whose modules executed in this process."""

    package_map = importlib.metadata.packages_distributions()
    names: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not getattr(module, "__file__", None):
            continue
        top_level = module_name.partition(".")[0]
        names.update(package_map.get(top_level) or ())
    return names


def _dependency_environment(
    *,
    package_names: set[str] | None = None,
) -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    selected_packages = sorted(package_names or CRITICAL_DISTRIBUTIONS)
    for name in selected_packages:
        packages[name] = _distribution_digest(name)
    project_root = _frozen_project_root()
    src_root = project_root / "src"
    local_src_paths = (
        [
            path
            for path in src_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ]
        if src_root.is_dir()
        else []
    )
    project_metadata_paths = [
        path
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.is_file()
    ]
    project_contract = _tree_digest(
        [*local_src_paths, *project_metadata_paths],
        root=project_root,
    )
    if project_root.name != f"project-{project_contract['sha256']}":
        raise RuntimeError("frozen project tree is not content-addressed")
    payload = {
        "schema_version": DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "frozen_project_root": str(project_root),
        "project_contract": project_contract,
        "local_src_contract": _tree_digest(
            local_src_paths,
            root=project_root,
        ),
        "project_metadata_contract": _tree_digest(
            project_metadata_paths,
            root=project_root,
        ),
        "scoring_environment": {
            key: os.environ.get(key)
            for key in (
                "RWKV_MATH_FAST_INTEGER_MATCH",
                "RWKV_MATH_VERIFY_TIMEOUT_S",
                "LANG",
                "LC_ALL",
                "TZ",
            )
        },
    }
    payload["sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _load_scorer_runtime_dependencies(module: Any, *, label: str) -> None:
    """Exercise the scorer's deferred dependency loader before provenance freezes.

    The production scorer deliberately imports ``math_verify`` lazily.  Importing
    only the scorer module is therefore not enough to discover distributions
    loaded by its real scoring path (including transitive parser packages).
    Reuse the scorer's own loader so dependency discovery and worker execution
    cannot silently take different import paths.
    """

    loader = getattr(module, "_load_math_verify", None)
    if not callable(loader):
        raise RuntimeError(
            f"{label} has no callable deferred math-verify loader"
        )
    api = loader()
    if not isinstance(api, tuple) or len(api) != 2 or not all(
        callable(value) for value in api
    ):
        raise RuntimeError(f"{label} could not load its math-verify runtime")


def _settled_dependency_environment(
    *,
    baseline: Any,
    candidate: Any,
) -> dict[str, Any]:
    """Build a manifest only after every deferred runtime import has settled.

    This is intentionally based on modules that actually execute, rather than
    naming optional wheel implementations.  For example, importing ``psycopg``
    may select ``psycopg-binary`` on one host and a different implementation on
    another.  Whichever distribution Python loads is discovered, byte-digested,
    and then required by every inventory/worker process.
    """

    _load_psycopg()
    _load_scorer_runtime_dependencies(baseline, label="baseline scorer")
    _load_scorer_runtime_dependencies(candidate, label="candidate scorer")

    # Computing package digests should not import package code.  Still settle
    # to a fixed point so a future metadata backend that does load a module is
    # captured rather than producing an order-dependent manifest.
    package_names = set(CRITICAL_DISTRIBUTIONS) | _loaded_distribution_names()
    for _ in range(8):
        manifest = _dependency_environment(package_names=package_names)
        discovered = set(CRITICAL_DISTRIBUTIONS) | _loaded_distribution_names()
        if discovered == package_names:
            _verify_dependency_environment(manifest)
            return manifest
        package_names = discovered
    raise RuntimeError("runtime dependency discovery did not reach a fixed point")


class _FrozenSrcImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import ``src.*`` only from bytes already checked against the manifest."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root.resolve()
        contract = dict(manifest.get("local_src_contract") or {})
        records = contract.get("files") or []
        self._sources: dict[str, tuple[Path, bytes, str, bool]] = {}
        namespace_candidates: dict[str, Path] = {}
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError("invalid local src contract record")
            relative = str(record.get("path") or "")
            if not relative.startswith("src/"):
                raise RuntimeError(f"invalid local src contract path: {relative}")
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or path.stat().st_mode & 0o222:
                raise RuntimeError(f"writable/escaped frozen src module: {path}")
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(record.get("sha256") or ""):
                raise RuntimeError(f"frozen src module digest mismatch: {relative}")
            recorded_bytes = record.get("bytes")
            if recorded_bytes is None or len(payload) != int(recorded_bytes):
                raise RuntimeError(f"frozen src module size mismatch: {relative}")
            if not relative.endswith(".py"):
                # Non-Python package resources are still part of the project
                # contract and are verified above.  They are intentionally not
                # executable import targets.
                continue
            if relative.endswith("/__init__.py"):
                module_name = relative[: -len("/__init__.py")].replace("/", ".")
                is_package = True
            else:
                module_name = relative[:-3].replace("/", ".")
                is_package = False
            if module_name in self._sources:
                raise RuntimeError(f"duplicate frozen src module: {module_name}")
            self._sources[module_name] = (
                path,
                payload,
                digest,
                is_package,
            )
            relative_parts = Path(relative).parts
            directory_parts = (
                relative_parts[:-1]
                if not is_package
                else relative_parts[:-1]
            )
            for length in range(1, len(directory_parts) + 1):
                namespace_name = ".".join(directory_parts[:length])
                namespace_candidates[namespace_name] = self.root.joinpath(
                    *directory_parts[:length]
                )
        if "src" not in self._sources:
            raise RuntimeError("frozen src contract has no src package")
        self._namespaces = {
            name: path.resolve()
            for name, path in namespace_candidates.items()
            if name not in self._sources
        }

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        del path, target
        value = self._sources.get(fullname)
        if value is None:
            namespace_path = self._namespaces.get(fullname)
            if namespace_path is None:
                return None
            source_path = namespace_path
            is_package = True
        else:
            source_path, _, _, is_package = value
        spec = importlib.util.spec_from_loader(
            fullname,
            self,
            origin=str(source_path),
            is_package=is_package,
        )
        if spec is None:
            raise RuntimeError(f"cannot create frozen src spec: {fullname}")
        return spec

    def create_module(self, spec: Any) -> Any:
        del spec
        return None

    def exec_module(self, module: Any) -> None:
        namespace_path = self._namespaces.get(module.__name__)
        if namespace_path is not None:
            module.__file__ = str(namespace_path)
            module.__path__ = [str(namespace_path)]
            module.__frozen_source_sha256__ = "namespace-bound-by-project-contract"
            return
        source_path, payload, digest, is_package = self._sources[module.__name__]
        module.__file__ = str(source_path)
        module.__frozen_source_sha256__ = digest
        if is_package:
            module.__path__ = [str(source_path.parent)]
        exec(compile(payload, str(source_path), "exec"), module.__dict__)  # noqa: S102


def _install_frozen_src_importer(
    dependency_manifest: dict[str, Any],
) -> _FrozenSrcImporter:
    existing = sorted(
        name for name in sys.modules if name == "src" or name.startswith("src.")
    )
    if existing:
        raise RuntimeError(
            "src modules were imported before frozen provenance validation: "
            + ", ".join(existing[:10])
        )
    root = _frozen_project_root()
    if str(dependency_manifest.get("frozen_project_root") or "") != str(root):
        raise RuntimeError("dependency manifest frozen project root mismatch")
    importer = _FrozenSrcImporter(root, dependency_manifest)
    sys.meta_path.insert(0, importer)
    return importer


def _load_src_contract_modules(importer: _FrozenSrcImporter) -> None:
    global DATASET_ROOTS  # noqa: PLW0603
    global JsonlFreeAnswerLoader  # noqa: PLW0603
    global find_dataset_file  # noqa: PLW0603
    global make_dataset_slug  # noqa: PLW0603
    global refresh_dataset_index  # noqa: PLW0603
    global resolve_reference_answer  # noqa: PLW0603

    from src.eval.datasets.data_loader.free_answer import (
        JsonlFreeAnswerLoader as frozen_loader,
    )
    from src.eval.metrics.free_response import (
        resolve_reference_answer as frozen_resolver,
    )
    from src.eval.scheduler.dataset_utils import (
        make_dataset_slug as frozen_slugger,
    )
    from src.eval.scheduler.datasets import (
        DATASET_ROOTS as frozen_dataset_roots,
        find_dataset_file as frozen_find_dataset_file,
        refresh_dataset_index as frozen_refresh_dataset_index,
    )

    JsonlFreeAnswerLoader = frozen_loader
    resolve_reference_answer = frozen_resolver
    make_dataset_slug = frozen_slugger
    DATASET_ROOTS = frozen_dataset_roots
    find_dataset_file = frozen_find_dataset_file
    refresh_dataset_index = frozen_refresh_dataset_index
    _verify_loaded_src_modules(importer)


def _verify_loaded_src_modules(
    importer: _FrozenSrcImporter,
    modules: dict[str, Any] | None = None,
) -> None:
    module_table = sys.modules if modules is None else modules
    loaded = {
        name: module
        for name, module in module_table.items()
        if (name == "src" or name.startswith("src.")) and module is not None
    }
    if not loaded:
        raise RuntimeError("no frozen src modules were loaded")
    for name, module in loaded.items():
        expected = importer._sources.get(name)  # noqa: SLF001
        spec = getattr(module, "__spec__", None)
        if spec is None or spec.loader is not importer:
            raise RuntimeError(f"src module bypassed frozen importer: {name}")
        if expected is None:
            namespace_path = importer._namespaces.get(name)  # noqa: SLF001
            if namespace_path is None:
                raise RuntimeError(
                    f"loaded src module is absent from manifest: {name}"
                )
            if Path(str(getattr(module, "__file__", ""))).resolve() != namespace_path:
                raise RuntimeError(f"src namespace origin drifted: {name}")
            if namespace_path.stat().st_mode & 0o222:
                raise RuntimeError(f"loaded src namespace became writable: {name}")
            continue
        source_path, _, digest, _ = expected
        actual_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if actual_path != source_path:
            raise RuntimeError(f"src module origin drifted: {name}")
        if getattr(module, "__frozen_source_sha256__", None) != digest:
            raise RuntimeError(f"loaded src module bytes are unbound: {name}")
        if source_path.stat().st_mode & 0o222:
            raise RuntimeError(f"loaded frozen src module became writable: {name}")
        if _sha256_file(source_path) != digest:
            raise RuntimeError(f"loaded frozen src module changed on disk: {name}")


def _payload_from_context(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context")
    context = context if isinstance(context, dict) else {}
    payload: dict[str, Any] = {
        "sample_index": int(row["sample_index"]),
        "repeat_index": int(row["avg_repeat_index"]),
        "pass_index": int(row["pass_index"]),
        "context": context,
    }
    stages = context.get("stages")
    if isinstance(stages, list):
        for index, stage in enumerate(stages, start=1):
            if not isinstance(stage, dict):
                continue
            payload[f"prompt{index}"] = stage.get("prompt")
            payload[f"completion{index}"] = stage.get("completion")
            payload[f"stop_reason{index}"] = stage.get("stop_reason")
    # A few historical imports persisted the already-flattened representation.
    for key in (
        "prompt1",
        "completion1",
        "stop_reason1",
        "prompt2",
        "completion2",
        "stop_reason2",
    ):
        if key in context and key not in payload:
            payload[key] = context.get(key)
    strategy_a = context.get("strategy_a")
    if isinstance(strategy_a, dict):
        payload["strategy_a_prompt"] = strategy_a.get("prompt")
        payload["strategy_a_completion"] = strategy_a.get("completion")
        payload["strategy_a_stop_reason"] = strategy_a.get("stop_reason")
    for key in (
        "strategy_a_prompt",
        "strategy_a_completion",
        "strategy_a_stop_reason",
    ):
        if key in context and key not in payload:
            payload[key] = context.get(key)
    stats = context.get("stats")
    if isinstance(stats, dict):
        payload["stats"] = stats
    return payload


def _task_groups(
    connection: psycopg.Connection[Any],
) -> tuple[
    dict[int, str],
    set[int],
    dict[int, dict[str, Any]],
    dict[int, list[int]],
]:
    groups: dict[int, str] = {}
    primary_c: set[int] = set()
    family_sets: list[set[int]] = []
    score_rows = connection.execute(
        """
        select score_id, task_id, metrics
        from scores
        where metrics ? 'strategy_task_ids'
        order by score_id, task_id
        """
    ).fetchall()
    for row in score_rows:
        metrics = row["metrics"] if isinstance(row["metrics"], dict) else {}
        mapping = metrics.get("strategy_task_ids")
        if not isinstance(mapping, dict):
            continue
        root_id = int(row["task_id"])
        family = {root_id}
        for group, task_id_value in mapping.items():
            if group not in STRATEGIES:
                continue
            task_id = int(task_id_value)
            family.add(task_id)
            previous = groups.get(task_id)
            if previous is not None and previous != group:
                raise RuntimeError(
                    "conflicting strategy family assignment for task "
                    f"{task_id}: {previous} != {group}"
                )
            groups[task_id] = group
            if task_id == root_id and group == "strategy_c":
                primary_c.add(root_id)
        family_sets.append(family)

    metadata: dict[int, dict[str, Any]] = {}
    task_rows = connection.execute(
        """
        select t.task_id, t.config_path, t.evaluator, t.sampling_config,
               t.model_id,
               t.benchmark_id, t.created_at as task_created_at,
               b.benchmark_name, b.benchmark_split,
               m.model_name, m.arch_version, m.num_params
        from task t
        join benchmark b on b.benchmark_id = t.benchmark_id
        join model m on m.model_id = t.model_id
        where t.evaluator like 'free_response%'
        order by t.task_id
        """
    ).fetchall()
    for row in task_rows:
        task_id = int(row["task_id"])
        evaluator = str(row["evaluator"] or "")
        for group in STRATEGIES:
            if evaluator.endswith(f":{group}"):
                previous = groups.get(task_id)
                if previous is not None and previous != group:
                    raise RuntimeError(
                        "task evaluator conflicts with strategy family mapping for "
                        f"task {task_id}: {previous} != {group}"
                    )
                groups[task_id] = group
                break
        groups.setdefault(task_id, "strategy_a")
        metadata[task_id] = dict(row)
    # Strategy mappings can be repeated on several score rows.  Compute the
    # transitive closure so every task points at the complete A/B/C family.
    merged_families: list[set[int]] = []
    for family in family_sets:
        overlaps = [value for value in merged_families if value & family]
        combined = set(family)
        for value in overlaps:
            combined.update(value)
            merged_families.remove(value)
        merged_families.append(combined)
    task_families: dict[int, list[int]] = {
        task_id: sorted(family)
        for family in merged_families
        for task_id in family
    }
    for task_id in metadata:
        task_families.setdefault(task_id, [task_id])
    return groups, primary_c, metadata, task_families


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
    # Hash the representation that can actually survive JSON publication.
    # Without this normalization, integer task-ID keys sort numerically in
    # memory but lexicographically after json.load(), invalidating an otherwise
    # unchanged immutable snapshot.
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
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
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


def _atomic_publish_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Atomically publish a new immutable artifact without overwriting evidence."""

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


def _atomic_publish_json(path: Path, document: dict[str, Any]) -> None:
    _atomic_publish_bytes(
        path,
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
    )


def _read_frozen_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    """Read and verify one immutable artifact through a single file handle."""

    if path.is_symlink():
        raise RuntimeError(f"frozen {label} must not be a symlink: {path}")
    resolved = path.expanduser().absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {resolved}")
    flags = os.O_RDONLY
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        stat_result = os.fstat(descriptor)
        if stat_result.st_mode & 0o222:
            raise RuntimeError(f"frozen {label} is writable: {resolved}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"frozen {label} SHA mismatch: {actual} != {expected_sha256}"
        )
    if resolved.stem != expected_sha256:
        raise RuntimeError(f"frozen {label} is not content-addressed: {resolved}")
    return payload


def _verify_frozen_file(path: Path, expected_sha256: str, *, label: str) -> str:
    _read_frozen_bytes(path, expected_sha256, label=label)
    return expected_sha256


def _read_dependency_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    payload = _read_frozen_bytes(
        path,
        expected_file_sha256,
        label="dependency manifest",
    )
    document = json.loads(payload.decode("utf-8"))
    claimed = str(document.get("sha256") or "")
    actual = hashlib.sha256(
        _canonical_json_bytes({key: value for key, value in document.items() if key != "sha256"})
    ).hexdigest()
    if not claimed or claimed != actual:
        raise RuntimeError(f"dependency manifest digest mismatch: {claimed} != {actual}")
    return document


def _verify_dependency_environment(expected: dict[str, Any]) -> dict[str, Any]:
    actual = _dependency_environment(
        package_names=set(dict(expected.get("packages") or {})),
    )
    if actual != expected:
        raise RuntimeError(
            "dependency/loader contract drifted during audit: "
            f"{actual.get('sha256')} != {expected.get('sha256')}"
        )
    # Repeated after scoring: a dependency imported lazily by a scorer must
    # already be represented by byte-level provenance in the bootstrap
    # manifest, otherwise the audit fails closed.
    unexpected_loaded = _loaded_distribution_names().difference(
        dict(expected.get("packages") or {})
    )
    if unexpected_loaded:
        raise RuntimeError(
            "loaded distributions are absent from dependency manifest: "
            + ", ".join(sorted(unexpected_loaded))
        )
    return actual


def _database_identity(
    connection: psycopg.Connection[Any],
    *,
    exported_snapshot_id: str | None,
) -> dict[str, Any]:
    row = connection.execute(
        """
        select current_database() as database_name,
               (select oid from pg_database where datname = current_database()) as database_oid,
               coalesce(inet_server_addr()::text, 'local-socket') as server_address,
               inet_server_port() as server_port,
               current_setting('server_version') as server_version,
               txid_current_snapshot()::text as transaction_snapshot
        """
    ).fetchone()
    return {
        **dict(row),
        "exported_snapshot_id": exported_snapshot_id,
    }


def _dataset_key(benchmark_name: str, benchmark_split: str) -> str:
    return json.dumps(
        [benchmark_name, benchmark_split],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _freeze_dataset_file(source: Path, snapshot_root: Path) -> tuple[Path, str]:
    """Read once, then publish a content-addressed, read-only audit input."""

    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory = snapshot_root / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.jsonl"
    if target.exists():
        if target.is_symlink() or target.stat().st_mode & 0o222:
            raise RuntimeError(
                f"invalid mutable/symlink dataset snapshot: {target}"
            )
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"corrupt content-addressed dataset snapshot: {target}")
        return target.resolve(), digest

    temporary = directory / f".{digest}.tmp.{os.getpid()}.{uuid4().hex}"
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
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or target.stat().st_mode & 0o222:
                raise RuntimeError(
                    f"conflicting mutable/symlink dataset snapshot: {target}"
                )
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(
                    f"conflicting content-addressed dataset snapshot: {target}"
                )
        target.chmod(0o444)
        _fsync_directory(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.resolve(), digest


def _build_dataset_snapshot(
    metadata: dict[int, dict[str, Any]],
    *,
    snapshot_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[int, str],
    dict[int, dict[str, Any]],
]:
    refresh_dataset_index(DATASET_ROOTS)
    sources: dict[str, dict[str, Any]] = {}
    task_dataset_keys: dict[int, str] = {}
    historical_bindings: dict[int, dict[str, Any]] = {}
    for task_id, task in sorted(metadata.items()):
        benchmark_name = str(task.get("benchmark_name") or "")
        benchmark_split = str(task.get("benchmark_split") or "")
        if not benchmark_name:
            raise RuntimeError(f"task {task_id} has no benchmark_name")
        key = _dataset_key(benchmark_name, benchmark_split)
        task_dataset_keys[task_id] = key
        if key not in sources:
            slug = (
                make_dataset_slug(benchmark_name, benchmark_split)
                if benchmark_split
                else benchmark_name
            )
            dataset_path = find_dataset_file(slug, DATASET_ROOTS)
            if dataset_path is None:
                raise FileNotFoundError(
                    "current production dataset missing for "
                    f"{benchmark_name}/{benchmark_split}"
                )
            resolved_path = dataset_path.resolve()
            frozen_path, file_sha256 = _freeze_dataset_file(
                resolved_path,
                snapshot_root,
            )
            records = list(JsonlFreeAnswerLoader(str(frozen_path)))
            questions = [record.question for record in records]
            references = [resolve_reference_answer(record) for record in records]
            if not records or any(
                not isinstance(question, str) or not question
                for question in questions
            ):
                raise RuntimeError(
                    f"invalid current production dataset records: {resolved_path}"
                )
            sources[key] = {
                "snapshot_role": "current_production_dataset_snapshot",
                "benchmark_name": benchmark_name,
                "benchmark_split": benchmark_split,
                "dataset_slug": slug,
                "source_path": str(resolved_path),
                "path": str(frozen_path),
                "file_sha256": file_sha256,
                "record_count": len(records),
                "records_sha256": hashlib.sha256(
                    _canonical_json_bytes(
                        [
                            {"question": question, "reference": reference}
                            for question, reference in zip(
                                questions,
                                references,
                                strict=True,
                            )
                        ]
                    )
                ).hexdigest(),
                "questions": questions,
                "references": references,
            }
        source = sources[key]
        sampling_config = task.get("sampling_config")
        sampling_config = (
            sampling_config if isinstance(sampling_config, dict) else {}
        )
        historical_snapshot = sampling_config.get("dataset_snapshot")
        if not isinstance(historical_snapshot, dict):
            historical_bindings[task_id] = {
                "status": "unbound",
                "reason": "missing_task_dataset_snapshot",
            }
            continue
        required_matches = {
            "file_sha256": source["file_sha256"],
            "records_sha256": source["records_sha256"],
            "record_count": source["record_count"],
        }
        mismatches = {
            field: {
                "task": historical_snapshot.get(field),
                "current": current,
            }
            for field, current in required_matches.items()
            if historical_snapshot.get(field) != current
        }
        historical_bindings[task_id] = {
            "status": "bound" if not mismatches else "mismatch",
            "mismatches": mismatches,
        }
    return sources, task_dataset_keys, historical_bindings


def _build_metadata_snapshot(
    connection: psycopg.Connection[Any],
    *,
    exported_snapshot_id: str | None,
    dataset_snapshot_root: Path,
) -> dict[str, Any]:
    groups, primary_c_tasks, metadata, task_families = _task_groups(connection)
    task_counts = {
        int(row["task_id"]): int(row["rows"])
        for row in connection.execute(
            """
            select c.task_id, count(*) as rows
            from eval e
            join completions c using(completions_id)
            join task t using(task_id)
            where t.evaluator like 'free_response%'
            group by c.task_id
            order by c.task_id
            """
        ).fetchall()
    }
    (
        dataset_sources,
        task_dataset_keys,
        historical_dataset_bindings,
    ) = _build_dataset_snapshot(
        metadata,
        snapshot_root=dataset_snapshot_root,
    )
    document: dict[str, Any] = {
        "schema_version": METADATA_SNAPSHOT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "database_identity": _database_identity(
            connection,
            exported_snapshot_id=exported_snapshot_id,
        ),
        "groups": groups,
        "primary_c_tasks": sorted(primary_c_tasks),
        "metadata": metadata,
        "task_counts": task_counts,
        "task_families": task_families,
        "dataset_sources": dataset_sources,
        "task_dataset_keys": task_dataset_keys,
        "historical_dataset_bindings": historical_dataset_bindings,
        "dataset_snapshot_root": str(dataset_snapshot_root.resolve()),
    }
    document["snapshot_digest"] = _snapshot_digest(document)
    return document


def _read_metadata_cache(path: Path) -> dict[str, Any]:
    if path.stat().st_mode & 0o222:
        raise RuntimeError(f"metadata snapshot is writable: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != METADATA_SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported metadata snapshot schema: {path}")
    claimed_digest = str(document.get("snapshot_digest") or "")
    actual_digest = _snapshot_digest(document)
    if not claimed_digest or claimed_digest != actual_digest:
        raise RuntimeError(
            f"metadata snapshot digest mismatch: {claimed_digest} != {actual_digest}"
        )
    return document


def _verify_dataset_snapshot(document: dict[str, Any]) -> None:
    snapshot_root = Path(str(document.get("dataset_snapshot_root") or "")).resolve()
    if not snapshot_root.is_dir():
        raise FileNotFoundError(f"dataset snapshot root disappeared: {snapshot_root}")
    for key, source_value in dict(document.get("dataset_sources") or {}).items():
        source = dict(source_value)
        path = Path(str(source.get("path") or ""))
        if path.is_symlink():
            raise RuntimeError(f"snapshot dataset became a symlink: {key}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"snapshot dataset disappeared: {key}: {path}")
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"snapshot dataset is not read-only: {key}: {path}")
        actual_sha = _sha256_file(path)
        expected_sha = str(source.get("file_sha256") or "")
        expected_path = snapshot_root / expected_sha[:2] / f"{expected_sha}.jsonl"
        if path.resolve() != expected_path.resolve():
            raise RuntimeError(
                f"snapshot dataset is not content-addressed under its root: {key}: {path}"
            )
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"snapshot dataset SHA mismatch: {key}: {actual_sha} != {expected_sha}"
            )
        questions = source.get("questions")
        references = source.get("references")
        expected_count = int(source.get("record_count") or 0)
        if (
            not isinstance(questions, list)
            or not isinstance(references, list)
            or len(questions) != expected_count
            or len(references) != expected_count
        ):
            raise RuntimeError(f"invalid dataset snapshot record arrays: {key}")
        records_sha = hashlib.sha256(
            _canonical_json_bytes(
                [
                    {"question": question, "reference": reference}
                    for question, reference in zip(
                        questions,
                        references,
                        strict=True,
                    )
                ]
            )
        ).hexdigest()
        if records_sha != str(source.get("records_sha256") or ""):
            raise RuntimeError(f"dataset records digest mismatch: {key}")


def _current_production_record(
    document: dict[str, Any],
    *,
    task_id: int,
    sample_index: int,
    stored_reference: str,
) -> tuple[str, str, bool]:
    task_dataset_keys = dict(document.get("task_dataset_keys") or {})
    key = task_dataset_keys.get(str(task_id), task_dataset_keys.get(task_id))
    if not key:
        raise RuntimeError(
            f"task {task_id} has no current production dataset mapping"
        )
    source = dict((document.get("dataset_sources") or {}).get(key) or {})
    questions = source.get("questions")
    references = source.get("references")
    if not isinstance(questions, list) or not isinstance(references, list):
        raise RuntimeError(f"task {task_id} has an invalid dataset snapshot")
    if sample_index < 0 or sample_index >= len(questions):
        raise IndexError(
            f"task {task_id} sample_index {sample_index} outside current production dataset"
        )
    question = questions[sample_index]
    reference = references[sample_index]
    if not isinstance(question, str) or not question:
        raise RuntimeError(
            f"task {task_id} sample_index {sample_index} has no current production question"
        )
    if not isinstance(reference, str):
        raise RuntimeError(
            f"task {task_id} sample_index {sample_index} has invalid current production reference"
        )
    return question, reference, reference != stored_reference


def _verification_window(
    module: Any,
    group: str,
    payload: dict[str, Any],
    reference: str,
) -> tuple[str, str]:
    if module._is_judgement_reference(reference):
        return "judgement", module._strategy_judgement_text(group, payload)
    scoring_text = module._strategy_scoring_text(group, payload)
    recover = group == "strategy_a" or not module._has_stage(payload, 2)
    try:
        window = module._math_verify_input(
            scoring_text,
            recover_incomplete_tail=recover,
        )
    except TypeError as exc:
        # The deployed baseline predates the keyword; do not hide unrelated
        # TypeErrors raised by a newer implementation.
        if "recover_incomplete_tail" not in str(exc):
            raise
        window = module._math_verify_input(scoring_text)
    return "math", window


def _score_math(
    module: Any,
    group: str,
    payload: dict[str, Any],
    question: str,
    reference: str,
    retry_stats: Counter[str],
) -> tuple[bool, str, str, str, bool, dict[str, Any], bool, bool]:
    def score() -> tuple[Any, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = module.score_free_response_strategy(
                group,
                payload,
                sample_index=int(payload["sample_index"]),
                repeat_index=int(payload["repeat_index"]),
                question=question,
                reference=reference,
            )
        transcript = "\n".join(
            value for value in (stdout.getvalue(), stderr.getvalue()) if value
        )
        return result, transcript

    def timed_out(result: Any) -> bool:
        return str(result.fail_reason or "") in TIMEOUT_REASONS

    def outcome(result: Any) -> tuple[bool, bool, str, str]:
        return (
            bool(result.math_passed),
            bool(getattr(result, "final_passed", result.math_passed)),
            str(result.display_answer or ""),
            str(result.fail_reason or ""),
        )

    @contextlib.contextmanager
    def extended_internal_timeout(seconds: int) -> Any:
        """Make the deployed baseline's math_verify retry genuinely longer.

        Older evaluator code relies on math_verify's fixed five-second inner
        comparison timeout.  Merely extending our outer alarm therefore does
        not extend the retry.  New candidates propagate the configured bound
        themselves; only wrap historical modules that lack that helper.
        """

        if hasattr(module, "_verify_with_configured_timeout"):
            yield
            return
        original_loader = module._load_math_verify
        api = original_loader()
        if api is None:
            yield
            return
        parse, verify = api

        def extended_verify(*values: Any, **kwargs: Any) -> Any:
            kwargs["timeout_seconds"] = max(1, seconds - 1)
            return verify(*values, **kwargs)

        module._load_math_verify = lambda: (parse, extended_verify)
        try:
            yield
        finally:
            module._load_math_verify = original_loader

    # Preserve structural blank-stage semantics, then isolate exact timeout
    # outcomes.  Both extended attempts are made after an initial timeout so
    # two resolved-but-conflicting results cannot be silently accepted.
    result, transcript = score()
    attempts: list[dict[str, Any]] = [
        {
            "timeout_seconds": None,
            "passed": bool(result.math_passed),
            "final_passed": bool(
                getattr(result, "final_passed", result.math_passed)
            ),
            "answer": str(result.display_answer or ""),
            "fail_reason": str(result.fail_reason or ""),
        }
    ]
    indeterminate = False
    status = "not_retried"
    if timed_out(result):
        retry_stats["initial_timeout"] += 1
        retry_stats["attempted"] += 1
        resolved: list[tuple[tuple[bool, bool, str, str], Any, str]] = []
        for seconds in TIMEOUT_RETRY_SECONDS:
            retry_stats[f"retry_{seconds}_attempted"] += 1
            previous = os.environ.get("RWKV_MATH_VERIFY_TIMEOUT_S")
            os.environ["RWKV_MATH_VERIFY_TIMEOUT_S"] = str(seconds)
            try:
                with extended_internal_timeout(seconds):
                    retry_result, retry_transcript = score()
            finally:
                if previous is None:
                    os.environ.pop("RWKV_MATH_VERIFY_TIMEOUT_S", None)
                else:
                    os.environ["RWKV_MATH_VERIFY_TIMEOUT_S"] = previous
            retry_timed_out = timed_out(retry_result)
            attempts.append(
                {
                    "timeout_seconds": seconds,
                    "passed": bool(retry_result.math_passed),
                    "final_passed": bool(
                        getattr(
                            retry_result,
                            "final_passed",
                            retry_result.math_passed,
                        )
                    ),
                    "answer": str(retry_result.display_answer or ""),
                    "fail_reason": str(retry_result.fail_reason or ""),
                }
            )
            if retry_timed_out:
                retry_stats[f"retry_{seconds}_timeout"] += 1
            else:
                retry_stats[f"retry_{seconds}_resolved"] += 1
                resolved.append(
                    (outcome(retry_result), retry_result, retry_transcript)
                )

        unique_resolved = {value for value, _, _ in resolved}
        if len(resolved) != len(TIMEOUT_RETRY_SECONDS):
            retry_stats["unresolved"] += 1
            retry_stats["indeterminate"] += 1
            indeterminate = True
            status = (
                "unresolved_timeout"
                if not resolved
                else "partially_resolved_timeout"
            )
            if resolved:
                _, result, transcript = resolved[-1]
            else:
                result, transcript = retry_result, retry_transcript
        elif len(unique_resolved) != 1:
            retry_stats["conflicting"] += 1
            retry_stats["indeterminate"] += 1
            indeterminate = True
            status = "conflicting_resolved_outcomes"
            _, result, transcript = resolved[-1]
        else:
            retry_stats["resolved"] += 1
            status = "resolved_consistently"
            _, result, transcript = resolved[-1]
    return (
        bool(result.math_passed),
        result.display_answer,
        result.fail_reason,
        _short(transcript, limit=2_000),
        indeterminate,
        {"status": status, "attempts": attempts},
        bool(getattr(result, "judge_eligible", False)),
        bool(getattr(result, "final_passed", result.math_passed)),
    )


def _final_answer_candidates(
    module: Any,
    scoring_text: str,
    *,
    recover_incomplete_tail: bool,
) -> list[str]:
    try:
        return module._final_answer_candidates(
            scoring_text,
            recover_incomplete_tail=recover_incomplete_tail,
        )
    except TypeError as exc:
        if "recover_incomplete_tail" not in str(exc):
            raise
        return module._final_answer_candidates(scoring_text)


def _fast_integer_match(
    module: Any,
    group: str,
    payload: dict[str, Any],
    reference: str,
) -> tuple[bool, str] | None:
    scoring_text = module._strategy_scoring_text(group, payload)
    recover = group == "strategy_a" or not module._has_stage(payload, 2)
    try:
        return module._fast_integer_match(
            reference,
            scoring_text,
            recover_incomplete_tail=recover,
        )
    except TypeError as exc:
        if "recover_incomplete_tail" not in str(exc):
            raise
        return module._fast_integer_match(reference, scoring_text)


def _mcq_signature(
    module: Any,
    group: str,
    payload: dict[str, Any],
    question: str,
    reference: str,
) -> tuple[str, ...]:
    """Return the evaluator's deterministic MCQ evidence signature."""

    reference_label = module._reference_option_label(reference)
    if reference_label is None:
        return ("not_mcq",)
    options = module._parse_question_options(
        question,
        required_label=reference_label,
    )
    if not options or reference_label not in options:
        return ("not_mcq",)
    scoring_text = module._strategy_scoring_text(group, payload)
    recover = group == "strategy_a" or not module._has_stage(payload, 2)
    labels = set(options)
    try:
        predicted_label = module._explicit_option_label(
            scoring_text,
            labels,
            recover_incomplete_tail=recover,
        )
    except TypeError as exc:
        if "recover_incomplete_tail" not in str(exc):
            raise
        predicted_label = module._explicit_option_label(scoring_text, labels)
    if predicted_label is not None:
        return ("conclusive", reference_label, predicted_label)

    normalized_options = {
        label: module._comparable_option_text(value)
        for label, value in options.items()
    }
    matched_labels: set[str] = set()
    for value in _final_answer_candidates(
        module,
        scoring_text,
        recover_incomplete_tail=recover,
    ):
        normalized = module._comparable_option_text(value)
        if not normalized:
            continue
        matched_labels.update(
            label
            for label, option in normalized_options.items()
            if normalized == option
        )
    if len(matched_labels) == 1:
        return ("conclusive", reference_label, next(iter(matched_labels)))
    return ("fallback", reference_label)


def _proof_equivalent(
    baseline: Any,
    candidate: Any,
    group: str,
    payload: dict[str, Any],
    question: str,
    reference: str,
    baseline_kind: str,
    baseline_window: str,
    candidate_kind: str,
    candidate_window: str,
) -> tuple[bool, str]:
    """Prove equal outcomes without substituting a synthetic scorer."""

    baseline_judgement = baseline._is_judgement_reference(reference)
    candidate_judgement = candidate._is_judgement_reference(reference)
    if baseline_judgement != candidate_judgement:
        return False, "judgement_detection_changed"
    if baseline_judgement:
        baseline_text = baseline._strategy_judgement_text(group, payload)
        candidate_text = candidate._strategy_judgement_text(group, payload)
        return baseline_text == candidate_text, "identical_judgement_text"

    baseline_signature = _mcq_signature(
        baseline,
        group,
        payload,
        question,
        reference,
    )
    candidate_signature = _mcq_signature(
        candidate,
        group,
        payload,
        question,
        reference,
    )
    if (
        baseline_signature[0] == "conclusive"
        and baseline_signature == candidate_signature
    ):
        return True, "identical_conclusive_mcq_label"
    if baseline_signature != candidate_signature:
        return False, "mcq_signature_changed"

    baseline_raw = baseline._strategy_scoring_text(group, payload)
    candidate_raw = candidate._strategy_scoring_text(group, payload)
    exact_status_equal = baseline._is_exact_match(
        baseline_raw, reference
    ) == candidate._is_exact_match(candidate_raw, reference)
    windows_equal = (baseline_kind, baseline_window) == (
        candidate_kind,
        candidate_window,
    )
    if exact_status_equal and windows_equal:
        fast_integer_enabled = baseline._env_flag(
            "RWKV_MATH_FAST_INTEGER_MATCH"
        ) or candidate._env_flag("RWKV_MATH_FAST_INTEGER_MATCH")
        if fast_integer_enabled:
            baseline_fast = _fast_integer_match(
                baseline,
                group,
                payload,
                reference,
            )
            candidate_fast = _fast_integer_match(
                candidate,
                group,
                payload,
                reference,
            )
            if baseline_fast != candidate_fast:
                return False, "fast_integer_signature_changed"
        return True, "identical_math_window_and_raw_exact_status"
    return False, "requires_real_scorer"


def _short(value: str, limit: int = 600) -> str:
    flattened = " ".join(value.split())
    return flattened if len(flattened) <= limit else f"{flattened[: limit - 3]}..."


def _score_cache_key(
    implementation: str,
    group: str,
    reference: str,
    question: str,
    payload: dict[str, Any],
    metadata_snapshot_digest: str,
) -> tuple[str, ...]:
    """Hash every scorer input without retaining full prompts/completions."""

    digest = hashlib.sha256
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return (
        implementation,
        group,
        digest(reference.encode("utf-8")).hexdigest(),
        digest(question.encode("utf-8")).hexdigest(),
        digest(serialized_payload.encode("utf-8")).hexdigest(),
        metadata_snapshot_digest,
    )


def _explain_one_to_zero(
    candidate: Any,
    group: str,
    payload: dict[str, Any],
    baseline_window: str,
    candidate_window: str,
) -> str:
    if group in {"strategy_b", "strategy_c"} and candidate._has_stage(payload, 2):
        return "authoritative_stage2_boundary_prevents_stage1_answer_inheritance"
    scoring_text = candidate._strategy_scoring_text(group, payload)
    candidate_list = candidate._explicit_answer_candidates(scoring_text)
    if candidate_list:
        if "?" in candidate_window or "\uff1f" in candidate_window:
            return "later_questioned_answer_is_fail_closed"
        if baseline_window != candidate_window:
            if (
                candidate._last_boxed_content(candidate_window) is not None
                and not candidate._tail_is_syntactically_incomplete(scoring_text)
            ):
                return "authoritative_terminal_box_replaces_whole_completion_parser"
            return "incomplete_tail_recovered_to_latest_complete_answer_evidence"
    return "UNEXPLAINED"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=Path, default=Path("/home/rwkv/chase/rwkv-skills/.env")
    )
    parser.add_argument("--database", default="chase_rwkv_skills")
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--candidate-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-audit-script-sha256", required=True)
    parser.add_argument("--expected-baseline-module-sha256", required=True)
    parser.add_argument("--expected-candidate-module-sha256", required=True)
    parser.add_argument("--dependency-manifest", type=Path)
    parser.add_argument("--expected-dependency-manifest-sha256")
    parser.add_argument("--emit-dependency-manifest", type=Path)
    parser.add_argument(
        "--order-probe-mode",
        choices=("baseline_then_candidate", "candidate_then_baseline"),
    )
    parser.add_argument("--fetch-size", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument("--dataset-snapshot-root", type=Path)
    parser.add_argument("--dataset-source-root", type=Path)
    parser.add_argument(
        "--refresh-metadata-snapshot",
        action="store_true",
        help="rebuild and atomically replace the immutable metadata/dataset snapshot",
    )
    parser.add_argument(
        "--database-snapshot-id",
        help="PostgreSQL exported snapshot held open by the partition launcher",
    )
    parser.add_argument(
        "--metadata-snapshot-digest",
        help="required digest of the shared immutable metadata snapshot",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--order-consistency-probe-rows",
        type=int,
        default=DEFAULT_ORDER_CONSISTENCY_PROBE_ROWS,
    )
    parser.add_argument("--max-structural-rows", type=int)
    parser.add_argument(
        "--full-scan-a",
        action="store_true",
        help="scan every selected Strategy-A row instead of using the SQL cue filter",
    )
    parser.add_argument(
        "--full-real-scorer",
        action="store_true",
        help="run both real scorers for every selected row; do not proof-skip rows",
    )
    parser.add_argument("--prove-a-superset", action="store_true")
    parser.add_argument("--max-a-superset-proof-rows", type=int)
    parser.add_argument(
        "--groups",
        default=",".join(STRATEGIES),
        help="comma-separated strategy_a,strategy_b,strategy_c",
    )
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    return parser.parse_args()


def _verify_code_provenance(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "audit_script_sha256": _verify_frozen_file(
            Path(__file__),
            args.expected_audit_script_sha256,
            label="audit script",
        ),
        "baseline_module_sha256": _verify_frozen_file(
            args.baseline_module,
            args.expected_baseline_module_sha256,
            label="baseline module",
        ),
        "candidate_module_sha256": _verify_frozen_file(
            args.candidate_module,
            args.expected_candidate_module_sha256,
            label="candidate module",
        ),
    }
    return values


def _score_signature(result: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "math_passed": bool(result[0]),
        "final_passed": bool(result[7]),
        "answer": str(result[1] or ""),
        "fail_reason": str(result[2] or ""),
        "indeterminate": bool(result[4]),
        "timeout_resolution": result[5],
        "judge_eligible": bool(result[6]),
    }


def _judge_request(
    *,
    final_passed: bool,
    judge_eligible: bool,
    question: str,
    reference: str,
    answer: str,
) -> tuple[str, str, str] | None:
    if final_passed or not judge_eligible:
        return None
    return question, reference, answer


def _replay_tasks_for_change(
    *,
    group: str,
    task_id: int,
    task_families: dict[int, list[int]],
) -> list[int]:
    if group == "strategy_a":
        return task_families.get(task_id, [task_id])
    return [task_id]


def _order_probe_child(
    args: argparse.Namespace,
    dependency_manifest: dict[str, Any],
    importer: _FrozenSrcImporter,
) -> None:
    request = json.loads(sys.stdin.read())
    if args.order_probe_mode == "baseline_then_candidate":
        baseline = _load_verified_module(
            "order_probe_baseline",
            args.baseline_module,
            args.expected_baseline_module_sha256,
            label="baseline module",
        )
        candidate = _load_verified_module(
            "order_probe_candidate",
            args.candidate_module,
            args.expected_candidate_module_sha256,
            label="candidate module",
        )
        order = ("baseline", "candidate")
    else:
        candidate = _load_verified_module(
            "order_probe_candidate",
            args.candidate_module,
            args.expected_candidate_module_sha256,
            label="candidate module",
        )
        baseline = _load_verified_module(
            "order_probe_baseline",
            args.baseline_module,
            args.expected_baseline_module_sha256,
            label="baseline module",
        )
        order = ("candidate", "baseline")
    modules = {"baseline": baseline, "candidate": candidate}
    results: dict[str, Any] = {}
    timeout_stats: dict[str, dict[str, int]] = {}
    for implementation in order:
        stats: Counter[str] = Counter()
        result = _score_math(
            modules[implementation],
            str(request["group"]),
            dict(request["payload"]),
            str(request["question"]),
            str(request["reference"]),
            stats,
        )
        results[implementation] = _score_signature(result)
        timeout_stats[implementation] = dict(stats)
    _verify_dependency_environment(dependency_manifest)
    _verify_loaded_src_modules(importer)
    _verify_code_provenance(args)
    print(
        json.dumps(
            {
                "order": list(order),
                "results": results,
                "timeout_stats": timeout_stats,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _run_isolated_order_probe(
    args: argparse.Namespace,
    *,
    mode: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    frozen_project_root = _frozen_project_root()
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = str(frozen_project_root)
    child_environment["RWKV_AUDIT_FROZEN_PROJECT_ROOT"] = str(
        frozen_project_root
    )
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--baseline-module",
        str(args.baseline_module),
        "--candidate-module",
        str(args.candidate_module),
        "--output",
        "/dev/null",
        "--expected-audit-script-sha256",
        args.expected_audit_script_sha256,
        "--expected-baseline-module-sha256",
        args.expected_baseline_module_sha256,
        "--expected-candidate-module-sha256",
        args.expected_candidate_module_sha256,
        "--dependency-manifest",
        str(args.dependency_manifest),
        "--expected-dependency-manifest-sha256",
        str(args.expected_dependency_manifest_sha256),
        "--order-probe-mode",
        mode,
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
        timeout=360,
        cwd=frozen_project_root,
        env=child_environment,
    )
    lines = [value for value in completed.stdout.splitlines() if value.strip()]
    if not lines:
        raise RuntimeError("isolated order probe produced no result")
    return json.loads(lines[-1])


def main() -> None:  # noqa: C901, PLR0915
    args = _parse_args()
    code_provenance = _verify_code_provenance(args)
    if args.emit_dependency_manifest is not None:
        bootstrap_manifest = _dependency_environment()
        importer = _install_frozen_src_importer(bootstrap_manifest)
        _load_src_contract_modules(importer)
        baseline = _load_verified_module(
            "manifest_baseline",
            args.baseline_module,
            args.expected_baseline_module_sha256,
            label="baseline module",
        )
        candidate = _load_verified_module(
            "manifest_candidate",
            args.candidate_module,
            args.expected_candidate_module_sha256,
            label="candidate module",
        )
        dependency_manifest = _settled_dependency_environment(
            baseline=baseline,
            candidate=candidate,
        )
        _verify_loaded_src_modules(importer)
        _verify_dependency_environment(dependency_manifest)
        _atomic_publish_json(args.emit_dependency_manifest, dependency_manifest)
        _verify_code_provenance(args)
        return
    if args.dependency_manifest is None or not args.expected_dependency_manifest_sha256:
        raise ValueError("a frozen dependency manifest is required")
    dependency_manifest = _read_dependency_manifest(
        args.dependency_manifest,
        args.expected_dependency_manifest_sha256,
    )
    _verify_dependency_environment(dependency_manifest)
    importer = _install_frozen_src_importer(dependency_manifest)
    _load_src_contract_modules(importer)
    if args.order_probe_mode is not None:
        _order_probe_child(args, dependency_manifest, importer)
        return
    baseline = _load_verified_module(
        "baseline_free_response_global",
        args.baseline_module,
        args.expected_baseline_module_sha256,
        label="baseline module",
    )
    candidate = _load_verified_module(
        "candidate_free_response_global",
        args.candidate_module,
        args.expected_candidate_module_sha256,
        label="candidate module",
    )
    _verify_loaded_src_modules(importer)
    _verify_dependency_environment(dependency_manifest)
    if args.order_consistency_probe_rows < 0:
        raise ValueError("order-consistency-probe-rows cannot be negative")
    if args.dataset_source_root is None:
        raise ValueError("--dataset-source-root is required")
    dataset_source_root = args.dataset_source_root.resolve()
    if not dataset_source_root.is_dir():
        raise FileNotFoundError(f"dataset source root is missing: {dataset_source_root}")
    DATASET_ROOTS[:] = [dataset_source_root]
    refresh_dataset_index(DATASET_ROOTS)
    _load_psycopg()
    # Detect an update/replacement racing the deferred import before using the
    # driver to import the exported PostgreSQL snapshot.
    _verify_dependency_environment(dependency_manifest)
    env = _load_env(args.env)
    connection = psycopg.connect(
        _connection_string(env, args.database), row_factory=dict_row
    )
    connection.execute(
        "set transaction isolation level repeatable read, read only"
    )
    if args.database_snapshot_id:
        connection.execute(
            SQL("set transaction snapshot {}").format(
                Literal(args.database_snapshot_id)
            )
        )
    started = time.monotonic()
    with connection:
        dataset_snapshot_root = args.dataset_snapshot_root or (
            args.metadata_cache.parent / f"{args.metadata_cache.name}.datasets"
            if args.metadata_cache is not None
            else args.output.parent / f".{args.output.name}.datasets"
        )
        if args.refresh_metadata_snapshot:
            if args.metadata_cache is None:
                raise ValueError(
                    "--refresh-metadata-snapshot requires --metadata-cache"
                )
            metadata_snapshot = _build_metadata_snapshot(
                connection,
                exported_snapshot_id=args.database_snapshot_id,
                dataset_snapshot_root=dataset_snapshot_root,
            )
            _atomic_publish_json(args.metadata_cache, metadata_snapshot)
        elif args.metadata_cache is not None and args.metadata_cache.exists():
            metadata_snapshot = _read_metadata_cache(args.metadata_cache)
        else:
            if args.metadata_snapshot_digest:
                raise FileNotFoundError(
                    "shared metadata snapshot is required but missing"
                )
            metadata_snapshot = _build_metadata_snapshot(
                connection,
                exported_snapshot_id=args.database_snapshot_id,
                dataset_snapshot_root=dataset_snapshot_root,
            )
        metadata_snapshot_digest = str(
            metadata_snapshot.get("snapshot_digest") or ""
        )
        if (
            args.metadata_snapshot_digest
            and metadata_snapshot_digest != args.metadata_snapshot_digest
        ):
            raise RuntimeError(
                "stale metadata snapshot: "
                f"{metadata_snapshot_digest} != {args.metadata_snapshot_digest}"
            )
        snapshot_identity = dict(
            metadata_snapshot.get("database_identity") or {}
        )
        live_identity = _database_identity(
            connection,
            exported_snapshot_id=args.database_snapshot_id,
        )
        identity_keys = (
            "database_name",
            "database_oid",
            "server_address",
            "server_port",
            "server_version",
            "transaction_snapshot",
            "exported_snapshot_id",
        )
        identity_mismatches = {
            key: {
                "snapshot": snapshot_identity.get(key),
                "live": live_identity.get(key),
            }
            for key in identity_keys
            if snapshot_identity.get(key) != live_identity.get(key)
        }
        if identity_mismatches:
            raise RuntimeError(
                f"database snapshot identity mismatch: {identity_mismatches}"
            )
        _verify_dataset_snapshot(metadata_snapshot)
        groups = {
            int(key): str(value)
            for key, value in dict(metadata_snapshot["groups"]).items()
        }
        primary_c_tasks = {
            int(value) for value in metadata_snapshot["primary_c_tasks"]
        }
        metadata = {
            int(key): dict(value)
            for key, value in dict(metadata_snapshot["metadata"]).items()
        }
        task_counts = {
            int(key): int(value)
            for key, value in dict(metadata_snapshot["task_counts"]).items()
        }
        task_families = {
            int(key): [int(item) for item in value]
            for key, value in dict(
                metadata_snapshot.get("task_families") or {}
            ).items()
            if isinstance(value, list)
        }
        historical_binding_counts = Counter(
            str(value.get("status") or "unbound")
            for value in dict(
                metadata_snapshot.get("historical_dataset_bindings") or {}
            ).values()
            if isinstance(value, dict)
        )
        strategy_totals: Counter[str] = Counter()
        for task_id, count in task_counts.items():
            group = groups.get(task_id, "strategy_a")
            strategy_totals[group] += count
            if task_id in primary_c_tasks and group == "strategy_c":
                strategy_totals["primary_c"] += count
        inventory = {
            "database": args.database,
            "database_rows": sum(task_counts.values()),
            "tasks": len(task_counts),
            "strategy_totals": dict(strategy_totals),
            "primary_c_tasks": len(primary_c_tasks),
            "task_inventory": {
                str(task_id): {
                    "group": groups.get(task_id, "strategy_a"),
                    "rows": task_counts.get(task_id, 0),
                }
                for task_id in sorted(groups)
            },
            "task_inventory_digest": hashlib.sha256(
                _canonical_json_bytes(
                    {
                        str(task_id): {
                            "group": groups.get(task_id, "strategy_a"),
                            "rows": task_counts.get(task_id, 0),
                        }
                        for task_id in sorted(groups)
                    }
                )
            ).hexdigest(),
            "task_families": {
                str(task_id): family
                for task_id, family in sorted(task_families.items())
            },
            "metadata_snapshot": {
                "schema_version": metadata_snapshot.get("schema_version"),
                "generated_at_utc": metadata_snapshot.get("generated_at_utc"),
                "digest": metadata_snapshot_digest,
                "database_identity": snapshot_identity,
                "dataset_count": len(
                    metadata_snapshot.get("dataset_sources") or {}
                ),
                "dataset_snapshot_root": metadata_snapshot.get(
                    "dataset_snapshot_root"
                ),
                "dataset_digests": {
                    key: {
                        "file_sha256": value.get("file_sha256"),
                        "records_sha256": value.get("records_sha256"),
                        "record_count": value.get("record_count"),
                    }
                    for key, value in dict(
                        metadata_snapshot.get("dataset_sources") or {}
                    ).items()
                    if isinstance(value, dict)
                },
                "task_inventory_digest": hashlib.sha256(
                    _canonical_json_bytes(
                        {
                            str(task_id): {
                                "group": groups.get(task_id, "strategy_a"),
                                "rows": task_counts.get(task_id, 0),
                            }
                            for task_id in sorted(groups)
                        }
                    )
                ).hexdigest(),
                "task_families_digest": hashlib.sha256(
                    _canonical_json_bytes(
                        {
                            str(task_id): family
                            for task_id, family in sorted(task_families.items())
                        }
                    )
                ).hexdigest(),
            },
            "historical_generation_provenance": {
                "status": (
                    "bound"
                    if historical_binding_counts
                    and not (
                        historical_binding_counts["unbound"]
                        or historical_binding_counts["mismatch"]
                    )
                    else "unbound_or_mismatched"
                ),
                "task_counts": dict(historical_binding_counts),
                "note": (
                    "The deployment regression replays historical completion payloads "
                    "against the frozen current-production dataset. It does not claim "
                    "to reproduce an unbound historical generation dataset."
                ),
            },
        }
        if args.inventory_only:
            _verify_dependency_environment(dependency_manifest)
            _verify_loaded_src_modules(importer)
            _verify_code_provenance(args)
            print(json.dumps(inventory, ensure_ascii=False, indent=2))
            return

        requested_groups = {
            value.strip() for value in args.groups.split(",") if value.strip()
        }
        unsupported = requested_groups.difference(STRATEGIES)
        if unsupported:
            raise ValueError(f"unsupported strategy groups: {sorted(unsupported)}")
        if args.partitions < 1 or not 0 <= args.partition_index < args.partitions:
            raise ValueError("partition-index must be in [0, partitions)")
        def selected_task(task_id: int) -> bool:
            return task_id % args.partitions == args.partition_index
        a_task_ids = sorted(
            task_id
            for task_id, group in groups.items()
            if group == "strategy_a"
            and group in requested_groups
            and selected_task(task_id)
        )
        bc_task_ids = sorted(
            task_id
            for task_id, group in groups.items()
            if group in {"strategy_b", "strategy_c"}
            and group in requested_groups
            and selected_task(task_id)
        )
        if args.full_scan_a:
            sql = """
                select e.eval_id, c.completions_id, c.task_id, c.sample_index,
                       c.avg_repeat_index, c.pass_index, c.context,
                       e.answer, e.ref_answer, e.is_passed, e.fail_reason
                from completions c
                join eval e using(completions_id)
                where c.task_id = any(%s) or c.task_id = any(%s)
                order by c.task_id, c.completions_id, e.eval_id
            """
            sql_params: tuple[Any, ...] = (a_task_ids, bc_task_ids)
        else:
            sql = """
                select e.eval_id, c.completions_id, c.task_id, c.sample_index,
                       c.avg_repeat_index, c.pass_index, c.context,
                       e.answer, e.ref_answer, e.is_passed, e.fail_reason
                from completions c
                join eval e using(completions_id)
                where (
                    c.task_id = any(%s)
                    and (
                        coalesce(c.context #>> '{strategy_a,completion}', '') ~* %s
                        or coalesce(c.context #>> '{stages,0,completion}', '') ~* %s
                        or coalesce(c.context ->> 'strategy_a_completion', '') ~* %s
                        or coalesce(c.context ->> 'completion1', '') ~* %s
                    )
                ) or c.task_id = any(%s)
                order by c.task_id, c.completions_id, e.eval_id
            """
            sql_params = (
                a_task_ids,
                ANSWER_CUE_SQL_RE,
                ANSWER_CUE_SQL_RE,
                ANSWER_CUE_SQL_RE,
                ANSWER_CUE_SQL_RE,
                bc_task_ids,
            )
        cursor = connection.cursor(name="free_response_global_structural_audit")
        cursor.itersize = args.fetch_size
        cursor.execute(sql, sql_params)

        scanned = 0
        changed_windows = 0
        proof_equivalent_rows = 0
        proof_equivalent_reasons: Counter[str] = Counter()
        proof_equivalent_by_strategy: Counter[str] = Counter()
        real_scorer_reasons: Counter[str] = Counter()
        judgement_rows = 0
        stored_noncomparable_rows = 0
        stored_reference_drift_rows = 0
        stored_reference_drift_by_task: Counter[int] = Counter()
        deterministic_surface_changed_rows = 0
        judge_input_affected_rows = 0
        judge_input_affected_by_task: Counter[int] = Counter()
        replay_affected_rows = 0
        replay_affected_by_task: Counter[int] = Counter()
        replay_affected_reasons: Counter[str] = Counter()
        real_scorer_rows = 0
        full_candidate_scores = 0
        full_baseline_scores = 0
        scoring_errors = 0
        scanned_by_task: Counter[int] = Counter()
        order_consistency_probed_rows = 0
        order_consistency_conflicts: list[dict[str, Any]] = []
        order_probe_timeout_stats: Counter[str] = Counter()
        timeout_stats_by_implementation: dict[str, Counter[str]] = {
            "baseline": Counter(),
            "candidate": Counter(),
        }
        indeterminate_rows = 0
        transitions: Counter[str] = Counter()
        stored_transitions: Counter[str] = Counter()
        transition_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
        stored_transition_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
        task_deltas: dict[int, Counter[str]] = defaultdict(Counter)
        semantic_fingerprints: dict[str, str] = {}
        primary_c_semantic_fingerprints: dict[str, str] = {}
        semantic_transitions: Counter[str] = Counter()
        semantic_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
        changes: list[dict[str, Any]] = []
        score_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        a_superset_proof_scanned = 0
        a_superset_proof_errors = 0
        a_superset_violations: list[dict[str, Any]] = []

        for row in cursor:
            if (
                args.max_structural_rows is not None
                and scanned >= args.max_structural_rows
            ):
                break
            scanned += 1
            task_id = int(row["task_id"])
            group = groups.get(task_id, "strategy_a")
            meta = metadata.get(task_id, {})
            evaluator = str(meta.get("evaluator") or "")
            stored_comparable = not evaluator.startswith("free_response_judge")
            is_primary_c = task_id in primary_c_tasks and group == "strategy_c"
            payload = _payload_from_context(row)
            stored_reference = str(row["ref_answer"] or "")
            question, reference, reference_drift = _current_production_record(
                metadata_snapshot,
                task_id=task_id,
                sample_index=int(row["sample_index"]),
                stored_reference=stored_reference,
            )
            scanned_by_task[task_id] += 1
            if reference_drift:
                stored_reference_drift_rows += 1
                stored_reference_drift_by_task[task_id] += 1
                stored_comparable = False
            if baseline._is_judgement_reference(
                reference
            ) or candidate._is_judgement_reference(reference):
                judgement_rows += 1
            if not stored_comparable:
                stored_noncomparable_rows += 1
            try:
                baseline_kind, baseline_window = _verification_window(
                    baseline, group, payload, reference
                )
                candidate_kind, candidate_window = _verification_window(
                    candidate, group, payload, reference
                )
            except Exception as exc:  # noqa: BLE001
                scoring_errors += 1
                changes.append(
                    {
                        "task_id": task_id,
                        "completion_id": int(row["completions_id"]),
                        "group": group,
                        "error": f"window:{type(exc).__name__}:{exc}",
                    }
                )
                continue
            if (baseline_kind, baseline_window) != (
                candidate_kind,
                candidate_window,
            ):
                changed_windows += 1
            stored_passed = bool(row["is_passed"])
            if args.full_real_scorer:
                equivalent = False
                equivalence_reason = "forced_full_real_scorer"
            else:
                try:
                    equivalent, equivalence_reason = _proof_equivalent(
                        baseline,
                        candidate,
                        group,
                        payload,
                        question,
                        reference,
                        baseline_kind,
                        baseline_window,
                        candidate_kind,
                        candidate_window,
                    )
                except Exception as exc:  # noqa: BLE001
                    scoring_errors += 1
                    changes.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "group": group,
                            "error": f"equivalence:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
            if equivalent:
                proof_equivalent_rows += 1
                proof_equivalent_reasons[equivalence_reason] += 1
                proof_equivalent_by_strategy[group] += 1
                if is_primary_c:
                    proof_equivalent_by_strategy["primary_c"] += 1
                continue
            real_scorer_rows += 1
            real_scorer_reasons[equivalence_reason] += 1
            candidate_cache_key = _score_cache_key(
                "candidate",
                group,
                reference,
                question,
                payload,
                metadata_snapshot_digest,
            )
            candidate_result = (
                None
                if args.full_real_scorer
                else score_cache.get(candidate_cache_key)
            )
            if candidate_result is None:
                try:
                    candidate_result = _score_math(
                        candidate,
                        group,
                        payload,
                        question,
                        reference,
                        timeout_stats_by_implementation["candidate"],
                    )
                    full_candidate_scores += 1
                except Exception as exc:  # noqa: BLE001
                    scoring_errors += 1
                    changes.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "group": group,
                            "error": f"score:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                if not args.full_real_scorer:
                    score_cache[candidate_cache_key] = candidate_result
            (
                candidate_passed,
                candidate_answer,
                candidate_reason,
                candidate_transcript,
                candidate_indeterminate,
                candidate_timeout_resolution,
                candidate_judge_eligible,
                candidate_final_passed,
            ) = candidate_result
            baseline_cache_key = _score_cache_key(
                "baseline",
                group,
                reference,
                question,
                payload,
                metadata_snapshot_digest,
            )
            baseline_result = (
                None
                if args.full_real_scorer
                else score_cache.get(baseline_cache_key)
            )
            if baseline_result is None:
                try:
                    baseline_result = _score_math(
                        baseline,
                        group,
                        payload,
                        question,
                        reference,
                        timeout_stats_by_implementation["baseline"],
                    )
                    full_baseline_scores += 1
                except Exception as exc:  # noqa: BLE001
                    scoring_errors += 1
                    changes.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "group": group,
                            "error": f"baseline_recheck:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                if not args.full_real_scorer:
                    score_cache[baseline_cache_key] = baseline_result
            (
                baseline_passed,
                baseline_answer,
                baseline_reason,
                baseline_transcript,
                baseline_indeterminate,
                baseline_timeout_resolution,
                baseline_judge_eligible,
                baseline_final_passed,
            ) = baseline_result
            if candidate_indeterminate or baseline_indeterminate:
                indeterminate_rows += 1
                scoring_errors += 1
                changes.append(
                    {
                        "task_id": task_id,
                        "completion_id": int(row["completions_id"]),
                        "group": group,
                        "error": "INDETERMINATE_TIMEOUT",
                        "baseline_fail_reason": baseline_reason,
                        "candidate_fail_reason": candidate_reason,
                        "baseline_timeout_resolution": baseline_timeout_resolution,
                        "candidate_timeout_resolution": candidate_timeout_resolution,
                        "baseline_scorer_output": baseline_transcript,
                        "candidate_scorer_output": candidate_transcript,
                        "baseline_transcript": baseline_transcript,
                        "candidate_transcript": candidate_transcript,
                    }
                )
                continue
            if (
                order_consistency_probed_rows
                < args.order_consistency_probe_rows
            ):
                order_consistency_probed_rows += 1
                try:
                    request = {
                        "group": group,
                        "payload": payload,
                        "question": question,
                        "reference": reference,
                    }
                    candidate_first = _run_isolated_order_probe(
                        args,
                        mode="candidate_then_baseline",
                        request=request,
                    )
                    baseline_first = _run_isolated_order_probe(
                        args,
                        mode="baseline_then_candidate",
                        request=request,
                    )
                    for result in (candidate_first, baseline_first):
                        for stats in dict(result.get("timeout_stats") or {}).values():
                            if isinstance(stats, dict):
                                order_probe_timeout_stats.update(
                                    {str(key): int(value) for key, value in stats.items()}
                                )
                    main_signatures = {
                        "baseline": _score_signature(baseline_result),
                        "candidate": _score_signature(candidate_result),
                    }
                    candidate_first_signatures = dict(
                        candidate_first.get("results") or {}
                    )
                    baseline_first_signatures = dict(
                        baseline_first.get("results") or {}
                    )
                    if not (
                        main_signatures
                        == candidate_first_signatures
                        == baseline_first_signatures
                    ):
                        order_consistency_conflicts.append(
                            {
                                "task_id": task_id,
                                "completion_id": int(row["completions_id"]),
                                "main_candidate_then_baseline": main_signatures,
                                "isolated_candidate_then_baseline": candidate_first_signatures,
                                "isolated_baseline_then_candidate": baseline_first_signatures,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    order_consistency_conflicts.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
            transition = f"{int(baseline_passed)}->{int(candidate_passed)}"
            baseline_judge_input = _judge_request(
                final_passed=baseline_final_passed,
                judge_eligible=baseline_judge_eligible,
                question=question,
                reference=reference,
                answer=str(baseline_answer or ""),
            )
            candidate_judge_input = _judge_request(
                final_passed=candidate_final_passed,
                judge_eligible=candidate_judge_eligible,
                question=question,
                reference=reference,
                answer=str(candidate_answer or ""),
            )
            baseline_would_judge = baseline_judge_input is not None
            candidate_would_judge = candidate_judge_input is not None
            deterministic_surface_changed = (
                baseline_passed != candidate_passed
                or baseline_final_passed != candidate_final_passed
                or str(baseline_answer or "") != str(candidate_answer or "")
                or str(baseline_reason or "") != str(candidate_reason or "")
                or baseline_judge_eligible != candidate_judge_eligible
            )
            judge_input_affected = baseline_judge_input != candidate_judge_input
            if deterministic_surface_changed:
                deterministic_surface_changed_rows += 1
            if judge_input_affected:
                judge_input_affected_rows += 1
                judge_input_affected_by_task[task_id] += 1
            replay_affected = deterministic_surface_changed or judge_input_affected
            if replay_affected:
                replay_affected_rows += 1
                reasons: list[str] = []
                if baseline_passed != candidate_passed:
                    reasons.append("math_passed_changed")
                if baseline_final_passed != candidate_final_passed:
                    reasons.append("final_passed_changed")
                if str(baseline_answer or "") != str(candidate_answer or ""):
                    reasons.append("display_answer_changed")
                if str(baseline_reason or "") != str(candidate_reason or ""):
                    reasons.append("fail_reason_changed")
                if baseline_judge_eligible != candidate_judge_eligible:
                    reasons.append("judge_eligible_changed")
                if judge_input_affected:
                    reasons.append("judge_route_or_input_changed")
                replay_affected_reasons.update(reasons)
                affected_tasks = _replay_tasks_for_change(
                    group=group,
                    task_id=task_id,
                    task_families=task_families,
                )
                for affected_task_id in affected_tasks:
                    replay_affected_by_task[affected_task_id] += 1
            stored_transition = (
                f"{int(stored_passed)}->{int(candidate_final_passed)}"
                if stored_comparable
                else None
            )
            transitions[transition] += 1
            transition_by_strategy[group][transition] += 1
            if stored_transition is not None:
                stored_transitions[stored_transition] += 1
                stored_transition_by_strategy[group][stored_transition] += 1
            task_deltas[task_id][transition] += 1
            if is_primary_c:
                transition_by_strategy["primary_c"][transition] += 1
                if stored_transition is not None:
                    stored_transition_by_strategy["primary_c"][
                        stored_transition
                    ] += 1

            semantic_key = (
                meta.get("model_id"),
                meta.get("benchmark_id"),
                group,
                int(row["sample_index"]),
                int(row["avg_repeat_index"]),
                int(row["pass_index"]),
                reference,
                hashlib.sha256(question.encode("utf-8")).hexdigest(),
                metadata_snapshot_digest,
                hashlib.sha256(baseline_window.encode("utf-8")).hexdigest(),
                hashlib.sha256(candidate_window.encode("utf-8")).hexdigest(),
            )
            semantic_fingerprint = hashlib.sha256(
                json.dumps(
                    semantic_key,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            semantic_duplicate = semantic_fingerprint in semantic_fingerprints
            if not semantic_duplicate:
                semantic_fingerprints[semantic_fingerprint] = (
                    f"{group}|{transition}"
                )
                semantic_transitions[transition] += 1
                semantic_by_strategy[group][transition] += 1
            elif not semantic_fingerprints[semantic_fingerprint].endswith(
                transition
            ):
                scoring_errors += 1
                changes.append(
                    {
                        "task_id": task_id,
                        "completion_id": int(row["completions_id"]),
                        "group": group,
                        "error": "canonical_fingerprint_transition_conflict",
                        "fingerprint": semantic_fingerprint,
                        "previous": semantic_fingerprints[semantic_fingerprint],
                        "current": transition,
                    }
                )
            if is_primary_c and semantic_fingerprint not in primary_c_semantic_fingerprints:
                primary_c_semantic_fingerprints[semantic_fingerprint] = transition
                semantic_by_strategy["primary_c"][transition] += 1

            explanation = ""
            if transition == "1->0":
                explanation = _explain_one_to_zero(
                    candidate,
                    group,
                    payload,
                    baseline_window,
                    candidate_window,
                )
            if transition in {"1->0", "0->1"} or deterministic_surface_changed:
                changes.append(
                    {
                        "task_id": task_id,
                        "completion_id": int(row["completions_id"]),
                        "sample": int(row["sample_index"]),
                        "repeat": int(row["avg_repeat_index"]),
                        "pass_index": int(row["pass_index"]),
                        "group": group,
                        "primary_c": is_primary_c,
                        "benchmark": str(meta.get("benchmark_name") or ""),
                        "model": str(meta.get("model_name") or ""),
                        "reference": reference,
                        "stored_answer": str(row["answer"] or ""),
                        "stored_passed": bool(row["is_passed"]),
                        "transition": transition,
                        "stored_final_transition": stored_transition,
                        "baseline_answer": baseline_answer,
                        "candidate_answer": candidate_answer,
                        "baseline_fail_reason": baseline_reason,
                        "candidate_fail_reason": candidate_reason,
                        "baseline_judge_eligible": baseline_judge_eligible,
                        "candidate_judge_eligible": candidate_judge_eligible,
                        "baseline_math_passed": baseline_passed,
                        "candidate_math_passed": candidate_passed,
                        "baseline_final_passed": baseline_final_passed,
                        "candidate_final_passed": candidate_final_passed,
                        "baseline_would_judge": baseline_would_judge,
                        "candidate_would_judge": candidate_would_judge,
                        "baseline_judge_input": baseline_judge_input,
                        "candidate_judge_input": candidate_judge_input,
                        "deterministic_surface_changed": deterministic_surface_changed,
                        "judge_input_affected": judge_input_affected,
                        "replay_affected": replay_affected,
                        "baseline_window": _short(baseline_window),
                        "candidate_window": _short(candidate_window),
                        "semantic_duplicate": semantic_duplicate,
                        "explanation": explanation,
                    }
                )
            if args.progress_every and scanned % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "scanned_structural": scanned,
                            "changed_windows": changed_windows,
                            "proof_equivalent_rows": proof_equivalent_rows,
                            "proof_equivalent_reasons": dict(
                                proof_equivalent_reasons
                            ),
                            "judgement_rows": judgement_rows,
                            "stored_noncomparable_rows": stored_noncomparable_rows,
                            "stored_reference_drift_rows": stored_reference_drift_rows,
                            "real_scorer_rows": real_scorer_rows,
                            "real_scorer_reasons": dict(real_scorer_reasons),
                            "full_candidate_scores": full_candidate_scores,
                            "full_baseline_scores": full_baseline_scores,
                            "timeout_stats_by_implementation": {
                                implementation: dict(stats)
                                for implementation, stats in sorted(
                                    timeout_stats_by_implementation.items()
                                )
                            },
                            "transitions": dict(transitions),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

        if args.prove_a_superset and a_task_ids and not args.full_scan_a:
            complement_sql = """
                select e.eval_id, c.completions_id, c.task_id, c.sample_index,
                       c.avg_repeat_index, c.pass_index, c.context,
                       e.answer, e.ref_answer, e.is_passed, e.fail_reason
                from completions c
                join eval e using(completions_id)
                where c.task_id = any(%s)
                  and not (
                      coalesce(c.context #>> '{strategy_a,completion}', '') ~* %s
                      or coalesce(c.context #>> '{stages,0,completion}', '') ~* %s
                      or coalesce(c.context ->> 'strategy_a_completion', '') ~* %s
                      or coalesce(c.context ->> 'completion1', '') ~* %s
                  )
                order by c.task_id, c.completions_id, e.eval_id
            """
            proof_cursor = connection.cursor(
                name="free_response_a_prefilter_superset_proof"
            )
            proof_cursor.itersize = args.fetch_size
            proof_cursor.execute(
                complement_sql,
                (
                    a_task_ids,
                    ANSWER_CUE_SQL_RE,
                    ANSWER_CUE_SQL_RE,
                    ANSWER_CUE_SQL_RE,
                    ANSWER_CUE_SQL_RE,
                ),
            )
            for row in proof_cursor:
                if (
                    args.max_a_superset_proof_rows is not None
                    and a_superset_proof_scanned
                    >= args.max_a_superset_proof_rows
                ):
                    break
                a_superset_proof_scanned += 1
                task_id = int(row["task_id"])
                group = groups.get(task_id, "strategy_a")
                payload = _payload_from_context(row)
                _question, reference, _reference_drift = _current_production_record(
                    metadata_snapshot,
                    task_id=task_id,
                    sample_index=int(row["sample_index"]),
                    stored_reference=str(row["ref_answer"] or ""),
                )
                try:
                    baseline_kind, baseline_window = _verification_window(
                        baseline, group, payload, reference
                    )
                    candidate_kind, candidate_window = _verification_window(
                        candidate, group, payload, reference
                    )
                except Exception as exc:  # noqa: BLE001
                    a_superset_proof_errors += 1
                    scoring_errors += 1
                    a_superset_violations.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "error": f"window:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                if (baseline_kind, baseline_window) != (
                    candidate_kind,
                    candidate_window,
                ):
                    a_superset_violations.append(
                        {
                            "task_id": task_id,
                            "completion_id": int(row["completions_id"]),
                            "sample": int(row["sample_index"]),
                            "repeat": int(row["avg_repeat_index"]),
                            "reference": reference,
                            "baseline_window": _short(baseline_window),
                            "candidate_window": _short(candidate_window),
                        }
                    )

    timeout_stats: Counter[str] = Counter()
    for implementation_stats in timeout_stats_by_implementation.values():
        timeout_stats.update(implementation_stats)
    blocking_timeout_count = indeterminate_rows

    cell_deltas: list[dict[str, Any]] = []
    for task_id, counts in task_deltas.items():
        meta = metadata.get(task_id, {})
        total = task_counts.get(task_id, 0)
        numerator_delta = counts["0->1"] - counts["1->0"]
        cell_deltas.append(
            {
                "task_id": task_id,
                "group": groups.get(task_id, "strategy_a"),
                "primary_c": task_id in primary_c_tasks,
                "benchmark": str(meta.get("benchmark_name") or ""),
                "model": str(meta.get("model_name") or ""),
                "arch": str(meta.get("arch_version") or ""),
                "params": str(meta.get("num_params") or ""),
                "rows": total,
                **dict(counts),
                "candidate_minus_baseline_pp": (
                    100.0 * numerator_delta / total if total else 0.0
                ),
            }
        )
    cell_deltas.sort(
        key=lambda item: (-abs(item["candidate_minus_baseline_pp"]), item["task_id"])
    )
    one_to_zero = [
        row for row in changes if row.get("transition") == "1->0"
    ]
    unexplained = [
        row for row in one_to_zero if row.get("explanation") == "UNEXPLAINED"
    ]
    _verify_dependency_environment(dependency_manifest)
    _verify_loaded_src_modules(importer)
    _read_dependency_manifest(
        args.dependency_manifest,
        args.expected_dependency_manifest_sha256,
    )
    end_code_provenance = _verify_code_provenance(args)
    if end_code_provenance != code_provenance:
        raise RuntimeError("frozen code provenance changed during audit")
    output = {
        **inventory,
        "schema_version": PART_ARTIFACT_SCHEMA_VERSION,
        "baseline_module": str(args.baseline_module),
        "baseline_module_sha256": code_provenance["baseline_module_sha256"],
        "candidate_module": str(args.candidate_module),
        "candidate_module_sha256": code_provenance["candidate_module_sha256"],
        "audit_script_sha256": code_provenance["audit_script_sha256"],
        "dependency_manifest_file_sha256": (
            args.expected_dependency_manifest_sha256
        ),
        "dependency_environment": dependency_manifest,
        "math_fast_integer_match_env": os.getenv(
            "RWKV_MATH_FAST_INTEGER_MATCH"
        ),
        "math_fast_integer_match_enabled": {
            "baseline": bool(
                baseline._env_flag("RWKV_MATH_FAST_INTEGER_MATCH")
            ),
            "candidate": bool(
                candidate._env_flag("RWKV_MATH_FAST_INTEGER_MATCH")
            ),
        },
        "requested_groups": sorted(requested_groups),
        "audit_mode": {
            "full_scan_a": bool(args.full_scan_a),
            "full_real_scorer": bool(args.full_real_scorer),
            "max_structural_rows": args.max_structural_rows,
            "database_snapshot_imported": bool(args.database_snapshot_id),
            "metadata_snapshot_digest_verified": bool(
                args.metadata_snapshot_digest
            ),
            "question_source": "current_production_snapshot",
            "order_consistency_probe_rows": args.order_consistency_probe_rows,
            "independent_order_probe": True,
            "stable_row_order": "task_id,completions_id,eval_id",
            "frozen_code_verified": True,
            "frozen_src_contract_verified": True,
            "dependency_manifest_verified": True,
            "atomic_part_artifact": True,
            "strategy_a_selection": (
                "all_rows" if args.full_scan_a else "sql_answer_cue_prefilter"
            ),
        },
        "partition": {
            "count": args.partitions,
            "index": args.partition_index,
            "selected_a_tasks": len(a_task_ids),
            "selected_bc_tasks": len(bc_task_ids),
            "selected_task_ids": sorted(set(a_task_ids).union(bc_task_ids)),
            "expected_task_counts": {
                str(task_id): task_counts.get(task_id, 0)
                for task_id in sorted(set(a_task_ids).union(bc_task_ids))
            },
            "scanned_task_counts": {
                str(task_id): count
                for task_id, count in sorted(scanned_by_task.items())
            },
        },
        "sql_answer_cue_regex": ANSWER_CUE_SQL_RE,
        "a_sql_prefilter_superset_proof": {
            "enabled": bool(args.prove_a_superset and not args.full_scan_a),
            "not_applicable": bool(args.full_scan_a),
            "exhaustive": bool(
                (
                    args.full_scan_a
                    and args.max_structural_rows is None
                )
                or (
                    args.prove_a_superset
                    and not args.full_scan_a
                    and args.max_a_superset_proof_rows is None
                )
            ),
            "complement_rows_scanned": a_superset_proof_scanned,
            "changed_windows_outside_prefilter": len(
                a_superset_violations
            ),
            "errors": a_superset_proof_errors,
            "violations": a_superset_violations,
        },
        "structural_rows_scanned": scanned,
        "changed_verification_windows": changed_windows,
        "proof_equivalent_rows": proof_equivalent_rows,
        "proof_equivalent_reasons": dict(proof_equivalent_reasons),
        "proof_equivalent_rows_by_strategy": dict(proof_equivalent_by_strategy),
        "judgement_rows": judgement_rows,
        "stored_noncomparable_rows": stored_noncomparable_rows,
        "stored_reference_drift_rows": stored_reference_drift_rows,
        "stored_reference_drift_by_task": {
            str(key): value
            for key, value in sorted(stored_reference_drift_by_task.items())
        },
        "deterministic_surface_changed_rows": deterministic_surface_changed_rows,
        "judge_input_affected_rows": judge_input_affected_rows,
        "judge_input_affected_by_task": {
            str(key): value
            for key, value in sorted(judge_input_affected_by_task.items())
        },
        "replay_affected_rows": replay_affected_rows,
        "replay_affected_by_task": {
            str(key): value
            for key, value in sorted(replay_affected_by_task.items())
        },
        "replay_affected_task_ids": sorted(replay_affected_by_task),
        "replay_affected_reasons": dict(replay_affected_reasons),
        "audit_scope": {
            "proved": (
                "deterministic score_free_response_strategy math_passed, "
                "pre-Judge final_passed, extraction, and Judge request routing"
            ),
            "not_proved": (
                "external Judge verdicts; changed Judge routes/inputs and Strategy "
                "A inheritance are expanded into the replay manifest"
            ),
            "required_follow_up": (
                "rerun every judge_input_affected task/cell with the locked "
                "deterministic Judge protocol"
            ),
        },
        "real_scorer_rows": real_scorer_rows,
        "real_scorer_reasons": dict(real_scorer_reasons),
        "full_candidate_scores": full_candidate_scores,
        "full_baseline_scores": full_baseline_scores,
        "timeout_retries": dict(timeout_stats),
        "timeout_retries_by_implementation": {
            implementation: dict(stats)
            for implementation, stats in sorted(
                timeout_stats_by_implementation.items()
            )
        },
        "blocking_timeout_count": blocking_timeout_count,
        "indeterminate_rows": indeterminate_rows,
        "score_cache_entries": len(score_cache),
        "scoring_errors": scoring_errors,
        "module_order_consistency": {
            "probe_processes": "two_independent_processes_per_row",
            "orders": ["candidate_then_baseline", "baseline_then_candidate"],
            "probed_rows": order_consistency_probed_rows,
            "conflict_count": len(order_consistency_conflicts),
            "conflicts": order_consistency_conflicts,
            "timeout_events": dict(order_probe_timeout_stats),
        },
        "row_transitions": dict(transitions),
        "stored_final_transitions": dict(stored_transitions),
        "row_transitions_by_strategy": {
            key: dict(value)
            for key, value in sorted(transition_by_strategy.items())
        },
        "stored_final_transitions_by_strategy": {
            key: dict(value)
            for key, value in sorted(stored_transition_by_strategy.items())
        },
        "canonical_changed_payloads": len(semantic_fingerprints),
        "canonical_transitions": dict(semantic_transitions),
        "canonical_transitions_by_strategy": {
            key: dict(value)
            for key, value in sorted(semantic_by_strategy.items())
        },
        "canonical_fingerprints": semantic_fingerprints,
        "primary_c_canonical_fingerprints": primary_c_semantic_fingerprints,
        "one_to_zero": len(one_to_zero),
        "unexplained_one_to_zero": len(unexplained),
        "cell_deltas": cell_deltas,
        "changes": changes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _atomic_publish_json(args.output, output)
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"cell_deltas", "changes"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"cells={len(cell_deltas)} changes={len(changes)} output={args.output}",
        flush=True,
    )
    if blocking_timeout_count or indeterminate_rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
