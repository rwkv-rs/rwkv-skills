from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable, TypeVar

_AUTO_INSTALL_ENV = "RWKV_AGENT_BENCH_AUTO_INSTALL_DEPS"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

# module name -> distribution name
_MODULE_TO_DIST: dict[str, str] = {
    "docstring_parser": "docstring-parser",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}

_TAU_V1_TASK_MODULES: tuple[str, ...] = ("pydantic",)
_TAU_V1_RUNTIME_MODULES: tuple[str, ...] = ("pydantic", "litellm")

_TAU_V2_TASK_MODULES: tuple[str, ...] = (
    "pydantic",
    "toml",
    "docstring_parser",
    "deepdiff",
    "addict",
)
_TAU_V2_RUNTIME_MODULES: tuple[str, ...] = (
    "pydantic",
    "litellm",
    "toml",
    "docstring_parser",
    "deepdiff",
    "addict",
)

_T = TypeVar("_T")
_PROJECT_DEP_SPEC_CACHE: dict[str, str] | None = None


def ensure_tau_v1_task_dependencies() -> None:
    ensure_modules_available(_TAU_V1_TASK_MODULES, context="tau-bench task loading")


def ensure_tau_v1_runtime_dependencies() -> None:
    ensure_modules_available(_TAU_V1_RUNTIME_MODULES, context="tau-bench runtime")


def ensure_tau_v2_task_dependencies() -> None:
    ensure_modules_available(_TAU_V2_TASK_MODULES, context="tau2-bench task loading")


def ensure_tau_v2_runtime_dependencies() -> None:
    ensure_modules_available(_TAU_V2_RUNTIME_MODULES, context="tau2-bench runtime")


def import_module_with_auto_install(module_name: str, *, context: str) -> object:
    return run_with_auto_install(lambda: importlib.import_module(module_name), context=context)


def run_with_auto_install(operation: Callable[[], _T], *, context: str, max_rounds: int = 8) -> _T:
    installed_missing: set[str] = set()
    for _ in range(max(1, int(max_rounds))):
        try:
            return operation()
        except ModuleNotFoundError as exc:
            missing = _normalize_missing_module(exc.name)
            if not missing:
                raise
            if missing.startswith("tau2") or missing.startswith("tau_bench"):
                raise
            if missing in installed_missing:
                raise
            ensure_modules_available((missing,), context=context)
            installed_missing.add(missing)
    return operation()


def ensure_modules_available(modules: tuple[str, ...] | list[str], *, context: str) -> None:
    missing = [name for name in modules if not _module_available(name)]
    if not missing:
        return
    if not _auto_install_enabled():
        details = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing dependencies for {context}: {details}. "
            f"Set {_AUTO_INSTALL_ENV}=1 or install them manually."
        )
    requirements = [_resolve_requirement_from_module(name) for name in missing]
    _pip_install(requirements=requirements, context=context)
    still_missing = [name for name in missing if not _module_available(name)]
    if still_missing:
        details = ", ".join(sorted(set(still_missing)))
        raise RuntimeError(f"Dependency auto-install completed but modules are still missing: {details}")


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _auto_install_enabled() -> bool:
    raw = os.environ.get(_AUTO_INSTALL_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _pip_install(*, requirements: list[str], context: str) -> None:
    unique_requirements: list[str] = []
    for item in requirements:
        if item and item not in unique_requirements:
            unique_requirements.append(item)
    if not unique_requirements:
        return

    print(
        f"[agent_bench] Auto-installing missing dependencies for {context}: {', '.join(unique_requirements)}",
        flush=True,
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *unique_requirements,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Failed to auto-install dependencies for {context}. "
            f"Please run: {joined}"
        ) from exc


def _resolve_requirement_from_module(module_name: str) -> str:
    dist_name = _MODULE_TO_DIST.get(module_name, module_name)
    dep_map = _project_dependency_specs()
    key = _normalize_name(dist_name)
    return dep_map.get(key, dist_name)


def _project_dependency_specs() -> dict[str, str]:
    global _PROJECT_DEP_SPEC_CACHE
    if _PROJECT_DEP_SPEC_CACHE is not None:
        return _PROJECT_DEP_SPEC_CACHE

    specs: dict[str, str] = {}
    try:
        raw = _PYPROJECT_PATH.read_text(encoding="utf-8")
        parsed = tomllib.loads(raw)
    except Exception:
        _PROJECT_DEP_SPEC_CACHE = specs
        return specs

    deps = parsed.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        _PROJECT_DEP_SPEC_CACHE = specs
        return specs

    for entry in deps:
        if not isinstance(entry, str):
            continue
        requirement = entry.split(";", 1)[0].strip()
        if not requirement:
            continue
        name = _extract_dist_name(requirement)
        if not name:
            continue
        specs[_normalize_name(name)] = requirement

    _PROJECT_DEP_SPEC_CACHE = specs
    return specs


def _extract_dist_name(requirement: str) -> str | None:
    if "@" in requirement:
        lhs = requirement.split("@", 1)[0].strip()
        return lhs or None
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        return None
    return match.group(1)


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).strip().lower()


def _normalize_missing_module(value: str | None) -> str:
    if not value:
        return ""
    return value.split(".", 1)[0].strip()


__all__ = [
    "ensure_tau_v1_task_dependencies",
    "ensure_tau_v1_runtime_dependencies",
    "ensure_tau_v2_task_dependencies",
    "ensure_tau_v2_runtime_dependencies",
    "ensure_modules_available",
    "import_module_with_auto_install",
    "run_with_auto_install",
]
