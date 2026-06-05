from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.eval.agent_bench.deps import (
    ensure_tau_v1_task_dependencies,
    ensure_tau_v2_task_dependencies,
    import_module_with_auto_install,
    run_with_auto_install,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
TAU_V1_VENDOR_ROOT = ROOT / "data" / "tau_v1"
TAU_V2_VENDOR_ROOT = ROOT / "data" / "tau_v2"
TAU_V2_DATA_ROOT = TAU_V2_VENDOR_ROOT / "data"
TAU_V2_REFERENCE_ROOT = REPO_ROOT / "references" / "tau2-bench"


@dataclass(slots=True)
class ManifestTask:
    task_id: str
    domain: str
    index: int
    instruction: str
    payload: dict[str, Any]


def ensure_tau_v1_vendor_path() -> Path:
    ensure_tau_v1_task_dependencies()
    if str(TAU_V1_VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(TAU_V1_VENDOR_ROOT))
    return TAU_V1_VENDOR_ROOT


def ensure_tau_v2_vendor_path() -> Path:
    ensure_tau_v2_task_dependencies()
    vendor_root = tau_v2_vendor_root()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    # tau2 expects DATA_DIR/tau2/...; DATA_DIR is TAU2_DATA_DIR when set.
    os.environ.setdefault("TAU2_DATA_DIR", str(tau_v2_data_root()))
    return vendor_root


def tau_v2_vendor_root() -> Path:
    override = (
        os.environ.get("RWKV_TAU3_BENCH_ROOT")
        or os.environ.get("TAU3_BENCH_ROOT")
        or os.environ.get("RWKV_TAU2_BENCH_ROOT")
        or os.environ.get("TAU2_BENCH_ROOT")
    )
    if override:
        root = Path(override).expanduser().resolve()
        src_root = root / "src"
        if (src_root / "tau2").exists():
            return src_root
        return root
    reference_src = TAU_V2_REFERENCE_ROOT / "src"
    if (reference_src / "tau2").exists():
        return reference_src
    return TAU_V2_VENDOR_ROOT


def tau_v2_data_root() -> Path:
    override = (
        os.environ.get("RWKV_TAU3_DATA_ROOT")
        or os.environ.get("TAU3_DATA_ROOT")
        or os.environ.get("RWKV_TAU2_DATA_ROOT")
        or os.environ.get("TAU2_DATA_DIR")
    )
    if override:
        return Path(override).expanduser().resolve()
    vendor_root = tau_v2_vendor_root()
    if vendor_root.name == "src":
        return vendor_root.parent / "data"
    reference_data = TAU_V2_REFERENCE_ROOT / "data"
    if (reference_data / "tau2").exists():
        return reference_data
    return TAU_V2_DATA_ROOT


def tau_v3_source_available() -> bool:
    return (tau_v2_data_root() / "tau2" / "domains" / "banking_knowledge").exists()


def require_tau_v3_source(context: str = "tau3_bench") -> None:
    if tau_v3_source_available():
        return
    raise ValueError(
        f"{context} 需要最新版官方 tau2/tau3-bench 数据，当前解析到的 tau 数据根目录为 "
        f"{tau_v2_data_root()}，其中没有 tau2/domains/banking_knowledge。"
        "请设置 RWKV_TAU3_BENCH_ROOT/TAU3_BENCH_ROOT 指向官方仓库，或设置 "
        "RWKV_TAU3_DATA_ROOT/TAU3_DATA_ROOT 指向官方 data 目录。"
    )


def infer_domain_from_slug(slug: str) -> str:
    text = slug.lower()
    if "telecom" in text:
        return "telecom"
    if "retail" in text:
        return "retail"
    if "airline" in text:
        return "airline"
    raise ValueError(f"无法从 dataset slug 推断 domain: {slug}")


def load_manifest(path: str | Path, *, max_samples: int | None = None) -> list[ManifestTask]:
    items: list[ManifestTask] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            items.append(
                ManifestTask(
                    task_id=str(payload.get("task_id", "")),
                    domain=str(payload.get("domain", "")),
                    index=int(payload.get("index", 0)),
                    instruction=str(payload.get("instruction", "")),
                    payload=payload,
                )
            )
            if max_samples is not None and max_samples > 0 and len(items) >= max_samples:
                break
    return items


def _tau_v1_instruction(task: Any) -> str:
    return str(getattr(task, "instruction", ""))


def _tau_v2_instruction(task: Any) -> str:
    user_scenario = getattr(task, "user_scenario", None)
    if user_scenario is None:
        return str(getattr(task, "description", ""))
    instructions = getattr(user_scenario, "instructions", None)
    if instructions is None:
        return str(user_scenario)
    if isinstance(instructions, str):
        return instructions
    return str(instructions)


def _model_dump_any(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(item, dict):
        return dict(item)
    try:
        return dict(item)
    except Exception:
        return {"value": str(item)}


def _tau_v1_task_source(domain: str, split: str) -> tuple[str, str]:
    domain = domain.lower().strip()
    split = split.lower().strip()
    if domain == "retail":
        mapping = {
            "test": ("tau_bench.envs.retail.tasks_test", "TASKS_TEST"),
            "train": ("tau_bench.envs.retail.tasks_train", "TASKS_TRAIN"),
            "dev": ("tau_bench.envs.retail.tasks_dev", "TASKS_DEV"),
        }
    elif domain == "airline":
        mapping = {
            "test": ("tau_bench.envs.airline.tasks_test", "TASKS"),
        }
    else:
        raise ValueError(f"不支持的 tau v1 domain: {domain}")
    try:
        return mapping[split]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"tau v1 domain={domain} 不支持 split={split}，可用 split: {valid}") from exc


def load_tau_v1_tasks(*, domain: str, split: str = "test") -> list[dict[str, Any]]:
    ensure_tau_v1_vendor_path()
    module_name, attr_name = _tau_v1_task_source(domain, split)
    module = import_module_with_auto_install(module_name, context=f"tau-bench task module: {module_name}")
    tasks = getattr(module, attr_name)
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_id = str(idx)
        payload = _model_dump_any(task)
        rows.append(
            {
                "task_id": task_id,
                "domain": domain,
                "index": idx,
                "instruction": _tau_v1_instruction(task),
                "task": payload,
                "benchmark_version": "tau_v1",
            }
        )
    return rows


def _tau2_env_module(domain: str) -> str:
    mapping = {
        "retail": "tau2.domains.retail.environment",
        "airline": "tau2.domains.airline.environment",
        "telecom": "tau2.domains.telecom.environment",
    }
    try:
        return mapping[domain]
    except KeyError as exc:
        raise ValueError(f"不支持的 tau2 domain: {domain}") from exc


def load_tau_v2_tasks(*, domain: str, split: str = "base") -> list[dict[str, Any]]:
    ensure_tau_v2_vendor_path()
    get_tasks = _tau2_tasks_loader(domain)
    tasks = run_with_auto_install(
        lambda: get_tasks(split),
        context=f"tau2-bench task loading: domain={domain}, split={split}",
    )
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_payload = _model_dump_any(task)
        task_id = str(getattr(task, "id", idx))
        rows.append(
            {
                "task_id": task_id,
                "domain": domain,
                "index": idx,
                "instruction": _tau_v2_instruction(task),
                "task": task_payload,
                "benchmark_version": "tau_v2",
            }
        )
    return rows


def _tau2_tasks_loader(domain: str):
    try:
        registry_module = import_module_with_auto_install("tau2.registry", context="tau2 registry import")
        registry = getattr(registry_module, "registry")
        return registry.get_tasks_loader(domain)
    except Exception:
        module_name = _tau2_env_module(domain)
        module = import_module_with_auto_install(module_name, context=f"tau2-bench task module: {module_name}")
        return getattr(module, "get_tasks")


def iter_task_rows(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    name = dataset_name.lower()
    if name.startswith("tau_bench_"):
        domain = name.removeprefix("tau_bench_")
        yield from load_tau_v1_tasks(domain=domain, split=split)
        return
    if name.startswith("tau2_bench_"):
        domain = name.removeprefix("tau2_bench_")
        yield from load_tau_v2_tasks(domain=domain, split=split)
        return
    raise ValueError(f"未知 agent_bench 数据集: {dataset_name}")


__all__ = [
    "ManifestTask",
    "infer_domain_from_slug",
    "load_manifest",
    "ensure_tau_v1_vendor_path",
    "ensure_tau_v2_vendor_path",
    "tau_v2_vendor_root",
    "tau_v2_data_root",
    "tau_v3_source_available",
    "require_tau_v3_source",
    "load_tau_v1_tasks",
    "load_tau_v2_tasks",
    "iter_task_rows",
]
