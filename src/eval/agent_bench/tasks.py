from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
TAU_V1_VENDOR_ROOT = ROOT / "data" / "tau_v1"
TAU_V2_VENDOR_ROOT = ROOT / "data" / "tau_v2"
TAU_V2_DATA_ROOT = TAU_V2_VENDOR_ROOT / "data"


@dataclass(slots=True)
class ManifestTask:
    task_id: str
    domain: str
    index: int
    instruction: str
    payload: dict[str, Any]


def ensure_tau_v1_vendor_path() -> Path:
    if str(TAU_V1_VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(TAU_V1_VENDOR_ROOT))
    return TAU_V1_VENDOR_ROOT


def ensure_tau_v2_vendor_path() -> Path:
    if str(TAU_V2_VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(TAU_V2_VENDOR_ROOT))
    # tau2 expects DATA_DIR/tau2/...; DATA_DIR is TAU2_DATA_DIR when set.
    os.environ.setdefault("TAU2_DATA_DIR", str(TAU_V2_DATA_ROOT))
    return TAU_V2_VENDOR_ROOT


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
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            "failed to import tau v1 task definitions; ensure dependencies (e.g. pydantic) are installed"
        ) from exc
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
    try:
        module = importlib.import_module(_tau2_env_module(domain))
    except Exception as exc:
        raise RuntimeError(
            "failed to import tau2 task definitions; ensure dependencies (e.g. pydantic) are installed"
        ) from exc
    get_tasks = getattr(module, "get_tasks")
    tasks = get_tasks(split)
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
    "load_tau_v1_tasks",
    "load_tau_v2_tasks",
    "iter_task_rows",
]
