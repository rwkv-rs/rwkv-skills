from __future__ import annotations

"""TAU official task loading helpers."""

from typing import Any

from src.eval.agent_bench.deps import (
    TAU_V2_DATA_ROOT,
    ensure_tau_v2_vendor_path,
    import_module_with_auto_install,
    run_with_auto_install,
    tau_v2_data_root,
    tau_v2_vendor_root,
)


def tau_v3_source_available() -> bool:
    return (tau_v2_data_root() / "tau2" / "domains" / "banking_knowledge").exists()


def require_tau_v3_source(context: str = "tau3_bench") -> None:
    if tau_v3_source_available():
        return
    raise ValueError(
        f"{context} requires official TAU3 data. Resolved TAU data root is {tau_v2_data_root()}, "
        "but tau2/domains/banking_knowledge is missing."
    )


def infer_domain_from_slug(slug: str) -> str:
    text = slug.lower()
    for domain in ("banking_knowledge", "telecom", "retail", "airline"):
        if domain in text:
            return domain
    raise ValueError(f"Cannot infer TAU domain from dataset slug: {slug}")


def load_tau_tasks(
    *,
    task_set: str,
    domain: str | None = None,
    split: str | None = "base",
    benchmark_version: str = "tau_v2",
) -> list[dict[str, Any]]:
    ensure_tau_v2_vendor_path()
    get_tasks = _tau2_tasks_loader(task_set)
    task_split = split or None
    tasks = run_with_auto_install(
        lambda: get_tasks(task_split),
        context=f"tau-bench task loading: task_set={task_set}, split={task_split}",
    )
    resolved_domain = domain or task_set
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_payload = _model_dump_any(task)
        task_id = str(getattr(task, "id", idx))
        rows.append(
            {
                "task_id": task_id,
                "task_set": task_set,
                "task_split": task_split,
                "domain": resolved_domain,
                "index": idx,
                "instruction": _tau_v2_instruction(task),
                "task": task_payload,
                "benchmark_version": benchmark_version,
            }
        )
    return rows


def load_tau_v2_tasks(*, domain: str, split: str = "base") -> list[dict[str, Any]]:
    return load_tau_tasks(task_set=domain, domain=domain, split=split, benchmark_version="tau_v2")


def _tau2_tasks_loader(task_set: str):
    try:
        registry_module = import_module_with_auto_install("tau2.registry", context="tau2 registry import")
        registry = getattr(registry_module, "registry")
        return registry.get_tasks_loader(task_set)
    except Exception:
        module_name = _tau2_env_module(task_set)
        module = import_module_with_auto_install(module_name, context=f"tau2-bench task module: {module_name}")
        return getattr(module, "get_tasks")


def _tau2_env_module(domain: str) -> str:
    mapping = {
        "retail": "tau2.domains.retail.environment",
        "airline": "tau2.domains.airline.environment",
        "telecom": "tau2.domains.telecom.environment",
    }
    try:
        return mapping[domain]
    except KeyError as exc:
        raise ValueError(f"Unsupported tau2 domain: {domain}") from exc


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


__all__ = [
    "TAU_V2_DATA_ROOT",
    "ensure_tau_v2_vendor_path",
    "infer_domain_from_slug",
    "load_tau_tasks",
    "load_tau_v2_tasks",
    "require_tau_v3_source",
    "tau_v2_data_root",
    "tau_v2_vendor_root",
    "tau_v3_source_available",
]
