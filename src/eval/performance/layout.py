from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.eval.scheduler.config import RESULTS_ROOT
from src.eval.scheduler.dataset_utils import safe_slug


PERFORMANCE_RESULTS_ROOT = RESULTS_ROOT / "performance"


def ensure_performance_results_root() -> Path:
    PERFORMANCE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return PERFORMANCE_RESULTS_ROOT


def default_performance_result_path(
    *,
    model_name: str,
    protocol: str,
    stack_name: str,
) -> Path:
    root = ensure_performance_results_root()
    model_dir = root / safe_slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}__{safe_slug(protocol)}__{safe_slug(stack_name)}.json"
    return model_dir / filename


__all__ = [
    "PERFORMANCE_RESULTS_ROOT",
    "default_performance_result_path",
    "ensure_performance_results_root",
]
