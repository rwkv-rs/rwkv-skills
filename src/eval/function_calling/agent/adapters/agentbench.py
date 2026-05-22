from __future__ import annotations

"""AgentBench DB/KG adapter boundary."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgentBenchTask = Literal["db", "kg"]

DEFAULT_OFFICIAL_AGENTBENCH_ROOT = Path("/tmp/ref-AgentBench")
OFFICIAL_AGENTBENCH_SOURCE = "THUDM/AgentBench"


@dataclass(frozen=True, slots=True)
class AgentBenchAdapterConfig:
    task: AgentBenchTask
    official_root: Path = DEFAULT_OFFICIAL_AGENTBENCH_ROOT
    max_steps: int = 20


def official_agentbench_root() -> Path:
    return Path(os.environ.get("AGENTBENCH_OFFICIAL_ROOT") or DEFAULT_OFFICIAL_AGENTBENCH_ROOT)


def require_agentbench_assets(config: AgentBenchAdapterConfig) -> Path:
    root = Path(config.official_root or official_agentbench_root())
    task_dir = root / "src" / "server" / "tasks" / ("dbbench" if config.task == "db" else "knowledgegraph")
    if not task_dir.exists():
        raise FileNotFoundError(f"AgentBench {config.task} official task not found under {task_dir}")
    return root


__all__ = [
    "AgentBenchAdapterConfig",
    "AgentBenchTask",
    "DEFAULT_OFFICIAL_AGENTBENCH_ROOT",
    "OFFICIAL_AGENTBENCH_SOURCE",
    "official_agentbench_root",
    "require_agentbench_assets",
]
