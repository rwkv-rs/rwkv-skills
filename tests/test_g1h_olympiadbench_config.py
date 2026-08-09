from __future__ import annotations

import tomllib
from pathlib import Path


def test_g1h_olympiad_final_stage_closes_think_block() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "g1h" / "olympiadbench.toml").read_text(
            encoding="utf-8"
        )
    )

    final = config["final"]
    assert "<COT></think>" in final["final_prompt_template"]
    assert final["max_generate_tokens"] == 512


def test_all_g1h_final_prompt_overrides_close_think_block() -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs" / "g1h"
    failures: list[str] = []

    for path in sorted(config_root.glob("*.toml")):
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        final = config.get("final")
        if not isinstance(final, dict) or "final_prompt_template" not in final:
            continue
        template = str(final["final_prompt_template"])
        if "<COT></think>" not in template:
            failures.append(path.name)

    assert not failures, f"G1h final prompt overrides left <think> open: {failures}"
