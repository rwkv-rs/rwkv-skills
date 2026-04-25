from __future__ import annotations

from pathlib import Path

import scripts.prompt_ablation as prompt_ablation
from src.eval.benchmark_config import (
    CONFIG_OVERRIDE_ROOT_ENV,
    config_path_for_benchmark,
    resolve_benchmark_model_config,
)


def test_prompt_ablation_builds_overlay_configs_and_commands(tmp_path: Path) -> None:
    prompt_set = tmp_path / "prompts.toml"
    prompt_set.write_text(
        """
[trial.asdiv_short]
jobs = ["free_response"]
benchmarks = ["asdiv_cot", "gpqa_main_cot"]
max_samples = 3
cot_prompt_template = "COT <Q>"
final_prompt_template = "FINAL <COT>"
""",
        encoding="utf-8",
    )

    trials = prompt_ablation.load_trials(prompt_set)
    root, commands = prompt_ablation.build_experiment(
        trials,
        output_root=tmp_path / "runs",
        run_id="case",
        models=["/weights/model.pth"],
        overwrite=True,
        disable_checker=True,
        db_env={"PG_DBNAME": "rwkv-skills-prompt-ablation"},
    )

    assert root == tmp_path / "runs" / "case"
    assert trials[0].benchmarks == ("asdiv", "gpqa_main")
    assert trials[0].config_benchmarks == ("asdiv", "gpqa")
    assert (root / "asdiv_short" / "configs" / "asdiv.toml").exists()
    assert (root / "asdiv_short" / "configs" / "gpqa.toml").exists()
    config_text = (root / "asdiv_short" / "configs" / "asdiv.toml").read_text(encoding="utf-8")
    assert "[cot]" in config_text
    assert 'cot_prompt_template = "COT <Q>"' in config_text
    assert "--benchmark-config-root" in commands[0].command
    assert "--overwrite" in commands[0].command
    assert "--disable-checker" in commands[0].command
    assert (root / "commands.sh").exists()
    script_text = (root / "commands.sh").read_text(encoding="utf-8")
    assert "export PG_DBNAME=rwkv-skills-prompt-ablation" in script_text


def test_benchmark_config_overlay_inherits_base_and_adds_prompt(monkeypatch, tmp_path: Path) -> None:
    overlay = tmp_path / "configs"
    overlay.mkdir()
    (overlay / "asdiv.toml").write_text(
        """
[default]
max_samples = 9

[direct]
direct_prompt_template = "DIRECT <Q>"

[cot]
cot_prompt_template = "COT <Q>"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_OVERRIDE_ROOT_ENV, str(overlay))

    direct_config = resolve_benchmark_model_config("asdiv_test", "rwkv7-g1f-13.3b", stage="direct")
    cot_config = resolve_benchmark_model_config("asdiv_test", "rwkv7-g1f-13.3b", stage="cot")

    assert direct_config is not None
    assert direct_config.direct_prompt_template == "DIRECT <Q>"
    assert direct_config.max_samples == 9
    assert direct_config.avg_k == (2,)
    assert cot_config is not None
    assert cot_config.cot_prompt_template == "COT <Q>"
    assert config_path_for_benchmark("asdiv_test", "rwkv7-g1f-13.3b") == overlay / "asdiv.toml"


def test_config_benchmark_names_prefer_real_config_files(tmp_path: Path) -> None:
    prompt_set = tmp_path / "prompts.toml"
    prompt_set.write_text(
        """
[trial.math_names]
jobs = ["free_response"]
benchmarks = ["college_math_cot", "minerva_math_cot", "gpqa_main_cot"]
max_samples = 3
""",
        encoding="utf-8",
    )

    trials = prompt_ablation.load_trials(prompt_set)

    assert trials[0].config_benchmarks == ("college_math", "minerva_math", "gpqa")
