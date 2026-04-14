from __future__ import annotations

import json
import os
from pathlib import Path
import types

import pytest

from src import main as main_module


def test_load_run_config_parses_toml_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[run]
mode = "eval"
run_mode = "rerun"
batch_size = 16
probe_only = true

[dataset]
name = "mmlu"
split = "test"

[model]
path = "weights/model.pth"
device = "cuda:1"

[runner]
cot_mode = "cot"
db_write_queue = 2048
extra_args = ["--foo", "bar"]
""".strip(),
        encoding="utf-8",
    )

    config = main_module.load_run_config(config_path)

    assert config.run.run_mode.value == "rerun"
    assert config.run.batch_size == 16
    assert config.run.probe_only is True
    assert config.dataset.name == "mmlu"
    assert config.dataset.split == "test"
    assert config.model.path == "weights/model.pth"
    assert config.model.device == "cuda:1"
    assert config.runner.cot_mode == "cot"
    assert config.runner.db_write_queue == 2048
    assert config.runner.extra_args == ("--foo", "bar")


def test_resolve_run_config_path_accepts_named_config(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "configs" / "run"
    run_root.mkdir(parents=True)
    config_path = run_root / "bfcl_v3.toml"
    config_path.write_text("[dataset]\nname='bfcl_v3'\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "RUN_CONFIG_ROOT", run_root)

    resolved = main_module.resolve_run_config_path("bfcl_v3")

    assert resolved == config_path.resolve()


def test_resolve_run_config_path_accepts_benchmark_alias(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "configs" / "run"
    run_root.mkdir(parents=True)
    config_path = run_root / "bfcl_v3.toml"
    config_path.write_text("[dataset]\nname='bfcl_v3'\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "RUN_CONFIG_ROOT", run_root)

    resolved = main_module.resolve_run_config_path(benchmark="bfcl_v3")

    assert resolved == config_path.resolve()


def test_resolve_run_config_dispatches_to_field_runner(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "run": {"batch_size": 32, "run_mode": "resume"},
            "dataset": {"name": "mmlu"},
            "model": {"path": "weights/model.pth"},
            "runner": {"cot_mode": "cot", "db_write_queue": 1024},
        }
    )

    dataset_path = tmp_path / "mmlu" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.runner.name == "multi_choice_cot"
    assert resolved.module == "src.eval.knowledge.runner"
    assert resolved.dataset_slug == "mmlu_test"
    assert resolved.env["RWKV_SKILLS_JOB_NAME"] == "multi_choice_cot"
    assert resolved.env["RWKV_EVAL_RUN_MODE"] == "resume"
    assert "--dataset" in resolved.argv
    assert "--batch-size" in resolved.argv
    assert "--cot-mode" in resolved.argv
    assert "--db-write-queue" in resolved.argv


def test_resolve_run_config_passes_avg_k_to_function_calling_runner(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "bfcl_v3"},
            "model": {"infer_base_url": "http://127.0.0.1:8181", "infer_model": "demo"},
            "runner": {
                "benchmark_kind": "bfcl_v3",
                "avg_ks": [1.0],
                "max_steps": 20,
                "max_tool_errors": 20,
            },
        }
    )

    dataset_path = tmp_path / "bfcl_v3" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.runner.name == "function_bfcl_v3"
    assert resolved.module == "src.eval.function_calling.runner"
    assert "--avg-k" in resolved.argv
    assert "1.0" in resolved.argv


def test_run_from_config_invokes_runner_and_restores_env(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "run": {"run_mode": "rerun"},
            "dataset": {"name": "human_eval"},
            "model": {"path": "weights/model.pth"},
            "runner": {"max_tokens": 512},
        }
    )

    dataset_path = tmp_path / "human_eval" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    captured: dict[str, object] = {}

    def _fake_main(argv):
        captured["argv"] = tuple(argv)
        captured["job_name"] = os.environ.get("RWKV_SKILLS_JOB_NAME")
        captured["run_mode"] = os.environ.get("RWKV_EVAL_RUN_MODE")
        captured["overwrite"] = os.environ.get("RWKV_SCHEDULER_OVERWRITE")
        return 7

    fake_module = types.SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(main_module.importlib, "import_module", lambda _name: fake_module)
    monkeypatch.delenv("RWKV_SKILLS_JOB_NAME", raising=False)
    monkeypatch.delenv("RWKV_EVAL_RUN_MODE", raising=False)

    result = main_module.run_from_config(config)

    assert result == 7
    assert captured["job_name"] == "code_human_eval"
    assert captured["run_mode"] == "rerun"
    assert captured["overwrite"] == "1"
    assert "--benchmark-kind" in captured["argv"]
    assert "--max-tokens" in captured["argv"]
    assert "RWKV_SKILLS_JOB_NAME" not in os.environ
    assert "RWKV_EVAL_RUN_MODE" not in os.environ


def test_run_from_config_prefers_explicit_contracts_over_env(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "run": {"run_mode": "resume"},
            "dataset": {"name": "mmlu"},
            "model": {"path": "weights/model.pth"},
            "runner": {"cot_mode": "cot"},
        }
    )

    dataset_path = tmp_path / "mmlu" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    captured: dict[str, object] = {}

    def _fake_main(argv, *, run_context=None, task_spec=None):
        captured["argv"] = tuple(argv)
        captured["run_context"] = run_context
        captured["task_spec"] = task_spec
        captured["job_name_env"] = os.environ.get("RWKV_SKILLS_JOB_NAME")
        captured["run_mode_env"] = os.environ.get("RWKV_EVAL_RUN_MODE")
        return 0

    fake_module = types.SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(main_module.importlib, "import_module", lambda _name: fake_module)
    monkeypatch.delenv("RWKV_SKILLS_JOB_NAME", raising=False)
    monkeypatch.delenv("RWKV_EVAL_RUN_MODE", raising=False)

    result = main_module.run_from_config(config)

    assert result == 0
    assert captured["job_name_env"] is None
    assert captured["run_mode_env"] is None
    assert captured["run_context"].job_name == "multi_choice_cot"
    assert captured["run_context"].run_mode.value == "resume"
    assert captured["task_spec"].runner_name == "multi_choice_cot"
    assert captured["task_spec"].dataset_slug == "mmlu_test"
    assert captured["task_spec"].model_name == "model"


def test_main_dry_run_prints_resolved_invocation(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[dataset]
name = "ifeval"

[model]
path = "weights/model.pth"
""".strip(),
        encoding="utf-8",
    )

    dataset_path = tmp_path / "ifeval" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    assert main_module.main(["--config", str(config_path), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["job"] == "instruction_following"
    assert payload["module"] == "src.eval.instruction_following.runner"
    assert payload["dataset_slug"] == "ifeval_test"


def test_main_dry_run_accepts_benchmark_alias(monkeypatch, tmp_path: Path, capsys) -> None:
    run_root = tmp_path / "configs" / "run"
    run_root.mkdir(parents=True)
    config_path = run_root / "bfcl_v3.toml"
    config_path.write_text(
        """
[dataset]
name = "bfcl_v3"

[model]
infer_base_url = "http://127.0.0.1:8181"
infer_model = "demo"

[runner]
benchmark_kind = "bfcl_v3"
""".strip(),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "bfcl_v3" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "RUN_CONFIG_ROOT", run_root)
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    assert main_module.main(["--benchmark", "bfcl_v3", "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config_path"] == str(config_path.resolve())
    assert payload["job"] == "function_bfcl_v3"
    assert payload["dataset_slug"] == "bfcl_v3_test"


def test_dataset_prepare_false_uses_existing_index(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "mmlu" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")

    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "mmlu", "prepare": False},
            "model": {"path": "weights/model.pth"},
        }
    )

    monkeypatch.setattr(main_module, "find_dataset_file", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.dataset_path == dataset_path


def test_dataset_path_must_match_benchmark_slug(tmp_path: Path) -> None:
    path = tmp_path / "custom.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "mmlu", "path": str(path)},
            "model": {"path": "weights/model.pth"},
        }
    )

    with pytest.raises(ValueError, match="expected 'mmlu_test'"):
        _ = main_module.resolve_run_config(config)


def test_resolve_run_config_supports_param_search_mode(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "run": {"mode": "param_search"},
            "dataset": {"name": "gsm8k"},
            "model": {"path": "weights/model.pth"},
            "runner": {"db_write_queue": 512, "cot_max_tokens": 256, "final_max_tokens": 64},
        }
    )

    dataset_path = tmp_path / "gsm8k" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.runner.name == "param_search_free_response_judge"
    assert resolved.module == "src.bin.param_search_free_response_judge"
    assert "--db-write-queue" in resolved.argv
    assert "--cot-max-tokens" in resolved.argv
    assert "--final-max-tokens" in resolved.argv


def test_param_search_requires_compatible_maths_benchmark(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "run": {"mode": "param_search"},
            "dataset": {"name": "mmlu"},
            "model": {"path": "weights/model.pth"},
        }
    )

    dataset_path = tmp_path / "mmlu" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    with pytest.raises(ValueError, match="only supports maths benchmarks"):
        _ = main_module.resolve_run_config(config)
