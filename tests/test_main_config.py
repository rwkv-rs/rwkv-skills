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
result_store = "json"
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
    assert config.runner.result_store == "json"
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
                "prompt_style": "rwkv_official_json",
                "avg_ks": [1.0],
                "max_steps": 20,
                "max_tool_errors": 20,
                "prompt_max_chars": 8192,
                "long_doc_mode": "off",
                "long_doc_model_max_tokens": 64,
                "long_doc_model_parallel_batch_size": 6,
                "tool_router_mode": "lexical",
                "tool_router_max_tools": 8,
                "tool_router_parallel_chunk_tools": 3,
                "tool_router_parallel_batch_size": 5,
                "user_model": "gpt-5.4-mini",
                "user_base_url": "https://next-token.cc/v1",
                "judge_model": "gpt-5.4",
                "judge_base_url": "https://next-token.cc/v1",
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
    assert "--prompt-style" in resolved.argv
    assert "rwkv_official_json" in resolved.argv
    assert "--prompt-max-chars" in resolved.argv
    assert "8192" in resolved.argv
    assert "--long-doc-mode" in resolved.argv
    assert "off" in resolved.argv
    assert "--long-doc-model-max-tokens" in resolved.argv
    assert "64" in resolved.argv
    assert "--long-doc-model-parallel-batch-size" in resolved.argv
    assert "6" in resolved.argv
    assert "--tool-router-mode" in resolved.argv
    assert "lexical" in resolved.argv
    assert "--tool-router-max-tools" in resolved.argv
    assert "--tool-router-parallel-chunk-tools" in resolved.argv
    assert "3" in resolved.argv
    assert "--tool-router-parallel-batch-size" in resolved.argv
    assert "5" in resolved.argv
    assert "--user-model" in resolved.argv
    assert "gpt-5.4-mini" in resolved.argv
    assert "--user-base-url" in resolved.argv
    assert "https://next-token.cc/v1" in resolved.argv
    assert "--judge-model" in resolved.argv
    assert "gpt-5.4" in resolved.argv
    assert "--judge-base-url" in resolved.argv


def test_resolve_run_config_passes_long_doc_options_to_swebench_runner(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "swe_bench_lite_bm25_13k", "split": "test"},
            "model": {"infer_base_url": "http://127.0.0.1:8181", "infer_model": "demo"},
            "runner": {
                "benchmark_kind": "swe_bench",
                "cot_mode": "cot",
                "long_doc_mode": "model_parallel",
                "long_doc_max_evidence_chars": 3000,
                "long_doc_model_parallel_batch_size": 8,
            },
        }
    )

    dataset_path = tmp_path / "swe_bench_lite_bm25_13k" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps({"task_id": "a__b-1", "prompt": "Fix it.", "instance_id": "a__b-1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.runner.name == "code_swe_bench"
    assert resolved.module == "src.eval.coding.runner"
    assert "--long-doc-mode" in resolved.argv
    assert "model_parallel" in resolved.argv
    assert "--long-doc-max-evidence-chars" in resolved.argv
    assert "3000" in resolved.argv
    assert "--long-doc-model-parallel-batch-size" in resolved.argv
    assert "8" in resolved.argv


def test_resolve_run_config_passes_longcodebench_kind_and_answer_tokens(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "longcodeqa"},
            "model": {"infer_base_url": "http://127.0.0.1:8181", "infer_model": "demo"},
            "runner": {
                "result_store": "json",
                "long_doc_mode": "off",
                "answer_max_tokens": 64,
                "avg_ks": [1.0],
            },
        }
    )

    dataset_path = tmp_path / "longcodeqa" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps({"task_id": "a", "prompt": "p", "repo_text": "r", "question": "q", "correct_letter": "A"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.runner.name == "function_longcodebench"
    assert resolved.module == "src.eval.function_calling.runner"
    assert "--benchmark-kind" in resolved.argv
    assert "longcodebench" in resolved.argv
    assert "--answer-max-tokens" in resolved.argv
    assert "64" in resolved.argv
    assert resolved.env["RWKV_EVAL_STORE"] == "json"


def test_resolve_run_config_can_select_json_result_store(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "tau3_bench_airline"},
            "model": {"infer_base_url": "http://127.0.0.1:8181", "infer_model": "demo"},
            "runner": {
                "result_store": "json",
                "benchmark_kind": "tau3_bench",
                "prompt_style": "rwkv_official_json",
            },
        }
    )

    dataset_path = tmp_path / "tau3_bench_airline" / "base.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert resolved.env["RWKV_EVAL_STORE"] == "json"
    assert "RWKV_EVAL_STORE" not in main_module.resolve_run_config(
        main_module.RunConfig.from_mapping(
            {
                "dataset": {"name": "tau3_bench_airline"},
                "model": {"infer_base_url": "http://127.0.0.1:8181", "infer_model": "demo"},
                "runner": {"benchmark_kind": "tau3_bench"},
            }
        )
    ).env


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


def test_local_model_config_passes_lightning_state_cache_args(monkeypatch, tmp_path: Path) -> None:
    config = main_module.RunConfig.from_mapping(
        {
            "dataset": {"name": "mmlu"},
            "model": {
                "path": "weights/model.pth",
                "device": "cuda:0",
                "engine_mode": "lightning",
                "state_db_path": "tmp/state-cache.sqlite3",
            },
            "runner": {"cot_mode": "cot"},
        }
    )

    dataset_path = tmp_path / "mmlu" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "resolve_or_prepare_dataset", lambda *_args, **_kwargs: dataset_path)

    resolved = main_module.resolve_run_config(config)

    assert "--model-path" in resolved.argv
    assert "weights/model.pth" in resolved.argv
    assert "--engine-mode" in resolved.argv
    assert "lightning" in resolved.argv
    assert "--state-db-path" in resolved.argv
    assert "tmp/state-cache.sqlite3" in resolved.argv


def test_run_from_config_passes_contracts_and_patches_env(monkeypatch, tmp_path: Path) -> None:
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
    assert captured["job_name_env"] == "multi_choice_cot"
    assert captured["run_mode_env"] == "resume"
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
    assert "256" in resolved.argv
    assert "--final-max-tokens" in resolved.argv
    assert "64" in resolved.argv


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
