from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.bin.param_search_free_response import parse_args as parse_param_search_free_response_args
from src.bin.param_search_select import parse_args as parse_param_search_select_args
from src.eval.scheduler import actions, queue
from src.eval.scheduler.actions import DispatchOptions
from src.eval.scheduler.admin import SchedulerStartRequest
from src.eval.scheduler.cli import build_parser
from src.eval.scheduler.jobs import JOB_CATALOGUE
from src.eval.scheduler.state import RunningEntry
from src.infer.backend import resolve_backend_model_name, validate_inference_backend_args


def test_build_queue_supports_remote_inference_targets() -> None:
    dataset_slug = JOB_CATALOGUE["free_response"].dataset_slugs[0]

    items = queue.build_queue(
        model_globs=(),
        job_order=("free_response",),
        completed=(),
        failed=(),
        running=(),
        skip_dataset_slugs=(),
        only_dataset_slugs=(dataset_slug,),
        model_select="all",
        min_param_b=None,
        max_param_b=None,
        infer_base_url="http://127.0.0.1:8081",
        infer_models=("rwkv7-g1a4-2.9b-20250728",),
    )

    assert len(items) == 1
    item = items[0]
    assert item.is_remote is True
    assert item.model_path is None
    assert item.model_name == "rwkv7-g1a4-2.9b-20250728"
    assert item.infer_model == "rwkv7-g1a4-2.9b-20250728"
    assert item.infer_base_url == "http://127.0.0.1:8081"


def test_build_command_uses_remote_backend_arguments(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    item = queue.QueueItem(
        job_name="free_response",
        job_id="free_response__demo",
        dataset_slug="gsm8k_test",
        model_path=None,
        model_slug="remote_demo",
        model_name="remote-demo",
        infer_base_url="http://127.0.0.1:8081",
        infer_model="remote-demo",
    )

    command = actions.build_command(
        JOB_CATALOGUE["free_response"],
        item,
        dataset_path,
        None,
        batch_size=17,
        infer_api_key="secret",
        infer_timeout_s=12.5,
        infer_max_workers=9,
        infer_protocol="vllm",
    )

    assert "--infer-base-url" in command
    assert "--infer-model" in command
    assert "--infer-api-key" in command
    assert "--infer-timeout-s" in command
    assert "--infer-max-workers" in command
    assert "--infer-protocol" in command
    assert "vllm" in command
    assert "--model-path" not in command
    assert "--device" not in command
    assert "remote-demo" in command
    assert "17" in command


def test_remote_dispatch_resources_use_worker_slots(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response",),
        infer_base_url="http://127.0.0.1:8081",
        infer_models=("remote-demo",),
        max_concurrent_jobs=3,
    )
    running = {
        "job-a": RunningEntry(pid=101, gpu=None),
        "job-b": RunningEntry(pid=102, gpu=None),
    }

    assert actions._resolve_available_dispatch_resources(opts, running) == ["slot-1"]


def test_remote_launch_skips_busy_models_with_multiple_slots(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("[]", encoding="utf-8")
    launched: list[tuple[str, str]] = []

    def _item(model_name: str, dataset_slug: str) -> queue.QueueItem:
        return queue.QueueItem(
            job_name="code_human_eval",
            job_id=f"code_human_eval__{dataset_slug}_nocot_{actions.safe_slug(model_name)}",
            dataset_slug=dataset_slug,
            model_path=None,
            model_slug=actions.safe_slug(model_name),
            model_name=model_name,
            infer_base_url="http://127.0.0.1:19083/v1",
            infer_model=model_name,
        )

    monkeypatch.setattr(actions, "locate_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(actions, "_backup_run_config", lambda **_kwargs: None)
    monkeypatch.setattr(actions, "build_command", lambda *_args, **_kwargs: ["python", "-c", "pass"])

    def _fake_launch_job(job_id, _command, **_kwargs):
        model_name = next(item.model_name for item in items if item.job_id == job_id)
        launched.append((job_id, str(model_name)))
        return SimpleNamespace(pid=1000 + len(launched))

    monkeypatch.setattr(actions, "launch_job", _fake_launch_job)

    opts = DispatchOptions(
        log_dir=tmp_path / "log",
        pid_dir=tmp_path / "pid",
        run_log_dir=tmp_path / "run",
        job_order=("code_human_eval",),
        infer_base_url="http://127.0.0.1:19083/v1",
        infer_models=("remote-a", "remote-b", "remote-c"),
        max_concurrent_jobs=3,
    )
    items = [
        _item("remote-a", "human_eval_test"),
        _item("remote-a", "human_eval_cn_test"),
        _item("remote-b", "human_eval_test"),
        _item("remote-c", "human_eval_test"),
    ]

    actions.ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    actions._launch_queue_items(
        opts=opts,
        queue=items,
        available_resources=("slot-1", "slot-2", "slot-3"),
        question_counts={},
        batch_profiler=actions.BatchProfiler(tmp_path / "batch_cache.json"),
        pending_since={item.job_id: 1.0 for item in items},
        launch_times={},
        job_metadata={},
        lease_manager=None,
        claimed_job_ids=set(),
    )

    assert [model for _job_id, model in launched] == ["remote-a", "remote-b", "remote-c"]
    assert len(list((tmp_path / "pid").glob("*.pid"))) == 3


def test_scheduler_start_request_builds_remote_dispatch_options() -> None:
    request = SchedulerStartRequest(
        only_jobs=["free_response"],
        infer_base_url="http://127.0.0.1:8081",
        infer_models=["remote-demo"],
        infer_api_key="secret",
        infer_timeout_s=42.0,
        infer_max_workers=7,
        infer_protocol="vllm",
        max_concurrent_jobs=5,
    )

    opts = request.to_dispatch_options()

    assert opts.model_globs == ()
    assert opts.infer_base_url == "http://127.0.0.1:8081"
    assert opts.infer_models == ("remote-demo",)
    assert opts.infer_api_key == "secret"
    assert opts.infer_timeout_s == 42.0
    assert opts.infer_max_workers == 7
    assert opts.infer_protocol == "vllm"
    assert opts.max_concurrent_jobs == 5


def test_scheduler_cli_accepts_remote_inference_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "queue",
            "--infer-base-url",
            "http://127.0.0.1:8081",
            "--infer-models",
            "remote-demo",
            "--remote-batch-size",
            "64",
            "--infer-protocol",
            "vllm",
            "--max-concurrent-jobs",
            "4",
        ]
    )

    assert args.infer_base_url == "http://127.0.0.1:8081"
    assert args.infer_models == ["remote-demo"]
    assert args.remote_batch_size == 64
    assert args.infer_protocol == "vllm"
    assert args.max_concurrent_jobs == 4


def test_scheduler_cli_accepts_function_calling_runner_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "dispatch",
            "--function-cot-max-tokens",
            "4096",
            "--function-decision-max-tokens",
            "32",
            "--function-prompt-style",
            "rwkv_official_json",
            "--function-history-max-chars",
            "32000",
            "--function-prompt-max-chars",
            "8192",
            "--function-long-doc-mode",
            "off",
            "--function-tool-router-mode",
            "model",
            "--function-tool-router-max-tools",
            "8",
        ]
    )

    assert args.function_cot_max_tokens == 4096
    assert args.function_decision_max_tokens == 32
    assert args.function_prompt_style == "rwkv_official_json"
    assert args.function_history_max_chars == 32000
    assert args.function_prompt_max_chars == 8192
    assert args.function_long_doc_mode == "off"
    assert args.function_tool_router_mode == "model"
    assert args.function_tool_router_max_tools == 8


def test_function_calling_extra_args_only_apply_to_function_jobs(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("function_bfcl_v3",),
        function_prompt_style="rwkv_official_json",
        function_cot_max_tokens=4096,
        function_decision_max_tokens=32,
        function_max_steps=24,
        function_prompt_max_chars=8192,
        function_long_doc_mode="off",
        function_tool_router_mode="lexical",
        function_tool_router_max_tools=8,
    )

    assert actions._function_calling_extra_args(opts, JOB_CATALOGUE["function_bfcl_v3"]) == (
        "--prompt-style",
        "rwkv_official_json",
        "--cot-max-tokens",
        "4096",
        "--decision-max-tokens",
        "32",
        "--prompt-max-chars",
        "8192",
        "--long-doc-mode",
        "off",
        "--tool-router-mode",
        "lexical",
        "--tool-router-max-tools",
        "8",
        "--max-steps",
        "24",
    )
    assert actions._function_calling_extra_args(opts, JOB_CATALOGUE["free_response"]) == ()


def test_param_search_scripts_accept_remote_inference_args() -> None:
    free_response_args = parse_param_search_free_response_args(
        [
            "--dataset",
            "/tmp/gsm8k_test.jsonl",
            "--infer-base-url",
            "http://127.0.0.1:8081",
            "--infer-model",
            "remote-demo",
        ]
    )
    validate_inference_backend_args(free_response_args)
    assert resolve_backend_model_name(free_response_args) == "remote-demo"

    select_args = parse_param_search_select_args(
        [
            "--infer-base-url",
            "http://127.0.0.1:8081",
            "--infer-model",
            "remote-demo",
        ]
    )
    validate_inference_backend_args(select_args)
    assert resolve_backend_model_name(select_args) == "remote-demo"
