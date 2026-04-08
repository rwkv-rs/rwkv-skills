from __future__ import annotations

from pathlib import Path

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
    )

    assert "--infer-base-url" in command
    assert "--infer-model" in command
    assert "--infer-api-key" in command
    assert "--infer-timeout-s" in command
    assert "--infer-max-workers" in command
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


def test_scheduler_start_request_builds_remote_dispatch_options() -> None:
    request = SchedulerStartRequest(
        only_jobs=["free_response"],
        infer_base_url="http://127.0.0.1:8081",
        infer_models=["remote-demo"],
        infer_api_key="secret",
        infer_timeout_s=42.0,
        infer_max_workers=7,
        max_concurrent_jobs=5,
    )

    opts = request.to_dispatch_options()

    assert opts.model_globs == ()
    assert opts.infer_base_url == "http://127.0.0.1:8081"
    assert opts.infer_models == ("remote-demo",)
    assert opts.infer_api_key == "secret"
    assert opts.infer_timeout_s == 42.0
    assert opts.infer_max_workers == 7
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
            "--max-concurrent-jobs",
            "4",
        ]
    )

    assert args.infer_base_url == "http://127.0.0.1:8081"
    assert args.infer_models == ["remote-demo"]
    assert args.max_concurrent_jobs == 4


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
