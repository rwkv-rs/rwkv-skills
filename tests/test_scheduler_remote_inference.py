from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from src.bin.param_search_free_response import parse_args as parse_param_search_free_response_args
from src.bin.param_search_select import parse_args as parse_param_search_select_args
from src.eval.scheduler import actions, actions_base, action_dispatch, queue
from src.eval.scheduler.actions import DispatchOptions, FunctionCallingConfig, InferenceConfig, KnowledgeConfig, MathConfig
from src.eval.scheduler.admin import SchedulerStartRequest
from src.eval.scheduler.backpressure import (
    RemoteConcurrencyBudget,
    RemoteModelBackpressure,
    compute_remote_concurrency_budgets,
    parse_remote_backpressure,
)
from src.eval.scheduler.cli import _expand_infer_model_slots, build_parser
from src.eval.scheduler.jobs import JOB_CATALOGUE
from src.eval.scheduler.remote_slots import infer_workers_for_model
from src.eval.scheduler.state import RunningEntry
from src.infer.backend import resolve_backend_model_name, validate_inference_backend_args


def test_remote_backend_import_does_not_load_local_cuda_engines() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.infer.backend; "
                "print('src.infer.model' in sys.modules, "
                "'src.infer.lightning_engine' in sys.modules, "
                "'src.infer.engine' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False False"


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


def test_build_queue_deduplicates_remote_slot_aliases() -> None:
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
        infer_models=(
            "g1f15_slot1=rwkv7-g1f-1.5b-20260419-ctx8192",
            "g1f15_slot2=rwkv7-g1f-1.5b-20260419-ctx8192",
        ),
    )

    assert len(items) == 1
    assert items[0].model_name == "rwkv7-g1f-1.5b-20260419-ctx8192"
    assert items[0].infer_model == "rwkv7-g1f-1.5b-20260419-ctx8192"
    assert "slot" not in items[0].job_id


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


def test_param_size_worker_profile_gives_smaller_models_more_workers() -> None:
    assert (
        infer_workers_for_model(
            "rwkv7-g1f-1.5b-20260419-ctx8192",
            default_workers=32,
            profile="param-size",
        )
        == 256
    )
    assert (
        infer_workers_for_model(
            "rwkv7-g1f-2.9b-20260420-ctx8192",
            default_workers=32,
            profile="param-size",
        )
        == 128
    )
    assert (
        infer_workers_for_model(
            "rwkv7-g1f-7.2b-20260428-ctx8192",
            default_workers=32,
            profile="param-size",
        )
        == 96
    )
    assert (
        infer_workers_for_model(
            "rwkv7-g1f-14b-20260428-ctx8192",
            default_workers=32,
            profile="param-size",
        )
        == 48
    )


def test_remote_dispatch_resources_use_model_slots(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:8081",
            models=("remote-a", "remote-b"),
        ),
    )
    running = {
        "code_human_eval__human_eval_test_nocot_remote_a": RunningEntry(pid=101, gpu=None),
        "unrelated-model-job": RunningEntry(pid=102, gpu=None),
    }

    assert actions._resolve_available_dispatch_resources(opts, running) == ["model:remote_b"]


def test_remote_dispatch_resources_respect_backpressure_without_adding_slots(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:8081",
            models=("remote-a", "remote-b"),
        ),
    )
    budgets = {
        "remote_a": RemoteConcurrencyBudget(
            model="remote-a",
            model_slug="remote_a",
            infer_max_workers=4,
            remote_batch_size=4,
            launch_allowed=False,
            reason="backend_queue_pending",
            pending_queue=1,
        ),
        "remote_b": RemoteConcurrencyBudget(
            model="remote-b",
            model_slug="remote_b",
            infer_max_workers=4,
            remote_batch_size=4,
        ),
    }

    assert actions._resolve_available_dispatch_resources(opts, {}, remote_budgets=budgets) == ["model:remote_b"]


def test_remote_dispatch_resources_support_alias_slots(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:8081",
            models=("slot-a=remote-a", "slot-b=remote-a"),
        ),
    )
    running = {
        "free_response__gsm8k_test_nocot_remote_a": RunningEntry(pid=101, gpu="model:slot_a"),
    }

    assert actions._resolve_available_dispatch_resources(opts, running) == ["model:slot_b"]


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

    monkeypatch.setattr(actions_base, "locate_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(action_dispatch, "_backup_run_config", lambda **_kwargs: None)
    monkeypatch.setattr(action_dispatch, "build_command", lambda *_args, **_kwargs: ["python", "-c", "pass"])

    def _fake_launch_job(job_id, _command, **_kwargs):
        model_name = next(item.model_name for item in items if item.job_id == job_id)
        launched.append((job_id, str(model_name)))
        return SimpleNamespace(pid=1000 + len(launched))

    monkeypatch.setattr(actions_base, "launch_job", _fake_launch_job)

    opts = DispatchOptions(
        log_dir=tmp_path / "log",
        pid_dir=tmp_path / "pid",
        run_log_dir=tmp_path / "run",
        job_order=("code_human_eval",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:19083/v1",
            models=("remote-a", "remote-b", "remote-c"),
        ),
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
        available_resources=("model:remote_a", "model:remote_b", "model:remote_c"),
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


def test_remote_launch_uses_alias_slots_for_same_model(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("[]", encoding="utf-8")
    launched: list[tuple[str, str]] = []
    child_envs: list[dict[str, str]] = []

    def _item(dataset_slug: str) -> queue.QueueItem:
        return queue.QueueItem(
            job_name="code_human_eval",
            job_id=f"code_human_eval__{dataset_slug}_nocot_{actions.safe_slug('remote-a')}",
            dataset_slug=dataset_slug,
            model_path=None,
            model_slug=actions.safe_slug("remote-a"),
            model_name="remote-a",
            infer_base_url="http://127.0.0.1:19083/v1",
            infer_model="remote-a",
        )

    monkeypatch.setattr(actions_base, "locate_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(action_dispatch, "_backup_run_config", lambda **_kwargs: None)
    monkeypatch.setattr(action_dispatch, "build_command", lambda *_args, **_kwargs: ["python", "-c", "pass"])

    def _fake_launch_job(job_id, _command, **_kwargs):
        child_envs.append(dict(_kwargs.get("env", {})))
        launched.append((job_id, str(_kwargs.get("env", {}).get("RWKV_SKILLS_INFER_MODEL"))))
        return SimpleNamespace(pid=1000 + len(launched))

    monkeypatch.setattr(actions_base, "launch_job", _fake_launch_job)

    opts = DispatchOptions(
        log_dir=tmp_path / "log",
        pid_dir=tmp_path / "pid",
        run_log_dir=tmp_path / "run",
        job_order=("code_human_eval",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:19083/v1",
            models=("slot-a=remote-a", "slot-b=remote-a"),
        ),
    )
    items = [_item("human_eval_test"), _item("human_eval_cn_test")]

    actions.ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    actions._launch_queue_items(
        opts=opts,
        queue=items,
        available_resources=("model:slot_a", "model:slot_b"),
        question_counts={},
        batch_profiler=actions.BatchProfiler(tmp_path / "batch_cache.json"),
        pending_since={item.job_id: 1.0 for item in items},
        launch_times={},
        job_metadata={},
        lease_manager=None,
        claimed_job_ids=set(),
    )

    assert [model for _job_id, model in launched] == ["remote-a", "remote-a"]
    assert [env.get("CUDA_VISIBLE_DEVICES") for env in child_envs] == ["", ""]
    assert sorted(path.read_text(encoding="utf-8").splitlines()[1] for path in opts.pid_dir.glob("*.pid")) == [
        "model:slot_a",
        "model:slot_b",
    ]


def test_remote_launch_uses_backpressure_budget(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("[]", encoding="utf-8")
    item = queue.QueueItem(
        job_name="code_human_eval",
        job_id=f"code_human_eval__human_eval_test_nocot_{actions.safe_slug('remote-a')}",
        dataset_slug="human_eval_test",
        model_path=None,
        model_slug=actions.safe_slug("remote-a"),
        model_name="remote-a",
        infer_base_url="http://127.0.0.1:19083/v1",
        infer_model="remote-a",
    )
    captured: dict[str, int | None] = {}

    monkeypatch.setattr(actions_base, "locate_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(action_dispatch, "_backup_run_config", lambda **_kwargs: captured.update(_kwargs))

    def _fake_build_command(*_args, **kwargs):
        captured["batch_size"] = kwargs["batch_size"]
        captured["infer_max_workers"] = kwargs["infer_max_workers"]
        return ["python", "-c", "pass"]

    monkeypatch.setattr(action_dispatch, "build_command", _fake_build_command)
    monkeypatch.setattr(actions_base, "launch_job", lambda *_args, **_kwargs: SimpleNamespace(pid=1001))

    opts = DispatchOptions(
        log_dir=tmp_path / "log",
        pid_dir=tmp_path / "pid",
        run_log_dir=tmp_path / "run",
        job_order=("code_human_eval",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:19083/v1",
            models=("remote-a",),
            max_workers=64,
            remote_batch_size=64,
        ),
    )
    budget = RemoteConcurrencyBudget(
        model="remote-a",
        model_slug="remote_a",
        infer_max_workers=12,
        remote_batch_size=8,
        reason="backpressure_ok",
        max_batch_size=8,
    )

    actions.ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    actions._launch_queue_items(
        opts=opts,
        queue=[item],
        available_resources=("model:remote_a",),
        question_counts={},
        batch_profiler=actions.BatchProfiler(tmp_path / "batch_cache.json"),
        pending_since={item.job_id: 1.0},
        launch_times={},
        job_metadata={},
        lease_manager=None,
        claimed_job_ids=set(),
        remote_budgets={"remote_a": budget},
    )

    assert captured["batch_size"] == 8
    assert captured["infer_max_workers"] == 12
    assert captured["budget_reason"] == "backpressure_ok"


def test_remote_launch_continues_past_empty_model_slot(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("[]", encoding="utf-8")
    launched: list[str] = []
    item = queue.QueueItem(
        job_name="code_human_eval",
        job_id=f"code_human_eval__human_eval_test_nocot_{actions.safe_slug('remote-c')}",
        dataset_slug="human_eval_test",
        model_path=None,
        model_slug=actions.safe_slug("remote-c"),
        model_name="remote-c",
        infer_base_url="http://127.0.0.1:19083/v1",
        infer_model="remote-c",
    )

    monkeypatch.setattr(actions_base, "locate_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(action_dispatch, "_backup_run_config", lambda **_kwargs: None)
    monkeypatch.setattr(action_dispatch, "build_command", lambda *_args, **_kwargs: ["python", "-c", "pass"])

    def _fake_launch_job(job_id, _command, **_kwargs):
        launched.append(job_id)
        return SimpleNamespace(pid=1000 + len(launched))

    monkeypatch.setattr(actions_base, "launch_job", _fake_launch_job)

    opts = DispatchOptions(
        log_dir=tmp_path / "log",
        pid_dir=tmp_path / "pid",
        run_log_dir=tmp_path / "run",
        job_order=("code_human_eval",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:19083/v1",
            models=("remote-b", "remote-c"),
        ),
    )
    actions.ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    actions._launch_queue_items(
        opts=opts,
        queue=[item],
        available_resources=("model:remote_b", "model:remote_c"),
        question_counts={},
        batch_profiler=actions.BatchProfiler(tmp_path / "batch_cache.json"),
        pending_since={item.job_id: 1.0},
        launch_times={},
        job_metadata={},
        lease_manager=None,
        claimed_job_ids=set(),
    )

    assert launched == [item.job_id]


def test_scheduler_start_request_builds_remote_dispatch_options() -> None:
    request = SchedulerStartRequest(
        only_jobs=["free_response"],
        infer_base_url="http://127.0.0.1:8081",
        infer_models=["remote-demo"],
        infer_api_key="secret",
        infer_timeout_s=42.0,
        infer_max_workers=7,
        infer_protocol="vllm",
    )

    opts = request.to_dispatch_options()

    assert opts.model_globs == ()
    assert opts.inference.base_url == "http://127.0.0.1:8081"
    assert opts.inference.models == ("remote-demo",)
    assert opts.inference.api_key == "secret"
    assert opts.inference.timeout_s == 42.0
    assert opts.inference.max_workers == 7
    assert opts.inference.protocol == "vllm"
    assert opts.inference.backpressure is True


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
            "--infer-worker-profile",
            "param-size",
            "--infer-backpressure-timeout-s",
            "1.5",
            "--infer-backpressure-pending-high-watermark",
            "2",
            "--infer-budget-min-workers",
            "3",
            "--coding-swebench-max-prompt-chars",
            "18000",
        ]
    )

    assert args.infer_base_url == "http://127.0.0.1:8081"
    assert args.infer_models == ["remote-demo"]
    assert args.remote_batch_size == 64
    assert args.infer_protocol == "vllm"
    assert args.infer_worker_profile == "param-size"
    assert args.infer_backpressure_timeout_s == 1.5
    assert args.infer_backpressure_pending_high_watermark == 2
    assert args.infer_budget_min_workers == 3
    assert args.coding_swebench_max_prompt_chars == 18000


def test_scheduler_slot_expansion_preserves_explicit_slots() -> None:
    assert _expand_infer_model_slots(("slot-a=remote-a", "slot-b=remote-a"), 2) == (
        "slot-a=remote-a",
        "slot-b=remote-a",
    )


def test_scheduler_slot_expansion_expands_bare_model_names() -> None:
    assert _expand_infer_model_slots(("remote-a",), 2) == (
        "remote-a-s0=remote-a",
        "remote-a-s1=remote-a",
    )


def test_parse_remote_backpressure_payload_uses_router_aggregate() -> None:
    parsed = parse_remote_backpressure(
        {
            "models": {
                "remote-a": {
                    "model": "remote-a",
                    "status": "ok",
                    "route_count": 1,
                    "aggregate": {
                        "ok_route_count": 1,
                        "pending_queue": 0,
                        "prefill_reserved_bsz": 5,
                        "max_batch_size": 16,
                        "failed_batches": 0,
                        "last_total_tok_s": 12.5,
                    },
                }
            }
        }
    )

    signal = parsed["remote_a"]
    assert signal.status == "ok"
    assert signal.ok_route_count == 1
    assert signal.pending_queue == 0
    assert signal.prefill_reserved_bsz == 5
    assert signal.max_batch_size == 16


def test_prefill_reserved_backpressure_blocks_launches() -> None:
    budgets = compute_remote_concurrency_budgets(
        infer_models=("remote-a",),
        backpressure={
            "remote_a": RemoteModelBackpressure(
                model="remote-a",
                model_slug="remote_a",
                status="ok",
                route_count=1,
                ok_route_count=1,
                pending_queue=0,
                prefill_reserved_bsz=90,
                max_batch_size=400,
            )
        },
        default_infer_max_workers=8,
        default_remote_batch_size=64,
    )

    budget = budgets["remote_a"]
    assert budget.launch_allowed is False
    assert budget.reason == "backend_prefill_reserved"
    assert budget.prefill_reserved_bsz == 90


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
            "--function-judge-max-workers",
            "1",
            "--math-judge-max-workers",
            "2",
            "--sample-workers",
            "8",
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
    assert args.function_judge_max_workers == 1
    assert args.math_judge_max_workers == 2
    assert args.sample_workers == 8
    assert args.function_long_doc_mode == "off"
    assert args.function_tool_router_mode == "model"
    assert args.function_tool_router_max_tools == 8


def test_function_calling_extra_args_only_apply_to_function_jobs(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("function_bfcl_v3",),
        inference=InferenceConfig(
            sample_workers=8,
        ),
        functions=FunctionCallingConfig(
            prompt_style="rwkv_official_json",
            cot_max_tokens=4096,
            decision_max_tokens=32,
            max_steps=24,
            prompt_max_chars=8192,
            judge_max_workers=1,
            long_doc_mode="off",
            tool_router_mode="lexical",
            tool_router_max_tools=8,
        ),
    )

    assert actions._function_calling_extra_args(opts, JOB_CATALOGUE["function_bfcl_v3"]) == (
        "--prompt-style",
        "rwkv_official_json",
        "--sample-workers",
        "8",
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
    mcp_args = actions._function_calling_extra_args(opts, JOB_CATALOGUE["function_mcp_bench"])
    assert "--prompt-style" in mcp_args
    assert "--sample-workers" not in mcp_args
    assert actions._function_calling_extra_args(opts, JOB_CATALOGUE["free_response"]) == ()
    browsecomp_args = actions._function_calling_extra_args(opts, JOB_CATALOGUE["function_browsecomp"])
    assert "--judge-max-workers" in browsecomp_args
    assert "1" in browsecomp_args


def test_maths_extra_args_only_apply_to_llm_judge_jobs(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response_judge",),
        math=MathConfig(judge_max_workers=2),
    )

    assert actions._maths_extra_args(opts, JOB_CATALOGUE["free_response_judge"]) == (
        "--judge-max-workers",
        "2",
    )
    assert actions._maths_extra_args(opts, JOB_CATALOGUE["free_response"]) == ()
    assert actions._maths_extra_args(opts, JOB_CATALOGUE["function_browsecomp"]) == ()


def test_non_fc_long_doc_extra_args_are_scoped_by_runner_group(tmp_path: Path) -> None:
    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response", "multi_choice_plain"),
        math=MathConfig(prompt_max_chars=8192, long_doc_mode="lexical"),
        knowledge=KnowledgeConfig(prompt_max_chars=4096, long_doc_mode="off"),
    )

    assert actions._maths_extra_args(opts, JOB_CATALOGUE["free_response"]) == (
        "--prompt-max-chars",
        "8192",
        "--long-doc-mode",
        "lexical",
    )
    assert actions._maths_extra_args(opts, JOB_CATALOGUE["multi_choice_plain"]) == ()
    assert actions._knowledge_extra_args(opts, JOB_CATALOGUE["multi_choice_plain"]) == (
        "--prompt-max-chars",
        "4096",
        "--long-doc-mode",
        "off",
    )
    assert actions._knowledge_extra_args(opts, JOB_CATALOGUE["free_response"]) == ()


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
