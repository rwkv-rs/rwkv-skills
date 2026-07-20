from __future__ import annotations

from pathlib import Path

from src.eval.evaluating import RunMode
from src.eval.scheduler import cli
from src.eval.scheduler.db_bootstrap import DbSchemaReport
from src.eval.scheduler.launch_config import expand_infer_model_groups, load_launch_profile
from src.eval.scheduler.remote_slots import parse_remote_model_slots


def test_default_scheduler_profile_carries_launch_parameters_from_toml() -> None:
    request = load_launch_profile("local-6gpu-full")
    opts = request.to_dispatch_options()

    assert request.profile == "local-6gpu-full"
    assert request.log_dir.endswith("logs/scheduler/local_6gpu_full")
    assert opts.run_mode is RunMode.AUTO
    assert opts.inference.base_url == "http://127.0.0.1:19083/v1"
    assert opts.inference.protocol == "vllm"
    assert opts.inference.remote_batch_size == 32
    assert len(opts.inference.models) == 34
    assert opts.inference.models[0] == "g1f15_s01=rwkv7-g1f-1.5b-20260419-ctx8192"
    assert opts.inference.models[-1] == "g1g72_s01=rwkv7-g1g-7.2b-20260523-ctx8192"
    assert "function_agent_tool_call" in opts.job_order
    assert opts.functions.candidate_router_mode == "auto"
    assert opts.disable_checker is True


def test_scheduler_run_dry_run_uses_profile_payload(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "scheduler.toml"
    profile.write_text(
        f"""
profile = "unit"
run_tag = "unit_run"

[paths]
log_dir = "{tmp_path}/logs/{{run_tag}}"
pid_dir = "{tmp_path}/logs/{{run_tag}}/pids"
run_log_dir = "{tmp_path}/runlogs/{{run_tag}}"

[inference]
base_url = "http://127.0.0.1:19083/v1"
models = ["slot0=demo-model"]
protocol = "vllm"
max_workers = 7
remote_batch_size = 5

[selection]
only_jobs = ["free_response"]
only_datasets = ["gsm8k"]

[runtime]
run_mode = "auto"
skip_missing_dataset = true
disable_checker = true

[function_calling]
candidate_router_mode = "auto"

[math]
prompt_max_chars = 8192
long_doc_mode = "lexical"

[knowledge]
prompt_max_chars = 4096
long_doc_mode = "off"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "check_db_schema",
        lambda: DbSchemaReport(
            host="127.0.0.1",
            port=5432,
            user="postgres",
            dbname="rwkv-eval",
            sslmode="disable",
            schema_path="scripts/schema.sql",
            database_ok=True,
            schema_ok=True,
        ),
    )
    captured = {}
    monkeypatch.setattr(cli, "action_queue", lambda opts: captured.setdefault("opts", opts) or [])

    rc = cli.main(["run", "--profile", str(profile), "--dry-run"])

    assert rc == 0
    opts = captured["opts"]
    assert opts.log_dir == tmp_path / "logs" / "unit_run"
    assert opts.run_log_dir == tmp_path / "runlogs" / "unit_run"
    assert opts.inference.models == ("slot0=demo-model",)
    assert opts.inference.max_workers == 7
    assert opts.inference.remote_batch_size == 5
    assert opts.job_order == ("free_response",)
    assert opts.only_dataset_slugs == ("gsm8k_test",)
    assert opts.disable_checker is True
    assert opts.math.prompt_max_chars == 8192
    assert opts.math.long_doc_mode == "lexical"
    assert opts.knowledge.prompt_max_chars == 4096
    assert opts.knowledge.long_doc_mode == "off"
    assert (tmp_path / "logs" / "unit_run" / "resolved_config.json").exists()


def test_model_groups_can_pin_each_slot_to_an_endpoint() -> None:
    specs = expand_infer_model_groups(
        (),
        (
            {
                "slot_prefix": "demo",
                "model": "demo-model",
                "slots": 2,
                "base_urls": ["http://127.0.0.1:9001/v1", "http://127.0.0.1:9002/v1"],
            },
        ),
        slots_per_model=1,
    )

    slots = parse_remote_model_slots(specs)
    assert [slot.model for slot in slots] == ["demo-model", "demo-model"]
    assert [slot.base_url for slot in slots] == [
        "http://127.0.0.1:9001/v1",
        "http://127.0.0.1:9002/v1",
    ]
