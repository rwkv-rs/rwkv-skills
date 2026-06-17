from __future__ import annotations

from src.bin import prepare_infer_swap_launch_bundle


def test_launch_bundle_parameters_and_argv_include_lightning_protocol() -> None:
    args = prepare_infer_swap_launch_bundle.parse_args(
        [
            "--infer-max-workers",
            "8",
            "--remote-batch-size",
            "8",
            "--only-datasets",
            "mcp_bench_test",
        ]
    )

    params = prepare_infer_swap_launch_bundle.build_launch_parameters(args)
    queue_argv = prepare_infer_swap_launch_bundle.build_run_infer_swap_eval_argv(args, action="queue")
    dispatch_argv = prepare_infer_swap_launch_bundle.build_run_infer_swap_eval_argv(args, action="dispatch")

    assert params["infer_protocol"] == "lightning"
    assert params["infer_seed_policy"] == "preserve"
    assert _flag_value(queue_argv, "--infer-protocol") == "lightning"
    assert _flag_value(dispatch_argv, "--infer-protocol") == "lightning"


def _flag_value(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]
