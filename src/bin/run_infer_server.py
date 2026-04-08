from __future__ import annotations

"""Launch the standalone RWKV infer service."""

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from src.infer.backend import LocalInferenceBackend
from src.infer.model import ModelLoadConfig
from src.infer.server import create_app
from src.infer.service import InferenceService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone RWKV infer service")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--engine-mode",
        choices=("classic", "lightning"),
        default="classic",
        help="Local inference engine implementation to use",
    )
    parser.add_argument(
        "--state-db-path",
        help="Path to the local sqlite state cache database used by the lightning engine",
    )
    parser.add_argument("--model-name", help="Public model name exposed by the infer API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument("--api-key", default="", help="Optional bearer token required by the infer API")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Max number of queued requests per infer batch")
    parser.add_argument("--batch-collect-ms", type=int, default=5, help="How long the worker waits to collect a batch")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backend = LocalInferenceBackend.from_model_config(
        ModelLoadConfig(
            weights_path=args.model_path,
            device=args.device,
        ),
        engine_mode=str(args.engine_mode),
        state_db_path=None if args.state_db_path in (None, "") else str(args.state_db_path),
    )
    if args.model_name:
        backend.model_name = str(args.model_name)
    else:
        backend.model_name = Path(args.model_path).stem
    service = InferenceService(
        backend,
        max_batch_size=args.max_batch_size,
        batch_collect_ms=args.batch_collect_ms,
    )
    app = create_app(service, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
