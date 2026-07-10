from __future__ import annotations

"""Launch the RWKV Skills dashboard API.

Serves leaderboard, eval records, context, and scheduler admin APIs. Run the
Next.js frontend in ``client/`` separately; it rewrites ``/api`` back to this
server.
"""

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RWKV Skills evaluation dashboard server")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="bind port (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="enable auto-reload (development)")
    args = parser.parse_args(argv)

    import uvicorn

    if args.reload:
        uvicorn.run("src.dashboard.web.api:app", host=args.host, port=args.port, reload=True)
    else:
        from src.dashboard.web.api import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
