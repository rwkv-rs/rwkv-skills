from __future__ import annotations

"""Preflight checks before launching remote-inference scheduler evals."""

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import psycopg

from src.bin.verify_remote_infer_swap import verify_remote_infer_swap
from src.eval.scheduler.config import DEFAULT_DB_CONFIG, DBConfig
from src.infer.backend import normalize_api_base


DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE = 2


@dataclass(slots=True, frozen=True)
class PreflightCheck:
    name: str
    status: str
    details: Mapping[str, object]
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}


@dataclass(slots=True, frozen=True)
class RemoteEvalPreflightResult:
    infer_base_url: str
    infer_model: str
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "infer_base_url": self.infer_base_url,
            "infer_model": self.infer_model,
            "checks": [asdict(check) | {"ok": check.ok} for check in self.checks],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight remote inference eval readiness")
    parser.add_argument("--infer-base-url", "--base-url", required=True, help="Remote inference base URL")
    parser.add_argument("--infer-model", "--model", required=True, help="Remote model name")
    parser.add_argument("--infer-api-key", "--api-key", default="", help="Remote inference bearer token")
    parser.add_argument("--infer-timeout-s", "--timeout-s", type=float, default=600.0, help="Inference timeout")
    parser.add_argument("--protocols", default="vllm", help="Protocols for generation smoke")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
        help="Prompt count per protocol smoke",
    )
    parser.add_argument("--max-tokens", type=int, default=8, help="Max generated tokens for smoke")
    parser.add_argument("--skip-db", action="store_true", help="Skip scheduler DB connection check")
    parser.add_argument("--db-host", help="Override scheduler DB host for this preflight")
    parser.add_argument("--db-port", type=int, help="Override scheduler DB port for this preflight")
    parser.add_argument("--db-user", help="Override scheduler DB user for this preflight")
    parser.add_argument("--db-name", help="Override scheduler DB database name for this preflight")
    parser.add_argument("--db-sslmode", help="Override scheduler DB sslmode for this preflight")
    parser.add_argument("--db-timeout-s", type=float, default=5.0, help="Scheduler DB connection timeout")
    parser.add_argument("--output-json", help="Optional JSON result path")
    parser.add_argument(
        "--stdout",
        choices=("json", "summary", "none"),
        default="json",
        help="Stdout format; use summary for remote shell preflight runs",
    )
    return parser.parse_args(argv)


def run_preflight(
    *,
    infer_base_url: str,
    infer_model: str,
    infer_api_key: str = "",
    infer_timeout_s: float = 600.0,
    protocols: Sequence[str] = ("vllm",),
    batch_size: int = DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
    max_tokens: int = 8,
    check_db: bool = True,
    db_timeout_s: float = 5.0,
    db_config: DBConfig = DEFAULT_DB_CONFIG,
) -> RemoteEvalPreflightResult:
    checks = [
        _run_timed_check(
            "infer_health",
            lambda: _check_infer_health(infer_base_url=infer_base_url, timeout_s=min(float(infer_timeout_s), 30.0)),
        ),
        _run_timed_check(
            "infer_models",
            lambda: _check_infer_models(
                infer_base_url=infer_base_url,
                infer_model=infer_model,
                timeout_s=min(float(infer_timeout_s), 30.0),
                api_key=infer_api_key,
            ),
        ),
        _run_timed_check(
            "protocol_smoke",
            lambda: _check_protocol_smoke(
                infer_base_url=infer_base_url,
                infer_model=infer_model,
                infer_api_key=infer_api_key,
                infer_timeout_s=infer_timeout_s,
                protocols=protocols,
                batch_size=batch_size,
                max_tokens=max_tokens,
            ),
        ),
        _run_timed_check("scheduler_db", lambda: _check_scheduler_db(db_config=db_config, timeout_s=db_timeout_s))
        if check_db
        else _skipped("scheduler_db"),
    ]
    return RemoteEvalPreflightResult(
        infer_base_url=infer_base_url,
        infer_model=infer_model,
        checks=tuple(checks),
    )


def write_preflight_result(path: Path, result: RemoteEvalPreflightResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def format_preflight_summary(result: RemoteEvalPreflightResult) -> str:
    lines = [
        f"ok={str(result.ok).lower()}",
        f"infer_base_url={result.infer_base_url}",
        f"infer_model={result.infer_model}",
    ]
    for check in result.checks:
        line = f"{check.name}: {check.status} elapsed_s={check.elapsed_s:.3f}"
        if check.name == "protocol_smoke":
            protocols = _protocol_status_summary(check.details)
            if protocols:
                line = f"{line} protocols={protocols}"
        elif check.name == "infer_models":
            models = check.details.get("models")
            if isinstance(models, list):
                line = f"{line} models={','.join(str(model) for model in models)}"
        elif check.name == "scheduler_db":
            db_target = _db_target_summary(check.details)
            if db_target:
                line = f"{line} db={db_target}"
        if check.error:
            line = f"{line} error={check.error}"
        lines.append(line)
    return "\n".join(lines)


def resolve_db_config_from_args(args: argparse.Namespace, base_config: DBConfig | None = None) -> DBConfig:
    base = base_config or DEFAULT_DB_CONFIG
    values = {
        "host": str(args.db_host) if args.db_host else base.host,
        "port": int(args.db_port) if args.db_port is not None else base.port,
        "user": str(args.db_user) if args.db_user else base.user,
        "password": getattr(base, "password", ""),
        "dbname": str(args.db_name) if args.db_name else base.dbname,
        "sslmode": str(args.db_sslmode) if args.db_sslmode else getattr(base, "sslmode", "prefer"),
        "startup_recovery": getattr(base, "startup_recovery", False),
    }
    supported = {field.name for field in fields(DBConfig)}
    return DBConfig(**{key: value for key, value in values.items() if key in supported})


def _check_infer_health(*, infer_base_url: str, timeout_s: float) -> PreflightCheck:
    url = f"{_root_base_url(infer_base_url)}/healthz"
    try:
        status, data = _get_json(url, timeout_s=timeout_s)
    except Exception as exc:
        return PreflightCheck("infer_health", "failed", {"url": url}, str(exc))
    ok = status == 200 and isinstance(data, dict) and data.get("status") == "ok"
    return PreflightCheck("infer_health", "ok" if ok else "failed", {"url": url, "status_code": status, "body": data})


def _check_infer_models(
    *,
    infer_base_url: str,
    infer_model: str,
    timeout_s: float,
    api_key: str,
) -> PreflightCheck:
    url = f"{normalize_api_base(infer_base_url)}/models"
    try:
        status, data = _get_json(url, timeout_s=timeout_s, api_key=api_key)
    except Exception as exc:
        return PreflightCheck("infer_models", "failed", {"url": url, "model": infer_model}, str(exc))
    model_ids: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if isinstance(item, dict) and item.get("id") is not None:
                model_ids.append(str(item["id"]))
    ok = status == 200 and infer_model in model_ids
    return PreflightCheck(
        "infer_models",
        "ok" if ok else "failed",
        {"url": url, "status_code": status, "model": infer_model, "models": model_ids},
    )


def _check_protocol_smoke(
    *,
    infer_base_url: str,
    infer_model: str,
    infer_api_key: str,
    infer_timeout_s: float,
    protocols: Sequence[str],
    batch_size: int,
    max_tokens: int,
) -> PreflightCheck:
    try:
        result = verify_remote_infer_swap(
            base_url=infer_base_url,
            model=infer_model,
            api_key=infer_api_key,
            timeout_s=infer_timeout_s,
            protocols=tuple(str(item) for item in protocols),
            batch_size=batch_size,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return PreflightCheck("protocol_smoke", "failed", {"protocols": list(protocols)}, str(exc))
    return PreflightCheck(
        "protocol_smoke",
        "ok" if result.ok else "failed",
        result.to_dict(),
    )


def _check_scheduler_db(*, db_config: DBConfig, timeout_s: float) -> PreflightCheck:
    conninfo = _db_conninfo(db_config, connect_timeout_s=timeout_s)
    details = {
        "host": db_config.host,
        "port": db_config.port,
        "user": db_config.user,
        "dbname": db_config.dbname,
        "sslmode": getattr(db_config, "sslmode", ""),
    }
    try:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute("select 1")
                cursor.fetchone()
    except Exception as exc:
        return PreflightCheck("scheduler_db", "failed", details, str(exc))
    return PreflightCheck("scheduler_db", "ok", details)


def _get_json(url: str, *, timeout_s: float, api_key: str = "") -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(url, headers=headers, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=max(float(timeout_s), 1.0)) as response:
            raw = response.read()
            return int(response.status), json.loads(raw.decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raw = exc.read()
        try:
            data: object = json.loads(raw.decode("utf-8"))
        except Exception:
            data = raw.decode("utf-8", errors="replace")
        return int(exc.code), data


def _root_base_url(base_url: str) -> str:
    normalized = str(base_url).rstrip("/")
    for suffix in ("/openai/v1", "/v1", "/openai"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("/")
    return normalized


def _db_conninfo(config: DBConfig, *, connect_timeout_s: float) -> str:
    parts = [
        f"host={config.host}",
        f"port={int(config.port)}",
        f"user={config.user}",
        f"dbname={config.dbname}",
        f"connect_timeout={max(1, int(connect_timeout_s))}",
    ]
    if config.password:
        parts.append(f"password={config.password}")
    sslmode = str(getattr(config, "sslmode", "") or "").strip()
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)


def _skipped(name: str) -> PreflightCheck:
    return PreflightCheck(name, "skipped", {})


def _run_timed_check(name: str, callback: Callable[[], PreflightCheck]) -> PreflightCheck:
    started = time.perf_counter()
    try:
        check = callback()
    except Exception as exc:  # pragma: no cover - defensive guard
        check = PreflightCheck(name, "failed", {}, str(exc))
    elapsed_s = max(0.0, time.perf_counter() - started)
    return PreflightCheck(
        name=check.name,
        status=check.status,
        details=check.details,
        error=check.error,
        elapsed_s=elapsed_s,
    )


def _protocol_status_summary(details: Mapping[str, object]) -> str:
    raw_protocols = details.get("protocols")
    if not isinstance(raw_protocols, list):
        return ""
    pairs: list[str] = []
    for item in raw_protocols:
        if not isinstance(item, dict):
            continue
        protocol = str(item.get("protocol") or "").strip()
        if not protocol:
            continue
        status = "ok" if item.get("ok") is True else str(item.get("status") or "failed")
        pairs.append(f"{protocol}:{status}")
    return ",".join(pairs)


def _db_target_summary(details: Mapping[str, object]) -> str:
    host = details.get("host")
    port = details.get("port")
    dbname = details.get("dbname")
    user = details.get("user")
    if host is None or port is None or dbname is None or user is None:
        return ""
    return f"{host}:{port}/{dbname} user={user}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_preflight(
        infer_base_url=str(args.infer_base_url),
        infer_model=str(args.infer_model),
        infer_api_key=str(args.infer_api_key or ""),
        infer_timeout_s=float(args.infer_timeout_s),
        protocols=tuple(str(args.protocols).split(",")),
        batch_size=int(args.batch_size),
        max_tokens=int(args.max_tokens),
        check_db=not bool(args.skip_db),
        db_timeout_s=float(args.db_timeout_s),
        db_config=resolve_db_config_from_args(args),
    )
    payload = result.to_dict()
    if args.stdout == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    elif args.stdout == "summary":
        print(format_preflight_summary(result), flush=True)
    if args.output_json:
        write_preflight_result(Path(args.output_json).expanduser(), result)
    return 0 if result.ok else 1


__all__ = [
    "DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE",
    "PreflightCheck",
    "RemoteEvalPreflightResult",
    "main",
    "parse_args",
    "format_preflight_summary",
    "resolve_db_config_from_args",
    "run_preflight",
    "write_preflight_result",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
