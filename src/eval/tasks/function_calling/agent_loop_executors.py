from __future__ import annotations

"""Executors for the generic multi-turn agent-loop benchmark channel.

An executor turns the model's JSON tool call into a real environment action
(recorded manifest replay, sandbox shell command, or MCP worker call) and
returns a JSON-serializable outcome that is fed back to the model as
``User: Function output:\\n<json>``.
"""

import fcntl
import gzip
import hashlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.eval.tasks.function_calling.context_budget import truncate_text

DEFAULT_COMMAND_TIMEOUT_S = 60.0
DEFAULT_MAX_OUTPUT_CHARS = 8000
_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
)
_SANDBOX_GIT_ENV_NAMES = (
    "GIT_TERMINAL_PROMPT",
    "GIT_ASKPASS",
    "GIT_HTTP_LOW_SPEED_LIMIT",
    "GIT_HTTP_LOW_SPEED_TIME",
)

WEB_SEARCH_API_URL_ENV = "RWKV_WEB_SEARCH_API_URL"
WEB_SEARCH_API_KEY_ENV = "RWKV_WEB_SEARCH_API_KEY"
WEB_SEARCH_API_KEY_HEADER_ENV = "RWKV_WEB_SEARCH_API_KEY_HEADER"

_WEB_SEARCH_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "web_search",
        "description": "Search the web and return result snippets as JSON.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and return its text content (truncated).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
)


def _decode_http_body(body: bytes, headers: Any) -> str:
    encoding = str(headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        body = gzip.decompress(body)
    elif "deflate" in encoding:
        try:
            body = zlib.decompress(body)
        except zlib.error:
            body = zlib.decompress(body, -zlib.MAX_WBITS)
    charset_getter = getattr(headers, "get_content_charset", None)
    charset = charset_getter() if callable(charset_getter) else None
    return body.decode(charset or "utf-8", errors="replace")


def _docker_proxy_env_args() -> list[str]:
    args: list[str] = []
    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            args.extend(["-e", f"{name}={value}"])
    return args


def _docker_proxy_build_args() -> list[str]:
    args: list[str] = []
    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            args.extend(["--build-arg", f"{name}={value}"])
    return args


def _sandbox_command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_ASKPASS", "true")
    env.setdefault(
        "GIT_HTTP_LOW_SPEED_LIMIT",
        os.environ.get("RWKV_AGENT_LOOP_GIT_HTTP_LOW_SPEED_LIMIT", "1000"),
    )
    env.setdefault(
        "GIT_HTTP_LOW_SPEED_TIME",
        os.environ.get("RWKV_AGENT_LOOP_GIT_HTTP_LOW_SPEED_TIME", "20"),
    )
    return env


def _docker_sandbox_env_args() -> list[str]:
    env = _sandbox_command_env()
    args: list[str] = []
    for name in _SANDBOX_GIT_ENV_NAMES:
        value = env.get(name)
        if value:
            args.extend(["-e", f"{name}={value}"])
    return args


def _safe_docker_token(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "agent-loop"

_SHELL_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "bash",
        "description": "Run a shell command in the task workspace and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a UTF-8 text file inside the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
)


@dataclass(frozen=True, slots=True)
class ExecutorSpec:
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentLoopStepOutcome:
    ok: bool
    output: Any = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class AgentLoopExecutor(Protocol):
    def open(self) -> tuple[dict[str, Any], ...]:
        """Start the environment; returns extra tools to expose to the model."""

    def execute(self, name: str, arguments: Mapping[str, Any]) -> AgentLoopStepOutcome: ...

    def snapshot(self) -> dict[str, Any]:
        """Handle for the verifier (workspace dir, container id, call log)."""

    def close(self) -> None: ...


def step_outcome_to_function_output(outcome: AgentLoopStepOutcome, *, max_chars: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"success": bool(outcome.ok)}
    if outcome.output is not None:
        rendered = outcome.output
        if isinstance(rendered, str):
            rendered = truncate_text(rendered, max_chars)
        else:
            try:
                rendered_text = json.dumps(rendered, ensure_ascii=False)
            except (TypeError, ValueError):
                rendered_text = str(rendered)
            if len(rendered_text) > max_chars:
                rendered = truncate_text(rendered_text, max_chars)
        payload["output"] = rendered
    if outcome.error:
        payload["error"] = truncate_text(str(outcome.error), max_chars)
    for key, value in outcome.details.items():
        payload.setdefault(key, value)
    return payload


def shell_call_to_command(name: str, arguments: Mapping[str, Any]) -> str:
    """Convert a model JSON call into the shell command executed in the sandbox."""

    if name == "bash":
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise ValueError("bash call missing command")
        return command
    if name == "read_file":
        path = str(arguments.get("path") or "").strip()
        if not path:
            raise ValueError("read_file call missing path")
        return f"cat -- {_shell_quote(path)}"
    raise ValueError(f"unsupported shell tool: {name}")


def mcp_call_to_worker(name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert a model JSON call into the MCP worker (full_tool_name, arguments) pair."""

    full_name = str(name or "").strip()
    if not full_name:
        raise ValueError("mcp call missing tool name")
    return full_name, {key: value for key, value in dict(arguments or {}).items()}


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


class ManifestReplayExecutor:
    """Replays tool outputs recorded in the dataset row (offline execution)."""

    def __init__(self, *, recorded_tool_outputs: Sequence[Mapping[str, Any]], match: str = "by_name") -> None:
        self._rows = [dict(row) for row in recorded_tool_outputs]
        self._consumed = [False] * len(self._rows)
        self._match = match
        self._calls: list[dict[str, Any]] = []

    def open(self) -> tuple[dict[str, Any], ...]:
        return ()

    def execute(self, name: str, arguments: Mapping[str, Any]) -> AgentLoopStepOutcome:
        self._calls.append({"name": name, "arguments": dict(arguments)})
        index = self._find_row(name, arguments)
        if index is None:
            return AgentLoopStepOutcome(ok=False, error=f"no recorded output for tool {name!r}")
        self._consumed[index] = True
        row = self._rows[index]
        error = str(row.get("error") or "")
        return AgentLoopStepOutcome(
            ok=not error and bool(row.get("success", True)),
            output=row.get("output"),
            error=error or None,
        )

    def _find_row(self, name: str, arguments: Mapping[str, Any]) -> int | None:
        wanted_args = _canonical_json(arguments)
        fallback: int | None = None
        for index, row in enumerate(self._rows):
            if self._consumed[index] or str(row.get("name") or "") != name:
                continue
            if self._match == "by_step":
                return index
            recorded_args = row.get("arguments")
            if recorded_args is None:
                fallback = fallback if fallback is not None else index
                continue
            if _canonical_json(recorded_args) == wanted_args:
                return index
            fallback = fallback if fallback is not None else index
        return fallback

    def snapshot(self) -> dict[str, Any]:
        return {"calls": list(self._calls), "unconsumed_outputs": self._consumed.count(False)}

    def close(self) -> None:
        return None


class ShellSandboxExecutor:
    """Runs shell tools in a workspace directory or a docker container."""

    def __init__(
        self,
        *,
        backend: str = "subprocess",
        image: str | None = None,
        dockerfile_context: str | None = None,
        dockerfile_path: str | None = None,
        docker_compose_file: str | None = None,
        docker_copy_paths: Sequence[Mapping[str, str]] = (),
        workspace_archive: str | None = None,
        setup_commands: Sequence[str] = (),
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        workspace_root: str | None = None,
        container_workdir: str = "/app",
    ) -> None:
        if backend not in {"subprocess", "docker"}:
            raise ValueError(f"unsupported shell sandbox backend: {backend!r}")
        self._backend = backend
        self._image = image
        self._dockerfile_context = dockerfile_context
        self._dockerfile_path = dockerfile_path
        self._docker_compose_file = docker_compose_file
        self._docker_copy_paths = tuple(dict(item) for item in docker_copy_paths)
        self._workspace_archive = workspace_archive
        self._setup_commands = tuple(setup_commands)
        self._timeout_s = float(command_timeout_s)
        self._max_output_chars = int(max_output_chars)
        self._workspace_root = workspace_root
        self._container_workdir = container_workdir
        self._workspace: Path | None = None
        self._container_id: str | None = None
        self._compose_project: str | None = None
        self._compose_env: dict[str, str] | None = None
        self._compose_logs_root: Path | None = None
        self._calls: list[dict[str, Any]] = []

    def open(self) -> tuple[dict[str, Any], ...]:
        if self._backend == "subprocess":
            root = Path(self._workspace_root).expanduser() if self._workspace_root else None
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
            self._workspace = Path(tempfile.mkdtemp(prefix="agent-loop-", dir=str(root) if root else None))
            if self._workspace_archive:
                shutil.unpack_archive(str(Path(self._workspace_archive).expanduser()), str(self._workspace))
        else:
            if not self._image:
                raise ValueError("docker shell sandbox requires an image")
            name = f"agent-loop-{uuid.uuid4().hex[:12]}"
            if self._docker_compose_file:
                self._start_docker_compose(name)
            else:
                if self._dockerfile_context:
                    self._ensure_docker_image()
                elif str(
                    os.environ.get(
                        "RWKV_AGENT_LOOP_DOCKER_PULL_BEFORE_RUN", ""
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    self._ensure_registry_image()
                docker_run = [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    name,
                    *(
                        ["--network", "none"]
                        if str(
                            os.environ.get(
                                "RWKV_AGENT_LOOP_DOCKER_NETWORK_NONE", ""
                            )
                        ).strip().lower()
                        in {"1", "true", "yes", "on"}
                        else []
                    ),
                    *_docker_proxy_env_args(),
                    self._image,
                    "sleep",
                    "infinity",
                ]
                attempts = _positive_env_int(
                    "RWKV_AGENT_LOOP_DOCKER_RUN_RETRIES",
                    3,
                )
                last_output = ""
                for attempt in range(1, attempts + 1):
                    proc = subprocess.run(
                        docker_run,
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if proc.returncode == 0:
                        break
                    last_output = (proc.stdout + "\n" + proc.stderr).strip()
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if attempt < attempts:
                        time.sleep(min(30.0, 5.0 * attempt))
                else:
                    raise RuntimeError(
                        "docker run failed after "
                        f"{attempts} attempt(s): {last_output[-4000:]}"
                    )
                self._container_id = name
            self._container_id = name
            for item in self._docker_copy_paths:
                self._copy_path_to_container(str(item.get("src") or ""), str(item.get("dst") or ""))
        for command in self._setup_commands:
            outcome = self._run_command(command)
            if not outcome.ok:
                raise RuntimeError(f"shell sandbox setup command failed: {command}: {outcome.error}")
        return _SHELL_TOOLS

    def _copy_path_to_container(self, src: str, dst: str) -> None:
        if not src or not dst:
            raise ValueError("docker_copy_paths entries require src and dst")
        source = Path(src).expanduser()
        if not source.exists():
            raise ValueError(f"docker_copy_paths source does not exist: {source}")
        assert self._container_id is not None
        lock_path = self._docker_lock_path("container-copy-global")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            attempts = _positive_env_int(
                "RWKV_AGENT_LOOP_DOCKER_CP_RETRIES",
                5,
            )
            last_output = ""
            for attempt in range(1, attempts + 1):
                proc = subprocess.run(
                    [
                        "docker",
                        "cp",
                        str(source),
                        f"{self._container_id}:{dst}",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                if proc.returncode == 0:
                    return
                last_output = (proc.stdout + "\n" + proc.stderr).strip()
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        self._container_id,
                        "rm",
                        "-rf",
                        "--",
                        dst,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                if attempt < attempts:
                    time.sleep(min(30.0, 5.0 * attempt))
        raise RuntimeError(
            "docker cp failed after "
            f"{attempts} attempt(s) for {source} -> {dst}: "
            f"{last_output[-2000:]}"
        )

    def _ensure_docker_image(self) -> None:
        assert self._image is not None
        context = Path(str(self._dockerfile_context)).expanduser()
        dockerfile = Path(str(self._dockerfile_path)).expanduser() if self._dockerfile_path else context / "Dockerfile"
        if not dockerfile.is_file():
            raise ValueError(f"dockerfile_context missing Dockerfile: {dockerfile}")
        lock_path = self._docker_lock_path(self._image)
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            self._ensure_docker_image_unlocked(context, dockerfile)

    def _ensure_docker_image_unlocked(self, context: Path, dockerfile: Path) -> None:
        assert self._image is not None
        inspect = subprocess.run(
            ["docker", "image", "inspect", self._image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if inspect.returncode == 0:
            return
        self._run_with_retries(
            ["docker", "build", "-f", str(dockerfile), "-t", self._image, *_docker_proxy_build_args(), str(context)],
            timeout=max(300.0, self._timeout_s),
            action=f"docker build for {self._image} from {context}",
        )

    def _ensure_registry_image(self) -> None:
        assert self._image is not None
        if self._docker_image_exists(self._image):
            return
        lock_path = self._docker_lock_path("registry-pull-global")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if self._docker_image_exists(self._image):
                return
            attempts = _positive_env_int(
                "RWKV_AGENT_LOOP_DOCKER_PULL_RETRIES",
                8,
            )
            pull_timeout_s = float(
                _positive_env_int(
                    "RWKV_AGENT_LOOP_DOCKER_PULL_TIMEOUT_S",
                    1800,
                )
            )
            load_timeout_s = float(
                _positive_env_int(
                    "RWKV_AGENT_LOOP_DOCKER_LOAD_TIMEOUT_S",
                    1800,
                )
            )
            mirror_prefix = str(
                os.environ.get("RWKV_AGENT_LOOP_DOCKER_MIRROR_PREFIX", "")
            ).strip().rstrip("/")
            source_image = (
                f"{mirror_prefix}/{self._image}"
                if mirror_prefix
                else self._image
            )
            crane_path = str(
                os.environ.get("RWKV_AGENT_LOOP_CRANE_PATH", "")
            ).strip()
            last_output = ""
            for attempt in range(1, attempts + 1):
                if crane_path:
                    tar_path = (
                        Path(tempfile.gettempdir())
                        / f"rwkv-agent-image-{uuid.uuid4().hex}.tar"
                    )
                    crane_env = dict(os.environ)
                    for proxy_name in _PROXY_ENV_NAMES:
                        crane_env.pop(proxy_name, None)
                    crane_proxy = str(
                        os.environ.get(
                            "RWKV_AGENT_LOOP_CRANE_PROXY", ""
                        )
                    ).strip()
                    if crane_proxy:
                        crane_env["HTTP_PROXY"] = crane_proxy
                        crane_env["HTTPS_PROXY"] = crane_proxy
                    try:
                        proc = subprocess.run(
                            [
                                crane_path,
                                "pull",
                                "--platform",
                                "linux/amd64",
                                source_image,
                                str(tar_path),
                            ],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=pull_timeout_s,
                            check=False,
                            env=crane_env,
                        )
                        if proc.returncode == 0:
                            proc = subprocess.run(
                                ["docker", "load", "--input", str(tar_path)],
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=load_timeout_s,
                                check=False,
                            )
                    except subprocess.TimeoutExpired as exc:
                        last_output = (
                            f"attempt {attempt} timed out after "
                            f"{float(exc.timeout):.0f}s for {source_image}"
                        )
                        if attempt < attempts:
                            time.sleep(min(60.0, 10.0 * attempt))
                        continue
                    finally:
                        tar_path.unlink(missing_ok=True)
                else:
                    try:
                        proc = subprocess.run(
                            ["docker", "pull", source_image],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=pull_timeout_s,
                            check=False,
                        )
                    except subprocess.TimeoutExpired as exc:
                        last_output = (
                            f"attempt {attempt} timed out after "
                            f"{float(exc.timeout):.0f}s for {source_image}"
                        )
                        if attempt < attempts:
                            time.sleep(min(60.0, 10.0 * attempt))
                        continue
                if proc.returncode == 0:
                    if source_image != self._image:
                        tag_proc = subprocess.run(
                            ["docker", "tag", source_image, self._image],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            check=False,
                        )
                        if tag_proc.returncode != 0:
                            last_output = (
                                tag_proc.stdout + "\n" + tag_proc.stderr
                            ).strip()
                        else:
                            subprocess.run(
                                ["docker", "image", "rm", source_image],
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                check=False,
                            )
                            return
                    else:
                        return
                last_output = (proc.stdout + "\n" + proc.stderr).strip()
                if attempt < attempts:
                    time.sleep(min(60.0, 10.0 * attempt))
            raise RuntimeError(
                "docker pull failed after "
                f"{attempts} attempt(s) for {source_image}: {last_output[-4000:]}"
            )

    def _start_docker_compose(self, container_name: str) -> None:
        assert self._image is not None
        compose_file = Path(str(self._docker_compose_file)).expanduser()
        if not compose_file.is_file():
            raise ValueError(f"docker compose file not found: {compose_file}")
        logs_root = Path(tempfile.mkdtemp(prefix="agent-loop-tbench-"))
        sessions_logs = logs_root / "sessions"
        agent_logs = logs_root / "agent"
        sessions_logs.mkdir(parents=True, exist_ok=True)
        agent_logs.mkdir(parents=True, exist_ok=True)

        prefix = _safe_docker_token(self._image)
        env = dict(os.environ)
        env.update(
            {
                "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": self._image,
                "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": container_name,
                "T_BENCH_TASK_DOCKER_NAME_PREFIX": prefix,
                "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
                "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
                "T_BENCH_TEST_DIR": "/tests",
                "T_BENCH_TASK_LOGS_PATH": str(sessions_logs),
                "T_BENCH_TASK_AGENT_LOGS_PATH": str(agent_logs),
            }
        )
        self._compose_project = container_name
        self._compose_env = env
        self._compose_logs_root = logs_root

        base = ["docker", "compose", "-p", container_name, "-f", str(compose_file)]
        lock_path = self._docker_lock_path(f"{self._image}\0{compose_file.resolve()}")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not self._docker_image_exists(self._image):
                self._run_with_retries(
                    [*base, "build"],
                    timeout=max(1800.0, self._timeout_s),
                    action=f"docker compose build for {compose_file}",
                    env=env,
                )
        self._run_checked(
            [*base, "up", "-d", "--no-build"],
            timeout=max(300.0, self._timeout_s),
            action=f"docker compose up for {compose_file}",
            env=env,
        )
        self._run_checked(
            ["docker", "container", "inspect", container_name],
            timeout=60.0,
            action=f"docker inspect client container {container_name}",
            env=env,
        )

    def _docker_lock_path(self, key: str) -> Path:
        lock_dir = (
            Path(os.environ.get("RWKV_AGENT_LOOP_DOCKER_LOCK_DIR") or tempfile.gettempdir())
            / "rwkv-agent-loop-docker-locks"
        )
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_hash = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]
        return lock_dir / f"{lock_hash}.lock"

    def _docker_image_exists(self, image: str) -> bool:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60.0,
        )
        return proc.returncode == 0

    def _run_with_retries(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        action: str,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        attempts = _positive_env_int("RWKV_AGENT_LOOP_DOCKER_BUILD_RETRIES", 3)
        last_output = ""
        for attempt in range(1, attempts + 1):
            proc = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=dict(env) if env is not None else None,
            )
            if proc.returncode == 0:
                return proc
            last_output = (proc.stdout + "\n" + proc.stderr).strip()
            if attempt < attempts:
                time.sleep(min(30.0, 5.0 * attempt))
        raise RuntimeError(f"{action} failed after {attempts} attempt(s): {last_output[-4000:]}")

    def _run_checked(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        action: str,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=dict(env) if env is not None else None,
        )
        if proc.returncode != 0:
            output = (proc.stdout + "\n" + proc.stderr).strip()
            raise RuntimeError(f"{action} failed: {output[-4000:]}")
        return proc

    def execute(self, name: str, arguments: Mapping[str, Any]) -> AgentLoopStepOutcome:
        self._calls.append({"name": name, "arguments": dict(arguments)})
        try:
            if name == "write_file":
                return self._write_file(str(arguments.get("path") or ""), str(arguments.get("content") or ""))
            command = shell_call_to_command(name, arguments)
        except ValueError as exc:
            return AgentLoopStepOutcome(ok=False, error=str(exc))
        return self._run_command(command)

    def _write_file(self, path: str, content: str) -> AgentLoopStepOutcome:
        if not path:
            return AgentLoopStepOutcome(ok=False, error="write_file call missing path")
        if self._backend == "subprocess":
            assert self._workspace is not None
            target = (self._workspace / path).resolve()
            if not str(target).startswith(str(self._workspace.resolve())):
                return AgentLoopStepOutcome(ok=False, error=f"write_file path escapes workspace: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return AgentLoopStepOutcome(ok=True, output=f"wrote {len(content)} chars to {path}")
        command = f"mkdir -p -- {_shell_quote(str(Path(path).parent) or '.')} && cat > {_shell_quote(path)}"
        return self._run_command(command, stdin=content)

    def _run_command(self, command: str, *, stdin: str | None = None) -> AgentLoopStepOutcome:
        if self._backend == "subprocess":
            assert self._workspace is not None
            argv: list[str] = ["bash", "-lc", command]
            return self._run_subprocess_command(argv, cwd=str(self._workspace), stdin=stdin)
        else:
            assert self._container_id is not None
            argv = [
                "docker",
                "exec",
                "-i",
                "-w",
                self._container_workdir,
                *_docker_sandbox_env_args(),
                self._container_id,
                "bash",
                "-lc",
                command,
            ]
            cwd = None
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            return AgentLoopStepOutcome(ok=False, error=f"command timed out after {self._timeout_s:.0f}s")
        output = proc.stdout
        if proc.stderr:
            output = f"{output}\n{proc.stderr}" if output else proc.stderr
        return AgentLoopStepOutcome(
            ok=proc.returncode == 0,
            output=truncate_text(output, self._max_output_chars),
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            details={"exit_code": proc.returncode},
        )

    def _run_subprocess_command(self, argv: Sequence[str], *, cwd: str, stdin: str | None) -> AgentLoopStepOutcome:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            env=_sandbox_command_env(),
        )
        try:
            stdout, stderr = proc.communicate(input=stdin, timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            output = stdout
            if stderr:
                output = f"{output}\n{stderr}" if output else stderr
            return AgentLoopStepOutcome(
                ok=False,
                output=truncate_text(output, self._max_output_chars) if output else None,
                error=f"command timed out after {self._timeout_s:.0f}s",
            )
        output = stdout
        if stderr:
            output = f"{output}\n{stderr}" if output else stderr
        return AgentLoopStepOutcome(
            ok=proc.returncode == 0,
            output=truncate_text(output, self._max_output_chars),
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            details={"exit_code": proc.returncode},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "workspace": str(self._workspace) if self._workspace else None,
            "container_id": self._container_id,
            "calls": list(self._calls),
        }

    def close(self) -> None:
        if self._compose_project and self._docker_compose_file:
            compose_file = Path(str(self._docker_compose_file)).expanduser()
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-p",
                    self._compose_project,
                    "-f",
                    str(compose_file),
                    "down",
                    "--volumes",
                    "--remove-orphans",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(self._compose_env) if self._compose_env is not None else None,
                check=False,
            )
            self._compose_project = None
            self._container_id = None
            if self._compose_logs_root is not None:
                shutil.rmtree(self._compose_logs_root, ignore_errors=True)
                self._compose_logs_root = None
            return None
        if self._container_id is not None:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self._container_id = None


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return max(1, int(default))


class WebSearchExecutor:
    """Live web-search tools for browsing-style agent benchmarks.

    The search backend is a generic JSON POST endpoint configured in .env
    (RWKV_WEB_SEARCH_API_URL / RWKV_WEB_SEARCH_API_KEY, Serper-style
    X-API-KEY header by default), so providers can be swapped later.
    """

    def __init__(self, *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS, request_timeout_s: float = 30.0) -> None:
        self._max_output_chars = int(max_output_chars)
        self._timeout_s = float(request_timeout_s)
        self._calls: list[dict[str, Any]] = []

    @staticmethod
    def config_error() -> str | None:
        if not os.environ.get(WEB_SEARCH_API_URL_ENV) or not os.environ.get(WEB_SEARCH_API_KEY_ENV):
            return (
                f"web_search executor requires {WEB_SEARCH_API_URL_ENV} and {WEB_SEARCH_API_KEY_ENV} in .env"
            )
        return None

    @staticmethod
    def health_error(*, timeout_s: float = 5.0) -> str | None:
        error = WebSearchExecutor.config_error()
        if error:
            return error
        url = os.environ[WEB_SEARCH_API_URL_ENV]
        health_url = WebSearchExecutor._health_url(url)
        if health_url:
            try:
                with urllib.request.urlopen(health_url, timeout=max(1.0, float(timeout_s))) as response:
                    response.read(1)
                    return None
            except Exception:
                pass
        request = urllib.request.Request(
            url,
            data=json.dumps({"q": "rwkv skills web search preflight"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                os.environ.get(WEB_SEARCH_API_KEY_HEADER_ENV) or "X-API-KEY": os.environ[WEB_SEARCH_API_KEY_ENV],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
                response.read(1)
        except Exception as exc:  # noqa: BLE001 - convert network failures into preflight errors.
            return f"web_search executor cannot reach {WEB_SEARCH_API_URL_ENV}={url!r}: {type(exc).__name__}: {exc}"
        return None

    @staticmethod
    def _health_url(search_url: str) -> str | None:
        parsed = urllib.parse.urlparse(search_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))

    def open(self) -> tuple[dict[str, Any], ...]:
        error = self.config_error()
        if error:
            raise ValueError(error)
        return _WEB_SEARCH_TOOLS

    def execute(self, name: str, arguments: Mapping[str, Any]) -> AgentLoopStepOutcome:
        self._calls.append({"name": name, "arguments": dict(arguments)})
        try:
            if name == "web_search":
                query = str(arguments.get("query") or "").strip()
                if not query:
                    return AgentLoopStepOutcome(ok=False, error="web_search call missing query")
                return AgentLoopStepOutcome(ok=True, output=self._search(query))
            if name == "fetch_url":
                url = str(arguments.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    return AgentLoopStepOutcome(ok=False, error=f"fetch_url requires an http(s) url: {url!r}")
                return AgentLoopStepOutcome(ok=True, output=self._fetch(url))
            return AgentLoopStepOutcome(ok=False, error=f"unsupported web tool: {name}")
        except Exception as exc:  # noqa: BLE001 - network errors are step failures
            return AgentLoopStepOutcome(ok=False, error=str(exc))

    def _search(self, query: str) -> str:
        request = urllib.request.Request(
            os.environ[WEB_SEARCH_API_URL_ENV],
            data=json.dumps({"q": query}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                os.environ.get(WEB_SEARCH_API_KEY_HEADER_ENV) or "X-API-KEY": os.environ[WEB_SEARCH_API_KEY_ENV],
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return truncate_text(_decode_http_body(response.read(), response.headers), self._max_output_chars)

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json,*/*;q=0.8",
                "Accept-Encoding": "identity, gzip, deflate",
                "User-Agent": "rwkv-skills-agent-loop/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return truncate_text(_decode_http_body(response.read(), response.headers), self._max_output_chars)

    def snapshot(self) -> dict[str, Any]:
        return {"calls": list(self._calls)}

    def close(self) -> None:
        return None


class McpWorkerExecutor:
    """Bridges agent-loop tool calls onto the MCP-Bench subprocess worker."""

    def __init__(self, *, runtime_root: str, worker_script: str | None, servers: Sequence[str]) -> None:
        from types import SimpleNamespace

        from src.eval.scheduler.config import REPO_ROOT
        from src.eval.tasks.function_calling.mcp_bench import McpBenchWorkerClient

        script = Path(worker_script).expanduser() if worker_script else REPO_ROOT / "src" / "eval" / "tasks" / "function_calling" / "mcp_bench_worker.py"
        self._client = McpBenchWorkerClient(runtime_root=Path(runtime_root).expanduser(), worker_script=script)
        self._servers = tuple(str(server) for server in servers)
        self._task_shim = SimpleNamespace(servers=self._servers)
        self._calls: list[dict[str, Any]] = []

    def open(self) -> tuple[dict[str, Any], ...]:
        available_tools = self._client.open_task(self._task_shim)  # type: ignore[arg-type]
        tools: list[dict[str, Any]] = []
        for full_name, info in (available_tools or {}).items():
            entry = info if isinstance(info, Mapping) else {}
            tools.append(
                {
                    "name": str(full_name),
                    "description": str(entry.get("description") or ""),
                    "parameters": dict(entry.get("input_schema") or {"type": "object", "properties": {}}),
                }
            )
        return tuple(tools)

    def execute(self, name: str, arguments: Mapping[str, Any]) -> AgentLoopStepOutcome:
        self._calls.append({"name": name, "arguments": dict(arguments)})
        try:
            full_name, args = mcp_call_to_worker(name, arguments)
            response = self._client.call_tool(full_name, args)
        except Exception as exc:  # noqa: BLE001 - worker errors are step failures
            return AgentLoopStepOutcome(ok=False, error=str(exc))
        return AgentLoopStepOutcome(
            ok=bool(response.get("success", False)),
            output=response.get("result"),
            error=str(response.get("error") or "") or None,
        )

    def snapshot(self) -> dict[str, Any]:
        return {"servers": list(self._servers), "calls": list(self._calls)}

    def close(self) -> None:
        try:
            self._client.close_task()
        except Exception:
            pass
        self._client.close()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "AgentLoopExecutor",
    "AgentLoopStepOutcome",
    "DEFAULT_COMMAND_TIMEOUT_S",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "ExecutorSpec",
    "ManifestReplayExecutor",
    "McpWorkerExecutor",
    "ShellSandboxExecutor",
    "WEB_SEARCH_API_KEY_ENV",
    "WEB_SEARCH_API_KEY_HEADER_ENV",
    "WEB_SEARCH_API_URL_ENV",
    "WebSearchExecutor",
    "mcp_call_to_worker",
    "shell_call_to_command",
    "step_outcome_to_function_output",
]
