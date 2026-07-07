from __future__ import annotations

"""Executors for the generic multi-turn agent-loop benchmark channel.

An executor turns the model's JSON tool call into a real environment action
(recorded manifest replay, sandbox shell command, or MCP worker call) and
returns a JSON-serializable outcome that is fed back to the model as
``User: Function output:\\n<json>``.
"""

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.eval.tasks.function_calling.context_budget import truncate_text

DEFAULT_COMMAND_TIMEOUT_S = 60.0
DEFAULT_MAX_OUTPUT_CHARS = 8000

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
        self._workspace_archive = workspace_archive
        self._setup_commands = tuple(setup_commands)
        self._timeout_s = float(command_timeout_s)
        self._max_output_chars = int(max_output_chars)
        self._workspace_root = workspace_root
        self._container_workdir = container_workdir
        self._workspace: Path | None = None
        self._container_id: str | None = None
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
            subprocess.run(
                ["docker", "run", "-d", "--rm", "--name", name, self._image, "sleep", "infinity"],
                check=True,
                capture_output=True,
                text=True,
            )
            self._container_id = name
        for command in self._setup_commands:
            outcome = self._run_command(command)
            if not outcome.ok:
                raise RuntimeError(f"shell sandbox setup command failed: {command}: {outcome.error}")
        return _SHELL_TOOLS

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
            cwd: str | None = str(self._workspace)
        else:
            assert self._container_id is not None
            argv = [
                "docker",
                "exec",
                "-i",
                "-w",
                self._container_workdir,
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "workspace": str(self._workspace) if self._workspace else None,
            "container_id": self._container_id,
            "calls": list(self._calls),
        }

    def close(self) -> None:
        if self._container_id is not None:
            subprocess.run(["docker", "rm", "-f", self._container_id], capture_output=True, text=True, check=False)
            self._container_id = None


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
            return truncate_text(response.read().decode("utf-8", errors="replace"), self._max_output_chars)

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "rwkv-skills-agent-loop/1.0"})
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return truncate_text(response.read().decode("utf-8", errors="replace"), self._max_output_chars)

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

        script = Path(worker_script).expanduser() if worker_script else REPO_ROOT / "src" / "eval" / "function_calling" / "mcp_bench_worker.py"
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
