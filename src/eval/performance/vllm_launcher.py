from __future__ import annotations

import shlex
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

from src.eval.performance.service_client import OpenAIChatServiceClient


DEFAULT_VLLM_COMMAND = ("python", "-m", "vllm.entrypoints.openai.api_server")
DEFAULT_VLLM_MODULE = "vllm.entrypoints.openai.api_server"
_LEGACY_VLLM_ARG_ALIASES = {
    "--disable-log-requests": "--no-enable-log-requests",
}


def parse_shell_args(raw: str | None) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    return tuple(shlex.split(text))


def _normalize_vllm_args(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_LEGACY_VLLM_ARG_ALIASES.get(arg, arg) for arg in args)


def _tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, int(max_lines)):])


def _explain_launch_failure(log_tail: str) -> str:
    text = str(log_tail or "")
    if "No module named 'vllm'" in text:
        return (
            "检测到当前 Python 环境里没有可用的 vLLM。"
            " 请先在同一个虚拟环境里安装 vllm，或通过 --vllm-python / [vllm].python_executable"
            " 指向另一套已安装 vllm 的 Python 环境。"
        )
    if "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation" in text:
        return (
            "检测到 vLLM 与当前 PyTorch/CUDA 二进制不兼容。"
            " 这通常是 vllm 编译时绑定的 torch 版本，与当前环境实际加载的 torch 版本不一致。"
            " 请通过 --vllm-python / [vllm].python_executable 切到一套匹配的 vLLM Python 环境，"
            " 或重新安装与当前 torch/CUDA 匹配的 vllm。"
        )
    if "unrecognized arguments: --disable-log-requests" in text:
        return (
            "检测到 vLLM CLI 参数版本不兼容。"
            " vLLM 0.19.0 使用 --no-enable-log-requests，"
            " 不再接受旧参数 --disable-log-requests。"
        )
    return ""


@dataclass(slots=True)
class VllmLaunchConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""
    python_executable: str | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    module: str = DEFAULT_VLLM_MODULE
    dtype: str | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    trust_remote_code: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    startup_timeout_s: float = 600.0
    poll_interval_s: float = 2.0
    log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_command(self) -> list[str]:
        if self.command:
            command = list(self.command)
        else:
            command = [str(self.python_executable or "python"), "-m", str(self.module)]
        command.extend(
            [
                "--model",
                self.model,
                "--host",
                str(self.host),
                "--port",
                str(self.port),
            ]
        )
        if self.dtype:
            command.extend(["--dtype", str(self.dtype)])
        if self.tensor_parallel_size is not None:
            command.extend(["--tensor-parallel-size", str(int(self.tensor_parallel_size))])
        if self.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(float(self.gpu_memory_utilization))])
        if self.max_model_len is not None:
            command.extend(["--max-model-len", str(int(self.max_model_len))])
        if self.max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(int(self.max_num_seqs))])
        if self.max_num_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(int(self.max_num_batched_tokens))])
        if self.trust_remote_code:
            command.append("--trust-remote-code")
        command.extend(_normalize_vllm_args(self.extra_args))
        return command


@dataclass(slots=True)
class RunningVllmServer:
    config: VllmLaunchConfig
    process: subprocess.Popen[bytes]
    log_handle: IO[bytes]
    log_path: Path

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + max(float(self.config.startup_timeout_s), 1.0)
        client = OpenAIChatServiceClient(
            base_url=self.config.base_url,
            model="",
            api_key=str(self.config.api_key or ""),
            timeout_s=min(15.0, max(float(self.config.poll_interval_s), 1.0) * 2),
        )
        last_error = ""

        while time.monotonic() < deadline:
            exit_code = self.process.poll()
            if exit_code is not None:
                tail = _tail_text(self.log_path)
                explanation = _explain_launch_failure(tail)
                detail = ""
                if explanation:
                    detail += f"\n{explanation}"
                if tail:
                    detail += f"\n{tail}"
                raise RuntimeError(f"vLLM 启动失败，退出码={exit_code}{detail}")
            try:
                if client.list_models():
                    return
            except BaseException as exc:
                last_error = str(exc)
            time.sleep(max(float(self.config.poll_interval_s), 0.2))

        tail = _tail_text(self.log_path)
        detail = f"\n{tail}" if tail else ""
        if last_error:
            detail = f"\nlast error: {last_error}{detail}"
        raise TimeoutError(f"等待 vLLM 就绪超时: {self.config.base_url}{detail}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.log_handle.close()


@contextmanager
def launch_vllm_server(config: VllmLaunchConfig) -> Iterator[RunningVllmServer]:
    log_path = config.log_path or (Path("/tmp") / f"vllm_perf_{int(time.time())}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    command = config.build_command()
    print(f"启动 vLLM: {shlex.join(command)}")
    print(f"vLLM 日志: {log_path}")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    server = RunningVllmServer(
        config=config,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
    )
    try:
        server.wait_until_ready()
        print(f"vLLM 已就绪: {config.base_url}")
        yield server
    finally:
        server.stop()


__all__ = [
    "DEFAULT_VLLM_COMMAND",
    "DEFAULT_VLLM_MODULE",
    "RunningVllmServer",
    "VllmLaunchConfig",
    "_normalize_vllm_args",
    "launch_vllm_server",
    "parse_shell_args",
]
