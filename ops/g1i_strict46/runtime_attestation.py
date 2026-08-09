#!/usr/bin/env python3
"""Fail-closed runtime identity attestation for strict-46 G1i inference.

The OpenAI-compatible ``/models`` response is only a presentation-level
signal: ``--served-model-name`` can name any file.  This verifier starts at
the local listening socket and walks back to the owning process.  It then
binds the process, the actual model argument, the model bytes, the inference
source tree, GPU selection, cgroup/systemd unit, and all output-affecting
launch settings to a root-owned approval artifact.

The production verifier intentionally has no "best effort" mode.  A runtime
that has not been provisioned under the trusted UID, an ambiguous listener,
or a field that cannot be observed is rejected.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ATTESTATION_SCHEMA = "rwkv.g1i-runtime-attestation.v1"
FORWARD_ATTESTATION_SCHEMA = "rwkv.g1i-forward-attestation.v1"
VERIFICATION_SCHEMA = "rwkv.g1i-runtime-verification.v1"
FORWARD_VERIFICATION_SCHEMA = "rwkv.g1i-forward-verification.v1"

G1I_MODEL_RE = re.compile(
    r"^rwkv7-g1i-(?:1\.5|2\.9|7\.2|13\.3)b-\d{8}-ctx(?P<context>\d+)$"
)

SEMANTIC_ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
}
SEMANTIC_ENVIRONMENT_PREFIXES = (
    "VLLM_",
    "RWKV_",
    "CUDA_",
    "PYTORCH_",
    "TORCH_",
)
SECRET_ENVIRONMENT_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
REQUIRED_ENVIRONMENT = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_RWKV7_WKV_MODE": "fp32io16",
    "VLLM_USE_RAPID_SAMPLER": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}

VALUE_OPTIONS = {
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--host": "host",
    "--max-model-len": "max_model_len",
    "--max-num-batched-tokens": "max_num_batched_tokens",
    "--max-num-seqs": "max_num_seqs",
    "--override-generation-config": "override_generation_config",
    "--port": "port",
    "--served-model-name": "served_model_name",
    "--tokenizer-mode": "tokenizer_mode",
    "--tool-call-parser": "tool_call_parser",
}
BOOLEAN_OPTIONS = {
    "--enable-auto-tool-choice": "enable_auto_tool_choice",
    "--trust-request-chat-template": "trust_request_chat_template",
}
REQUIRED_LAUNCH_PARAMETER_KEYS = set(VALUE_OPTIONS.values()) | set(
    BOOLEAN_OPTIONS.values()
)

_TOP_LEVEL_KEYS = {
    "schema",
    "artifact_sha256",
    "host_label",
    "endpoint",
    "model",
    "process",
    "runtime_tree",
}
_FILE_DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
_TREE_DESCRIPTOR_KEYS = {"path", "tree_sha256", "files"}
_TREE_FILE_DESCRIPTOR_KEYS = {"relative_path", "bytes", "sha256"}
_PROCESS_KEYS = {
    "uid",
    "executable",
    "working_directory",
    "argv_redacted",
    "environment",
    "cgroup",
    "systemd_unit",
    "gpu_index",
    "launch_parameters",
}
_FORWARD_TOP_LEVEL_KEYS = {
    "schema",
    "artifact_sha256",
    "host_label",
    "endpoint",
    "destination",
    "process",
    "transport_files",
    "verification_argv_prefix",
}
_FORWARD_DESTINATION_KEYS = {"host_label", "endpoint"}
_FORWARD_PROCESS_KEYS = {
    "uid",
    "executable",
    "working_directory",
    "argv_redacted",
    "cgroup",
    "systemd_unit",
}


class RuntimeAttestationError(RuntimeError):
    """The live inference runtime does not match its trusted attestation."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return canonical_sha256(unsigned)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_file(path: Path) -> bytes:
    if path.is_symlink():
        raise RuntimeAttestationError(f"attested file must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeAttestationError(f"attested file is missing: {path}") from exc
    if not resolved.is_file():
        raise RuntimeAttestationError(
            f"attested path is not a regular file: {resolved}"
        )
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = resolved.stat()
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(current):
        raise RuntimeAttestationError(f"attested file changed while read: {resolved}")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise RuntimeAttestationError(
            f"short read while hashing attested file: {resolved}"
        )
    return payload


def _read_proc_file(path: Path) -> bytes:
    """Read a procfs pseudo-file; its reported size is commonly zero."""

    try:
        with path.open("rb", buffering=0) as handle:
            return handle.read()
    except OSError as exc:
        raise RuntimeAttestationError(
            f"cannot read live process evidence {path}: {exc}"
        ) from exc


def describe_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = _read_stable_file(resolved)
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tree_paths(root: Path) -> list[Path]:
    if root.is_symlink():
        raise RuntimeAttestationError(f"runtime tree must not be a symlink: {root}")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeAttestationError(f"runtime tree is not a directory: {resolved}")
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
        base = Path(directory)
        directory_names[:] = sorted(name for name in directory_names if name != ".git")
        for name in directory_names:
            candidate = base / name
            if candidate.is_symlink():
                raise RuntimeAttestationError(
                    f"runtime tree contains a directory symlink: {candidate}"
                )
        for name in sorted(file_names):
            candidate = base / name
            if candidate.is_symlink():
                raise RuntimeAttestationError(
                    f"runtime tree contains a file symlink: {candidate}"
                )
            if not candidate.is_file():
                raise RuntimeAttestationError(
                    f"runtime tree contains a non-regular entry: {candidate}"
                )
            files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(resolved).as_posix())


def describe_tree(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve(strict=True)
    descriptors: list[dict[str, Any]] = []
    for candidate in _tree_paths(root):
        payload = _read_stable_file(candidate)
        descriptors.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not descriptors:
        raise RuntimeAttestationError(f"runtime tree is empty: {root}")
    return {
        "path": str(root),
        "tree_sha256": canonical_sha256(descriptors),
        "files": descriptors,
    }


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise RuntimeAttestationError(
            f"{label} has unexpected schema keys: missing={sorted(keys - actual)!r} "
            f"extra={sorted(actual - keys)!r}"
        )


def _require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeAttestationError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeAttestationError(
            f"{label} must be a normalized absolute path: {value!r}"
        )
    return path


def _validate_file_descriptor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeAttestationError(f"{label} must be an object")
    _require_exact_keys(value, _FILE_DESCRIPTOR_KEYS, label)
    _require_absolute_path(value["path"], f"{label}.path")
    if not isinstance(value["bytes"], int) or value["bytes"] <= 0:
        raise RuntimeAttestationError(f"{label}.bytes must be a positive integer")
    if not _valid_sha256(value["sha256"]):
        raise RuntimeAttestationError(f"{label}.sha256 is invalid")
    return value


def _validate_tree_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeAttestationError("runtime_tree must be an object")
    _require_exact_keys(value, _TREE_DESCRIPTOR_KEYS, "runtime_tree")
    _require_absolute_path(value["path"], "runtime_tree.path")
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise RuntimeAttestationError("runtime_tree.files must be a non-empty list")
    previous = ""
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise RuntimeAttestationError(
                f"runtime_tree.files[{index}] must be an object"
            )
        _require_exact_keys(
            item, _TREE_FILE_DESCRIPTOR_KEYS, f"runtime_tree.files[{index}]"
        )
        relative = item["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative == previous
            or (previous and relative < previous)
        ):
            raise RuntimeAttestationError(
                "runtime_tree file inventory is not strictly sorted and safe"
            )
        previous = relative
        if not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise RuntimeAttestationError("runtime_tree file byte count is invalid")
        if not _valid_sha256(item["sha256"]):
            raise RuntimeAttestationError("runtime_tree file sha256 is invalid")
    if value["tree_sha256"] != canonical_sha256(files):
        raise RuntimeAttestationError(
            "runtime_tree.tree_sha256 does not bind its file inventory"
        )
    return value


def _endpoint_contract(value: Any) -> dict[str, Any]:
    keys = {"scheme", "host", "port", "api_prefix"}
    if not isinstance(value, dict):
        raise RuntimeAttestationError("endpoint must be an object")
    _require_exact_keys(value, keys, "endpoint")
    if value["scheme"] != "http":
        raise RuntimeAttestationError("runtime endpoint must use local HTTP")
    host = str(value["host"])
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeAttestationError(
            "runtime endpoint host must be a numeric loopback address"
        ) from exc
    if not address.is_loopback:
        raise RuntimeAttestationError("runtime endpoint host must be loopback")
    port = value["port"]
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeAttestationError("runtime endpoint port is invalid")
    prefix = value["api_prefix"]
    if (
        not isinstance(prefix, str)
        or not prefix.startswith("/")
        or prefix.endswith("/")
    ):
        raise RuntimeAttestationError(
            "endpoint.api_prefix must be a normalized absolute URL path"
        )
    return {"scheme": "http", "host": str(address), "port": port, "api_prefix": prefix}


def endpoint_contract_from_url(endpoint_url: str) -> dict[str, Any]:
    parsed = urlsplit(endpoint_url)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise RuntimeAttestationError(
            "endpoint URL must not contain credentials, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeAttestationError("endpoint URL port is invalid") from exc
    if parsed.scheme != "http" or parsed.hostname is None or port is None:
        raise RuntimeAttestationError(
            "endpoint URL must be an explicit local http://host:port URL"
        )
    prefix = parsed.path.rstrip("/") or "/v1"
    return _endpoint_contract(
        {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "port": port,
            "api_prefix": prefix,
        }
    )


def _normalize_cgroup(payload: bytes) -> list[str]:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeAttestationError("process cgroup is not UTF-8") from exc
    normalized = sorted(line.strip() for line in lines if line.strip())
    if not normalized:
        raise RuntimeAttestationError("process cgroup is empty")
    return normalized


def _systemd_unit(cgroup: Sequence[str]) -> str:
    candidates: list[str] = []
    for line in cgroup:
        path = line.rsplit(":", 1)[-1]
        for component in path.split("/"):
            if component.endswith((".service", ".scope")):
                candidates.append(component)
    if not candidates:
        raise RuntimeAttestationError(
            "listener process is not in an observable systemd service/scope"
        )
    return candidates[-1]


def _secret_environment_key(key: str) -> bool:
    """Identify credential-shaped names without misclassifying TOKEN(S) knobs."""

    upper = key.upper()
    return any(
        upper == marker or upper.endswith(f"_{marker}")
        for marker in SECRET_ENVIRONMENT_MARKERS
    )


def _semantic_environment(environment: Mapping[str, str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in environment.items():
        semantic = key in SEMANTIC_ENVIRONMENT_KEYS or key.startswith(
            SEMANTIC_ENVIRONMENT_PREFIXES
        )
        if not semantic:
            continue
        # Secret-bearing semantic variables still affect the live contract and
        # therefore cannot simply disappear.  Bind their value without placing
        # plaintext credentials in the root-owned approval evidence.
        if _secret_environment_key(key):
            selected[key] = f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
        else:
            selected[key] = value
    return dict(sorted(selected.items()))


def _parse_environ(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        if b"=" not in raw:
            raise RuntimeAttestationError(
                "listener process contains a malformed environment entry"
            )
        key_raw, value_raw = raw.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="strict")
        value = value_raw.decode("utf-8", errors="strict")
        if key in result:
            raise RuntimeAttestationError(
                f"listener process contains duplicate environment key {key!r}"
            )
        result[key] = value
    return result


def _parse_cmdline(payload: bytes) -> list[str]:
    try:
        result = [
            part.decode("utf-8", errors="strict")
            for part in payload.split(b"\0")
            if part
        ]
    except UnicodeDecodeError as exc:
        raise RuntimeAttestationError(
            "listener process command line is not UTF-8"
        ) from exc
    if not result:
        raise RuntimeAttestationError("listener process command line is empty")
    return result


def redact_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--api-key":
            if index + 1 >= len(argv):
                raise RuntimeAttestationError("--api-key has no value")
            redacted.extend((value, "<redacted>"))
            index += 2
            continue
        if value.startswith("--api-key="):
            redacted.append("--api-key=<redacted>")
        else:
            redacted.append(value)
        index += 1
    return redacted


def _single_option(argv: Sequence[str], option: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise RuntimeAttestationError(f"{option} has no value")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) != 1:
        raise RuntimeAttestationError(
            f"expected exactly one {option}, observed {values!r}"
        )
    return values[0]


def _single_boolean_option(argv: Sequence[str], option: str) -> bool:
    count = sum(1 for token in argv if token == option)
    if count != 1:
        raise RuntimeAttestationError(
            f"expected exactly one {option}, observed {count}"
        )
    return True


def _model_argument(argv: Sequence[str]) -> str:
    explicit: list[str] = []
    for index, token in enumerate(argv):
        if token == "--model" and index + 1 < len(argv):
            explicit.append(argv[index + 1])
        elif token.startswith("--model="):
            explicit.append(token.split("=", 1)[1])
    served: list[str] = []
    for index, token in enumerate(argv[:-1]):
        if token == "serve":
            served.append(argv[index + 1])
    candidates = explicit or served
    if len(candidates) != 1:
        raise RuntimeAttestationError(
            f"cannot identify one actual vLLM model argument: {candidates!r}"
        )
    return candidates[0]


def launch_parameters(argv: Sequence[str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for option, key in VALUE_OPTIONS.items():
        raw = _single_option(argv, option)
        if key in {"port", "max_model_len", "max_num_batched_tokens", "max_num_seqs"}:
            try:
                parameters[key] = int(raw)
            except ValueError as exc:
                raise RuntimeAttestationError(f"{option} must be an integer") from exc
        elif key == "gpu_memory_utilization":
            try:
                parameters[key] = float(raw)
            except ValueError as exc:
                raise RuntimeAttestationError(f"{option} must be numeric") from exc
        elif key == "override_generation_config":
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeAttestationError(
                    "--override-generation-config is not valid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeAttestationError(
                    "--override-generation-config must be a JSON object"
                )
            parameters[key] = parsed
        else:
            parameters[key] = raw
    for option, key in BOOLEAN_OPTIONS.items():
        parameters[key] = _single_boolean_option(argv, option)
    return parameters


def _validate_launch_policy(
    *, parameters: Mapping[str, Any], endpoint: Mapping[str, Any], model_name: str
) -> None:
    if set(parameters) != REQUIRED_LAUNCH_PARAMETER_KEYS:
        raise RuntimeAttestationError(
            "launch_parameters does not contain the exact critical option set"
        )
    if parameters["host"] != endpoint["host"] or parameters["port"] != endpoint["port"]:
        raise RuntimeAttestationError(
            "launch host/port does not match the attested listener endpoint"
        )
    if parameters["served_model_name"] != model_name:
        raise RuntimeAttestationError(
            "--served-model-name does not match the display contract"
        )
    if (
        parameters["tokenizer_mode"] != "rwkv"
        or parameters["tool_call_parser"] != "rwkv"
    ):
        raise RuntimeAttestationError(
            "G1i requires RWKV tokenizer and tool-call parser"
        )
    match = G1I_MODEL_RE.fullmatch(model_name)
    if match is None:
        raise RuntimeAttestationError(
            f"model is not a strict G1i identity: {model_name!r}"
        )
    if parameters["max_model_len"] != int(match.group("context")):
        raise RuntimeAttestationError(
            "--max-model-len does not match the model ctx identity"
        )
    if parameters["max_num_seqs"] <= 0:
        raise RuntimeAttestationError("--max-num-seqs must be positive")
    if parameters["max_num_batched_tokens"] < parameters["max_model_len"]:
        raise RuntimeAttestationError(
            "--max-num-batched-tokens is below max model length"
        )
    if not 0 < parameters["gpu_memory_utilization"] <= 1:
        raise RuntimeAttestationError("--gpu-memory-utilization is outside (0, 1]")
    generation = parameters["override_generation_config"]
    if generation.get("temperature") != 1e-5:
        raise RuntimeAttestationError(
            "G1i default generation temperature must be exactly 1e-5"
        )


def _validate_artifact(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeAttestationError("runtime attestation must be a JSON object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "runtime attestation")
    if payload["schema"] != ATTESTATION_SCHEMA:
        raise RuntimeAttestationError(
            f"unexpected runtime attestation schema: {payload['schema']!r}"
        )
    if not _valid_sha256(payload["artifact_sha256"]):
        raise RuntimeAttestationError("runtime attestation artifact_sha256 is invalid")
    if payload["artifact_sha256"] != artifact_sha256(payload):
        raise RuntimeAttestationError("runtime attestation self-digest mismatch")
    if payload["host_label"] not in {"157", "8222"}:
        raise RuntimeAttestationError("host_label must be exactly '157' or '8222'")
    endpoint = _endpoint_contract(payload["endpoint"])
    model = payload["model"]
    if not isinstance(model, dict) or set(model) != {"name", "weight"}:
        raise RuntimeAttestationError("model must contain exactly name and weight")
    if G1I_MODEL_RE.fullmatch(str(model["name"])) is None:
        raise RuntimeAttestationError("model.name is not a strict G1i identity")
    _validate_file_descriptor(model["weight"], "model.weight")
    process = payload["process"]
    if not isinstance(process, dict):
        raise RuntimeAttestationError("process must be an object")
    _require_exact_keys(process, _PROCESS_KEYS, "process")
    if not isinstance(process["uid"], int) or process["uid"] < 0:
        raise RuntimeAttestationError("process.uid is invalid")
    _validate_file_descriptor(process["executable"], "process.executable")
    _require_absolute_path(process["working_directory"], "process.working_directory")
    if not isinstance(process["argv_redacted"], list) or not all(
        isinstance(item, str) for item in process["argv_redacted"]
    ):
        raise RuntimeAttestationError("process.argv_redacted must be a string list")
    if not isinstance(process["environment"], dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in process["environment"].items()
    ):
        raise RuntimeAttestationError("process.environment must be a string map")
    for key, expected in REQUIRED_ENVIRONMENT.items():
        if process["environment"].get(key) != expected:
            raise RuntimeAttestationError(
                f"process.environment must pin {key}={expected!r}"
            )
    if not isinstance(process["cgroup"], list) or not all(
        isinstance(item, str) and item for item in process["cgroup"]
    ):
        raise RuntimeAttestationError("process.cgroup must be a non-empty string list")
    if process["cgroup"] != sorted(process["cgroup"]):
        raise RuntimeAttestationError("process.cgroup must be sorted")
    if process["systemd_unit"] != _systemd_unit(process["cgroup"]):
        raise RuntimeAttestationError(
            "process.systemd_unit does not match process.cgroup"
        )
    gpu_index = process["gpu_index"]
    if not isinstance(gpu_index, int) or gpu_index < 0:
        raise RuntimeAttestationError("process.gpu_index is invalid")
    if process["environment"].get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeAttestationError(
            "process.gpu_index does not match CUDA_VISIBLE_DEVICES"
        )
    parameters = process["launch_parameters"]
    if not isinstance(parameters, dict):
        raise RuntimeAttestationError("process.launch_parameters must be an object")
    _validate_launch_policy(
        parameters=parameters, endpoint=endpoint, model_name=model["name"]
    )
    if launch_parameters(process["argv_redacted"]) != parameters:
        raise RuntimeAttestationError("launch_parameters does not bind argv_redacted")
    expected_model_argument = _model_argument(process["argv_redacted"])
    _require_absolute_path(expected_model_argument, "attested vLLM model argument")
    if Path(expected_model_argument) != Path(model["weight"]["path"]):
        raise RuntimeAttestationError(
            "attested argv does not launch the attested model weight path"
        )
    if payload["host_label"] == "8222" and (
        gpu_index == 3 or endpoint["port"] == 18073
    ):
        raise RuntimeAttestationError(
            "8222 GPU3 and port 18073 are reserved and forbidden"
        )
    runtime_tree = _validate_tree_descriptor(payload["runtime_tree"])
    if process["environment"].get("PYTHONPATH") != runtime_tree["path"]:
        raise RuntimeAttestationError(
            "PYTHONPATH must contain exactly the attested inference runtime tree"
        )
    return payload


def _ssh_options(argv: Sequence[str]) -> set[str]:
    values: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-o":
            if index + 1 >= len(argv):
                raise RuntimeAttestationError("ssh -o has no value")
            values.add(argv[index + 1])
            index += 2
            continue
        if token.startswith("-o") and len(token) > 2:
            values.add(token[2:])
        index += 1
    return values


def _ssh_option_map(argv: Sequence[str]) -> dict[str, str]:
    """Return an unambiguous, case-insensitive map of explicit SSH options.

    OpenSSH accepts repeated ``-o`` values and configuration option names are
    case-insensitive.  Accepting both a safe and unsafe spelling would make a
    simple set-membership check insufficient, so trusted forward commands must
    contain exactly one explicit value for every option they rely on.
    """

    options: dict[str, str] = {}
    for raw in _ssh_options(argv):
        name, separator, value = raw.partition("=")
        if not separator or not name or not value:
            raise RuntimeAttestationError(
                f"ssh option must use an explicit name=value form: {raw!r}"
            )
        key = name.casefold()
        if key in options:
            raise RuntimeAttestationError(
                f"ssh option is duplicated and ambiguous: {name}"
            )
        options[key] = value
    return options


def _forward_endpoints(argv: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _single_option(argv, "-L")
    parts = raw.split(":")
    if len(parts) != 4:
        raise RuntimeAttestationError(
            "ssh forward must be bind_host:bind_port:destination_host:destination_port"
        )
    bind_host, bind_port_raw, destination_host, destination_port_raw = parts
    try:
        bind_port = int(bind_port_raw)
        destination_port = int(destination_port_raw)
    except ValueError as exc:
        raise RuntimeAttestationError("ssh forward ports must be integers") from exc
    return (
        _endpoint_contract(
            {
                "scheme": "http",
                "host": bind_host,
                "port": bind_port,
                "api_prefix": "/v1",
            }
        ),
        _endpoint_contract(
            {
                "scheme": "http",
                "host": destination_host,
                "port": destination_port,
                "api_prefix": "/v1",
            }
        ),
    )


def _validate_forward_artifact(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeAttestationError("forward attestation must be a JSON object")
    _require_exact_keys(payload, _FORWARD_TOP_LEVEL_KEYS, "forward attestation")
    if payload["schema"] != FORWARD_ATTESTATION_SCHEMA:
        raise RuntimeAttestationError("forward attestation schema mismatch")
    if not _valid_sha256(payload["artifact_sha256"]):
        raise RuntimeAttestationError("forward attestation artifact_sha256 is invalid")
    if payload["artifact_sha256"] != artifact_sha256(payload):
        raise RuntimeAttestationError("forward attestation self-digest mismatch")
    if payload["host_label"] != "157":
        raise RuntimeAttestationError("forward listener must be attested on host 157")
    endpoint = _endpoint_contract(payload["endpoint"])
    destination = payload["destination"]
    if not isinstance(destination, dict):
        raise RuntimeAttestationError("forward destination must be an object")
    _require_exact_keys(destination, _FORWARD_DESTINATION_KEYS, "forward destination")
    if destination["host_label"] != "8222":
        raise RuntimeAttestationError("forward destination must be host 8222")
    destination_endpoint = _endpoint_contract(destination["endpoint"])
    if endpoint["port"] == 18073 or destination_endpoint["port"] == 18073:
        raise RuntimeAttestationError("8222 port 18073 is reserved and forbidden")

    process = payload["process"]
    if not isinstance(process, dict):
        raise RuntimeAttestationError("forward process must be an object")
    _require_exact_keys(process, _FORWARD_PROCESS_KEYS, "forward process")
    if not isinstance(process["uid"], int) or process["uid"] < 0:
        raise RuntimeAttestationError("forward process uid is invalid")
    executable = _validate_file_descriptor(
        process["executable"], "forward process executable"
    )
    _require_absolute_path(process["working_directory"], "forward working directory")
    argv = process["argv_redacted"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RuntimeAttestationError("forward argv_redacted must be a string list")
    if not argv or argv[0] != executable["path"] or argv.count("-N") != 1:
        raise RuntimeAttestationError("forward process must be an exact ssh -N command")
    observed_endpoint, observed_destination = _forward_endpoints(argv)
    if observed_endpoint != endpoint or observed_destination != destination_endpoint:
        raise RuntimeAttestationError("ssh -L mapping does not match forward endpoints")
    if not isinstance(process["cgroup"], list) or not all(
        isinstance(item, str) and item for item in process["cgroup"]
    ):
        raise RuntimeAttestationError("forward process cgroup is invalid")
    if process["cgroup"] != sorted(process["cgroup"]):
        raise RuntimeAttestationError("forward process cgroup must be sorted")
    if process["systemd_unit"] != _systemd_unit(process["cgroup"]):
        raise RuntimeAttestationError("forward systemd unit does not match cgroup")

    transport_files = payload["transport_files"]
    if not isinstance(transport_files, list) or not transport_files:
        raise RuntimeAttestationError("forward transport_files must not be empty")
    validated_files = [
        _validate_file_descriptor(item, f"forward transport_files[{index}]")
        for index, item in enumerate(transport_files)
    ]
    transport_paths = [str(item["path"]) for item in validated_files]
    if transport_paths != sorted(set(transport_paths)):
        raise RuntimeAttestationError("forward transport_files must be unique and sorted")

    verification_argv = payload["verification_argv_prefix"]
    if not isinstance(verification_argv, list) or not all(
        isinstance(item, str) and item for item in verification_argv
    ):
        raise RuntimeAttestationError(
            "forward verification_argv_prefix must be a non-empty string list"
        )
    if not verification_argv or verification_argv[0] != executable["path"]:
        raise RuntimeAttestationError("forward verifier must use the attested ssh binary")
    process_config_path = _single_option(argv, "-F")
    verifier_config_path = _single_option(verification_argv, "-F")
    process_identity_path = _single_option(argv, "-i")
    verifier_identity_path = _single_option(verification_argv, "-i")
    if process_config_path != verifier_config_path:
        raise RuntimeAttestationError(
            "forward process and verifier ssh config paths disagree"
        )
    if process_identity_path != verifier_identity_path:
        raise RuntimeAttestationError(
            "forward process and verifier identity paths disagree"
        )
    if process_config_path not in transport_paths:
        raise RuntimeAttestationError("forward verifier ssh config is not content-bound")
    if process_identity_path not in transport_paths:
        raise RuntimeAttestationError("forward SSH identity is not content-bound")

    required_options = {
        "batchmode": "yes",
        "stricthostkeychecking": "yes",
        "identitiesonly": "yes",
        "globalknownhostsfile": "/dev/null",
        "passwordauthentication": "no",
        "kbdinteractiveauthentication": "no",
    }
    process_options = _ssh_option_map(argv)
    verifier_options = _ssh_option_map(verification_argv)
    for option_name, expected_value in required_options.items():
        if (
            process_options.get(option_name) != expected_value
            or verifier_options.get(option_name) != expected_value
        ):
            raise RuntimeAttestationError(
                "forward SSH process/verifier options are not fail-closed"
            )
    process_known_hosts = process_options.get("userknownhostsfile")
    verifier_known_hosts = verifier_options.get("userknownhostsfile")
    if not process_known_hosts or process_known_hosts != verifier_known_hosts:
        raise RuntimeAttestationError(
            "forward SSH known-hosts path must be explicit and identical"
        )
    if process_known_hosts not in transport_paths:
        raise RuntimeAttestationError(
            "forward SSH known-hosts file is not content-bound"
        )
    target = verification_argv[-1]
    if target.startswith("-") or argv[-1] != target:
        raise RuntimeAttestationError("forward and verifier SSH targets disagree")
    return payload


def _require_trusted_ancestors(
    path: Path, *, trusted_uid: int, security_anchor: Path
) -> None:
    resolved = path.resolve(strict=True)
    anchor = security_anchor.resolve(strict=True)
    try:
        resolved.relative_to(anchor)
    except ValueError as exc:
        raise RuntimeAttestationError(
            f"trusted path escapes security anchor {anchor}: {resolved}"
        ) from exc
    current = resolved if resolved.is_dir() else resolved.parent
    while True:
        status = current.stat()
        if status.st_uid != trusted_uid:
            raise RuntimeAttestationError(f"trusted ancestor has wrong uid: {current}")
        if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeAttestationError(
                f"trusted ancestor is group/other writable: {current}"
            )
        if current == anchor:
            break
        current = current.parent


def _require_trusted_regular(
    path: Path, *, trusted_uid: int, security_anchor: Path, artifact: bool = False
) -> Path:
    if path.is_symlink():
        raise RuntimeAttestationError(
            f"trusted regular file must not be a symlink: {path}"
        )
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_uid != trusted_uid:
        raise RuntimeAttestationError(
            f"trusted regular file has wrong type/uid: {resolved}"
        )
    if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeAttestationError(
            f"trusted regular file is group/other writable: {resolved}"
        )
    if artifact and status.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeAttestationError(
            f"attestation artifact must have no write bits: {resolved}"
        )
    _require_trusted_ancestors(
        resolved, trusted_uid=trusted_uid, security_anchor=security_anchor
    )
    return resolved


def _require_trusted_tree(
    path: Path, *, trusted_uid: int, security_anchor: Path
) -> Path:
    resolved = path.resolve(strict=True)
    _require_trusted_ancestors(
        resolved, trusted_uid=trusted_uid, security_anchor=security_anchor
    )
    for candidate in [resolved, *_tree_paths(resolved)]:
        status = candidate.stat()
        if status.st_uid != trusted_uid:
            raise RuntimeAttestationError(
                f"runtime tree entry has wrong uid: {candidate}"
            )
        if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeAttestationError(
                f"runtime tree entry is group/other writable: {candidate}"
            )
    return resolved


def load_trusted_attestation(
    path: Path, *, trusted_uid: int = 0, security_anchor: Path = Path("/")
) -> dict[str, Any]:
    resolved = _require_trusted_regular(
        path.expanduser(),
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
        artifact=True,
    )
    try:
        payload = json.loads(_read_stable_file(resolved))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAttestationError(
            f"runtime attestation is not valid UTF-8 JSON: {resolved}"
        ) from exc
    return _validate_artifact(payload)


def _proc_address_hex(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    packed = address.packed
    if address.version == 4:
        return packed[::-1].hex().upper()
    return (
        b"".join(packed[index : index + 4][::-1] for index in range(0, 16, 4))
        .hex()
        .upper()
    )


def _listener_inodes(proc_root: Path, endpoint: Mapping[str, Any]) -> set[str]:
    address = ipaddress.ip_address(endpoint["host"])
    desired_address = _proc_address_hex(address)
    wildcard = "0" * (8 if address.version == 4 else 32)
    filename = "tcp" if address.version == 4 else "tcp6"
    table = proc_root / "net" / filename
    try:
        lines = _read_proc_file(table).decode("ascii", errors="strict").splitlines()[1:]
    except UnicodeDecodeError as exc:
        raise RuntimeAttestationError(
            f"kernel socket table is not ASCII: {table}"
        ) from exc
    matches: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        try:
            local_address, local_port = fields[1].rsplit(":", 1)
            port = int(local_port, 16)
        except (ValueError, IndexError):
            raise RuntimeAttestationError(
                f"malformed kernel socket table row: {line!r}"
            )
        if port == endpoint["port"] and local_address.upper() in {
            desired_address,
            wildcard,
        }:
            matches.add(fields[9])
    if len(matches) != 1:
        raise RuntimeAttestationError(
            f"expected exactly one listening socket for {endpoint['host']}:{endpoint['port']}, "
            f"observed inodes={sorted(matches)!r}"
        )
    return matches


def _listener_pid(proc_root: Path, inode: str) -> int:
    owners: set[int] = set()
    expected = f"socket:[{inode}]"
    for candidate in proc_root.iterdir():
        if not candidate.name.isdecimal() or not candidate.is_dir():
            continue
        fd_root = candidate / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == expected:
                owners.add(int(candidate.name))
                break
    if len(owners) != 1:
        raise RuntimeAttestationError(
            f"expected exactly one process owning listener inode {inode}, observed {sorted(owners)!r}"
        )
    return next(iter(owners))


def _process_start_time(proc_root: Path, pid: int) -> str:
    raw = (
        _read_proc_file(proc_root / str(pid) / "stat")
        .decode("utf-8", errors="strict")
        .strip()
    )
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeAttestationError("listener process stat is malformed")
    fields = raw[closing + 1 :].strip().split()
    if len(fields) <= 19:
        raise RuntimeAttestationError("listener process stat lacks start time")
    return fields[19]


def _resolved_proc_link(path: Path, label: str) -> Path:
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise RuntimeAttestationError(
            f"cannot resolve listener process {label}: {exc}"
        ) from exc
    if target.endswith(" (deleted)"):
        raise RuntimeAttestationError(
            f"listener process {label} references a deleted object"
        )
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = (path.parent / candidate).resolve(strict=True)
    else:
        candidate = candidate.resolve(strict=True)
    return candidate


def verify_forward_attestation_payload(
    attestation_payload: Mapping[str, Any],
    expected_base_url: str,
    *,
    proc_root: Path = Path("/proc"),
    trusted_uid: int = 0,
    security_anchor: Path = Path("/"),
) -> dict[str, Any]:
    """Prove that a 157 listener is the approved SSH forward to 8222."""

    artifact = _validate_forward_artifact(dict(attestation_payload))
    endpoint = _endpoint_contract(artifact["endpoint"])
    if endpoint_contract_from_url(expected_base_url) != endpoint:
        raise RuntimeAttestationError(
            "scheduler endpoint does not match the approved forward listener"
        )
    inode = next(iter(_listener_inodes(proc_root, endpoint)))
    pid = _listener_pid(proc_root, inode)
    start_time_before = _process_start_time(proc_root, pid)
    process_root = proc_root / str(pid)
    argv = _parse_cmdline(_read_proc_file(process_root / "cmdline"))
    cgroup = _normalize_cgroup(_read_proc_file(process_root / "cgroup"))
    unit = _systemd_unit(cgroup)
    executable_path = _resolved_proc_link(process_root / "exe", "executable")
    working_directory = _resolved_proc_link(
        process_root / "cwd", "working directory"
    )
    expected = artifact["process"]
    if process_root.stat().st_uid != expected["uid"]:
        raise RuntimeAttestationError("forward listener process uid mismatch")
    if redact_argv(argv) != expected["argv_redacted"]:
        raise RuntimeAttestationError("forward listener command line mismatch")
    if cgroup != expected["cgroup"] or unit != expected["systemd_unit"]:
        raise RuntimeAttestationError("forward listener cgroup/systemd unit mismatch")
    if str(working_directory) != expected["working_directory"]:
        raise RuntimeAttestationError("forward listener working directory mismatch")
    if executable_path != Path(expected["executable"]["path"]):
        raise RuntimeAttestationError("forward listener executable path mismatch")
    observed_executable = _verify_descriptor(
        expected["executable"],
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
    )
    observed_transport_files = [
        _verify_descriptor(
            descriptor,
            trusted_uid=trusted_uid,
            security_anchor=security_anchor,
        )
        for descriptor in artifact["transport_files"]
    ]
    start_time_after = _process_start_time(proc_root, pid)
    if start_time_after != start_time_before:
        raise RuntimeAttestationError(
            "forward listener PID was reused during verification"
        )
    final_inode = next(iter(_listener_inodes(proc_root, endpoint)))
    final_pid = _listener_pid(proc_root, final_inode)
    if final_inode != inode or final_pid != pid:
        raise RuntimeAttestationError("forward listener changed during verification")
    return {
        "schema": FORWARD_VERIFICATION_SCHEMA,
        "artifact_sha256": artifact["artifact_sha256"],
        "host_label": artifact["host_label"],
        "endpoint": endpoint,
        "destination": artifact["destination"],
        "listener_inode": inode,
        "pid": pid,
        "process_start_time_ticks": start_time_before,
        "systemd_unit": unit,
        "cgroup": cgroup,
        "executable": observed_executable,
        "transport_files": observed_transport_files,
    }


def _http_models(endpoint: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    prefix = endpoint["api_prefix"]
    host = endpoint["host"]
    authority = f"[{host}]" if ipaddress.ip_address(host).version == 6 else host
    url = f"http://{authority}:{endpoint['port']}{prefix}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback enforced
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeAttestationError(
            f"cannot query attested endpoint models: {exc}"
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeAttestationError(
            "attested endpoint /models response is not JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeAttestationError(
            "attested endpoint /models response is not an object"
        )
    return value


def _display_model_id(payload: Any) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeAttestationError("/models response does not contain a data list")
    identifiers = [
        item.get("id")
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if len(identifiers) != 1:
        raise RuntimeAttestationError(
            f"/models must expose exactly one model id: {identifiers!r}"
        )
    return identifiers[0]


def _verify_descriptor(
    expected: Mapping[str, Any], *, trusted_uid: int, security_anchor: Path
) -> dict[str, Any]:
    path = _require_trusted_regular(
        Path(expected["path"]), trusted_uid=trusted_uid, security_anchor=security_anchor
    )
    observed = describe_file(path)
    if observed != dict(expected):
        raise RuntimeAttestationError(
            f"file identity mismatch for {path}: expected bytes/sha "
            f"{expected['bytes']}/{expected['sha256']}, observed "
            f"{observed['bytes']}/{observed['sha256']}"
        )
    return observed


def verify_runtime_attestation_payload(
    attestation_payload: Mapping[str, Any],
    expected_model: str,
    expected_base_url: str,
    *,
    api_key: str = "",
    proc_root: Path = Path("/proc"),
    trusted_uid: int = 0,
    security_anchor: Path = Path("/"),
    models_probe: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify a local live runtime against an already-trusted artifact payload.

    The caller is responsible for obtaining ``attestation_payload`` from a
    root-owned, content-addressed approval.  File/tree ownership and hashes are
    still checked live here.  ``models_probe`` exists solely for deterministic
    tests; production callers should leave it unset.
    """

    artifact = _validate_artifact(dict(attestation_payload))
    if expected_model != artifact["model"]["name"]:
        raise RuntimeAttestationError(
            f"requested model {expected_model!r} does not match trusted attestation "
            f"{artifact['model']['name']!r}"
        )
    endpoint = _endpoint_contract(artifact["endpoint"])
    if endpoint_contract_from_url(expected_base_url) != endpoint:
        raise RuntimeAttestationError(
            "requested endpoint does not match the trusted attestation"
        )

    inode = next(iter(_listener_inodes(proc_root, endpoint)))
    pid = _listener_pid(proc_root, inode)
    start_time_before = _process_start_time(proc_root, pid)
    process_root = proc_root / str(pid)
    argv = _parse_cmdline(_read_proc_file(process_root / "cmdline"))
    environment = _parse_environ(_read_proc_file(process_root / "environ"))
    semantic_environment = _semantic_environment(environment)
    cgroup = _normalize_cgroup(_read_proc_file(process_root / "cgroup"))
    unit = _systemd_unit(cgroup)
    executable_path = _resolved_proc_link(process_root / "exe", "executable")
    working_directory = _resolved_proc_link(process_root / "cwd", "working directory")
    process_uid = process_root.stat().st_uid

    expected_process = artifact["process"]
    if process_uid != expected_process["uid"]:
        raise RuntimeAttestationError(
            f"listener process uid mismatch: expected {expected_process['uid']}, observed {process_uid}"
        )
    model_argument = _model_argument(argv)
    actual_argument_path = _require_absolute_path(
        model_argument, "live vLLM model argument"
    )
    if actual_argument_path.is_symlink():
        raise RuntimeAttestationError("live vLLM model argument must not be a symlink")
    actual_weight_path = actual_argument_path.resolve(strict=True)
    expected_weight_path = Path(artifact["model"]["weight"]["path"])
    if (
        actual_argument_path != expected_weight_path
        or actual_weight_path != expected_weight_path
    ):
        raise RuntimeAttestationError(
            f"actual model weight path mismatch: expected {expected_weight_path}, observed {actual_weight_path}"
        )
    if redact_argv(argv) != expected_process["argv_redacted"]:
        raise RuntimeAttestationError(
            "listener process command line does not match attestation"
        )
    if semantic_environment != expected_process["environment"]:
        raise RuntimeAttestationError(
            "listener process semantic environment does not match attestation"
        )
    if cgroup != expected_process["cgroup"] or unit != expected_process["systemd_unit"]:
        raise RuntimeAttestationError(
            "listener process cgroup/systemd unit does not match attestation"
        )
    if str(working_directory) != expected_process["working_directory"]:
        raise RuntimeAttestationError(
            "listener process working directory does not match attestation"
        )

    gpu_raw = semantic_environment.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_raw.isdecimal() or int(gpu_raw) != expected_process["gpu_index"]:
        raise RuntimeAttestationError(
            "live CUDA_VISIBLE_DEVICES does not select the attested single GPU"
        )
    if artifact["host_label"] == "8222" and (
        int(gpu_raw) == 3 or endpoint["port"] == 18073
    ):
        raise RuntimeAttestationError(
            "8222 GPU3 and port 18073 are reserved and forbidden"
        )

    observed_parameters = launch_parameters(argv)
    _validate_launch_policy(
        parameters=observed_parameters,
        endpoint=endpoint,
        model_name=artifact["model"]["name"],
    )
    if observed_parameters != expected_process["launch_parameters"]:
        raise RuntimeAttestationError("live launch parameters do not match attestation")

    observed_weight = _verify_descriptor(
        artifact["model"]["weight"],
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
    )

    if executable_path != Path(expected_process["executable"]["path"]):
        raise RuntimeAttestationError(
            "listener executable path does not match attestation"
        )
    observed_executable = _verify_descriptor(
        expected_process["executable"],
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
    )

    runtime_path = Path(artifact["runtime_tree"]["path"])
    python_paths = [
        Path(item).resolve()
        for item in semantic_environment.get("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    if python_paths != [runtime_path]:
        raise RuntimeAttestationError(
            "live PYTHONPATH must contain exactly the attested inference runtime tree"
        )
    _require_trusted_tree(
        runtime_path, trusted_uid=trusted_uid, security_anchor=security_anchor
    )
    observed_tree = describe_tree(runtime_path)
    if observed_tree != artifact["runtime_tree"]:
        raise RuntimeAttestationError(
            "live inference runtime tree digest/inventory does not match attestation"
        )

    probe = models_probe or _http_models
    model_response = probe(endpoint, api_key)
    display_model = _display_model_id(model_response)
    if display_model != artifact["model"]["name"]:
        raise RuntimeAttestationError(
            "endpoint display model does not match attestation"
        )

    start_time_after = _process_start_time(proc_root, pid)
    if start_time_after != start_time_before:
        raise RuntimeAttestationError(
            "listener PID was reused during runtime verification"
        )
    final_inode = next(iter(_listener_inodes(proc_root, endpoint)))
    final_pid = _listener_pid(proc_root, final_inode)
    if final_inode != inode or final_pid != pid:
        raise RuntimeAttestationError("listener changed during runtime verification")

    return {
        "schema": VERIFICATION_SCHEMA,
        "artifact_sha256": artifact["artifact_sha256"],
        "host_label": artifact["host_label"],
        "endpoint": endpoint,
        "listener_inode": inode,
        "pid": pid,
        "process_start_time_ticks": start_time_before,
        "systemd_unit": unit,
        "cgroup": cgroup,
        "gpu_index": int(gpu_raw),
        "display_model": display_model,
        "actual_weight": observed_weight,
        "executable": observed_executable,
        "runtime_tree_sha256": observed_tree["tree_sha256"],
        "launch_parameters": observed_parameters,
        "semantic_environment_sha256": canonical_sha256(semantic_environment),
    }


def verify_runtime_attestation(
    attestation_path: Path,
    expected_model: str,
    expected_base_url: str,
    *,
    api_key: str = "",
    proc_root: Path = Path("/proc"),
    trusted_uid: int = 0,
    security_anchor: Path = Path("/"),
    models_probe: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a trusted artifact file and verify its local live runtime."""

    artifact = load_trusted_attestation(
        attestation_path,
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
    )
    return verify_runtime_attestation_payload(
        artifact,
        expected_model,
        expected_base_url,
        api_key=api_key,
        proc_root=proc_root,
        trusted_uid=trusted_uid,
        security_anchor=security_anchor,
        models_probe=models_probe,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify the live local listener")
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--model", required=True)
    verify.add_argument("--endpoint", required=True)
    verify.add_argument("--api-key", default=os.environ.get("RWKV_INFER_API_KEY", ""))
    verify.add_argument(
        "--proc-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS
    )
    describe = subparsers.add_parser(
        "describe", help="print a file/tree descriptor for provisioning"
    )
    describe.add_argument("path", type=Path)
    describe.add_argument("--tree", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "describe":
            result = describe_tree(args.path) if args.tree else describe_file(args.path)
        else:
            result = verify_runtime_attestation(
                args.attestation,
                args.model,
                args.endpoint,
                api_key=args.api_key,
                proc_root=args.proc_root,
            )
    except RuntimeAttestationError as exc:
        print(f"runtime-attestation: FAIL: {exc}", file=sys.stderr)
        return 42
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
