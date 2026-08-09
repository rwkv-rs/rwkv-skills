"""Immutable record/replay evidence for strict-46 external Math judging.

The Math replay attestation is executed under several ``PYTHONHASHSEED``
values.  Calling an external Judge independently in every process would make
that attestation both expensive and non-reproducible.  This module records one
successful, deterministic Judge decision for every distinct semantic request
and replays those decisions without network access.

The transcript deliberately contains no API key.  Endpoint identity is
reduced to scheme, host, port and path; userinfo, query parameters and
fragments are discarded.  A transcript is immutable once persisted: a second
writer may only present byte-for-byte identical canonical content.

This module does not read or write the evaluation database and does not alter
the production scorer in :mod:`src.eval.metrics.free_response`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Protocol
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from src.eval.metrics.free_response import (
    LLM_JUDGE_PROTOCOL_VERSION,
    LLM_JUDGE_RESPONSE_CONTRACT,
    LLMJudgeConfig,
    LLMJudgeStats,
    llm_judge_protocol,
    llm_judge_protocol_stats_reasons,
)


TRANSCRIPT_SCHEMA_VERSION = "rwkv.strict46.judge_transcript.v1"
SHA256_HEX_LENGTH = 64
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_LOCK_WAIT_SECONDS = 30.0
_LOCK_STALE_SECONDS = 300.0


class JudgeTranscriptError(RuntimeError):
    """Base class for fail-closed transcript errors."""


class JudgeTranscriptIntegrityError(JudgeTranscriptError):
    """The persisted artifact is malformed, tampered with, or incompatible."""


class JudgeTranscriptUsageError(JudgeTranscriptError):
    """Replay requests do not exactly match the recorded coordinate set."""


class _JudgeLike(Protocol):
    config: LLMJudgeConfig
    last_run_stats: LLMJudgeStats | None

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]: ...


def _canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise JudgeTranscriptIntegrityError(
            f"transcript value is not canonical JSON: {exc}"
        ) from exc
    return rendered.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def normalize_judge_text(value: object) -> str:
    """Normalize transport-neutral text without discarding semantic spacing."""

    if not isinstance(value, str):
        raise JudgeTranscriptIntegrityError("Judge input fields must be strings")
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    )


def sanitize_endpoint_identity(value: object) -> str:
    """Return a credential-free, query-free endpoint identity.

    The path remains part of the identity because two OpenAI-compatible APIs
    on the same host may implement different Judge deployments.
    """

    raw = str(value or DEFAULT_OPENAI_BASE_URL).strip()
    try:
        split = urlsplit(raw)
        port = split.port
    except ValueError as exc:
        raise JudgeTranscriptIntegrityError("invalid Judge endpoint URL") from exc
    scheme = split.scheme.lower()
    if scheme not in {"http", "https"} or not split.hostname:
        raise JudgeTranscriptIntegrityError(
            "Judge endpoint must be an absolute HTTP(S) URL"
        )
    hostname = split.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = split.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))


def _delegate_endpoint(delegate: _JudgeLike) -> str:
    client = getattr(delegate, "client", None)
    client_base_url = getattr(client, "base_url", None)
    return sanitize_endpoint_identity(
        client_base_url or delegate.config.base_url or DEFAULT_OPENAI_BASE_URL
    )


def _protocol_payload(
    config: LLMJudgeConfig,
    *,
    endpoint_url: object | None = None,
) -> dict[str, object]:
    judge_protocol = llm_judge_protocol(config)
    judge_fingerprint = _canonical_sha256(judge_protocol)
    stats = {**judge_protocol, "protocol_fingerprint_sha256": judge_fingerprint}
    reasons = llm_judge_protocol_stats_reasons(stats)
    if reasons:
        raise JudgeTranscriptIntegrityError(
            "Judge protocol is not deterministic/current: " + ",".join(reasons)
        )
    endpoint_identity = sanitize_endpoint_identity(
        endpoint_url or config.base_url or DEFAULT_OPENAI_BASE_URL
    )
    envelope = {
        "judge_protocol": judge_protocol,
        "judge_protocol_fingerprint_sha256": judge_fingerprint,
        "endpoint_identity": endpoint_identity,
        "endpoint_identity_sha256": _sha256_bytes(endpoint_identity.encode("utf-8")),
    }
    envelope["protocol_fingerprint_sha256"] = _canonical_sha256(envelope)
    return envelope


def _normalized_input(item: tuple[str, str, str]) -> dict[str, str]:
    if not isinstance(item, tuple) or len(item) != 3:
        raise JudgeTranscriptIntegrityError(
            "Judge input must be a (question, reference, prediction) tuple"
        )
    question, reference, prediction = item
    return {
        "question": normalize_judge_text(question),
        "reference": normalize_judge_text(reference),
        "prediction": normalize_judge_text(prediction),
    }


def _rendered_prompt(config: LLMJudgeConfig, item: Mapping[str, str]) -> str:
    prompt = config.prompt_template
    prompt = prompt.replace("<Q>", item["question"])
    prompt = prompt.replace("<REF>", item["reference"])
    prompt = prompt.replace("<A>", item["prediction"])
    return normalize_judge_text(prompt)


def _request_identity(
    config: LLMJudgeConfig,
    protocol: Mapping[str, object],
    item: tuple[str, str, str],
) -> tuple[str, dict[str, object]]:
    normalized = _normalized_input(item)
    normalized_sha = _canonical_sha256(normalized)
    prompt_sha = _sha256_bytes(_rendered_prompt(config, normalized).encode("utf-8"))
    identity: dict[str, object] = {
        "protocol_fingerprint_sha256": protocol["protocol_fingerprint_sha256"],
        "input": normalized,
        "input_sha256": normalized_sha,
        "rendered_prompt_sha256": prompt_sha,
    }
    return _canonical_sha256(identity), identity


def _response_sha256(
    result: bool,
    *,
    protocol_fingerprint_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "protocol_fingerprint_sha256": protocol_fingerprint_sha256,
            "response_contract": LLM_JUDGE_RESPONSE_CONTRACT,
            "literal": "True" if result else "False",
        }
    )


def _stats_for(config: LLMJudgeConfig, count: int) -> LLMJudgeStats:
    return LLMJudgeStats(
        total=count,
        parsed_count=count,
        protocol=llm_judge_protocol(config),
    )


@dataclass(frozen=True)
class JudgeTranscriptArtifact:
    path: str
    sha256: str
    payload: dict[str, Any]

    def provenance(self) -> dict[str, object]:
        statistics = self.payload.get("statistics")
        return {
            "schema_version": self.payload.get("schema_version"),
            "sha256": self.sha256,
            "protocol_fingerprint_sha256": list(
                self.payload.get("protocol_fingerprint_sha256", [])
            ),
            "statistics": dict(statistics) if isinstance(statistics, dict) else {},
        }


def _validate_protocol(protocol: object) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise JudgeTranscriptIntegrityError("transcript protocol is not an object")
    required = {
        "judge_protocol",
        "judge_protocol_fingerprint_sha256",
        "endpoint_identity",
        "endpoint_identity_sha256",
        "protocol_fingerprint_sha256",
    }
    if set(protocol) != required:
        raise JudgeTranscriptIntegrityError("transcript protocol fields differ")
    judge_protocol = protocol["judge_protocol"]
    if not isinstance(judge_protocol, dict):
        raise JudgeTranscriptIntegrityError("judge_protocol is not an object")
    if judge_protocol.get("protocol_version") != LLM_JUDGE_PROTOCOL_VERSION:
        raise JudgeTranscriptIntegrityError("Judge protocol version drift")
    judge_fingerprint = _canonical_sha256(judge_protocol)
    if protocol["judge_protocol_fingerprint_sha256"] != judge_fingerprint:
        raise JudgeTranscriptIntegrityError("Judge protocol fingerprint mismatch")
    stats = {**judge_protocol, "protocol_fingerprint_sha256": judge_fingerprint}
    reasons = llm_judge_protocol_stats_reasons(stats)
    if reasons:
        raise JudgeTranscriptIntegrityError(
            "non-deterministic Judge protocol: " + ",".join(reasons)
        )
    endpoint = str(protocol["endpoint_identity"])
    if sanitize_endpoint_identity(endpoint) != endpoint:
        raise JudgeTranscriptIntegrityError("endpoint identity is not sanitized")
    if protocol["endpoint_identity_sha256"] != _sha256_bytes(
        endpoint.encode("utf-8")
    ):
        raise JudgeTranscriptIntegrityError("endpoint identity hash mismatch")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_fingerprint_sha256"}
    if protocol["protocol_fingerprint_sha256"] != _canonical_sha256(unsigned):
        raise JudgeTranscriptIntegrityError("transcript protocol fingerprint mismatch")
    return dict(protocol)


def _validate_entry(
    entry: object,
    *,
    protocols: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise JudgeTranscriptIntegrityError("transcript entry is not an object")
    required = {
        "input_key_sha256",
        "protocol_fingerprint_sha256",
        "input",
        "input_sha256",
        "rendered_prompt_sha256",
        "result",
        "response_sha256",
        "scope_occurrences",
    }
    if set(entry) != required:
        raise JudgeTranscriptIntegrityError("transcript entry fields differ")
    protocol_fingerprint = str(entry["protocol_fingerprint_sha256"])
    if protocol_fingerprint not in protocols:
        raise JudgeTranscriptIntegrityError("entry refers to an unknown protocol")
    normalized = entry["input"]
    if not isinstance(normalized, dict) or set(normalized) != {
        "question",
        "reference",
        "prediction",
    }:
        raise JudgeTranscriptIntegrityError("entry input fields differ")
    normalized_again = {
        key: normalize_judge_text(normalized[key])
        for key in ("question", "reference", "prediction")
    }
    if normalized_again != normalized:
        raise JudgeTranscriptIntegrityError("entry input is not normalized")
    if entry["input_sha256"] != _canonical_sha256(normalized):
        raise JudgeTranscriptIntegrityError("entry input hash mismatch")
    if not _is_sha256(entry["rendered_prompt_sha256"]):
        raise JudgeTranscriptIntegrityError("entry prompt hash is invalid")
    identity = {
        "protocol_fingerprint_sha256": protocol_fingerprint,
        "input": normalized,
        "input_sha256": entry["input_sha256"],
        "rendered_prompt_sha256": entry["rendered_prompt_sha256"],
    }
    if entry["input_key_sha256"] != _canonical_sha256(identity):
        raise JudgeTranscriptIntegrityError("entry request key mismatch")
    result = entry["result"]
    if not isinstance(result, bool):
        raise JudgeTranscriptIntegrityError("entry result is not boolean")
    if entry["response_sha256"] != _response_sha256(
        result,
        protocol_fingerprint_sha256=protocol_fingerprint,
    ):
        raise JudgeTranscriptIntegrityError("entry response hash mismatch")
    occurrences = entry["scope_occurrences"]
    if not isinstance(occurrences, dict) or not occurrences:
        raise JudgeTranscriptIntegrityError("entry scope occurrences are missing")
    for scope, count in occurrences.items():
        if not isinstance(scope, str) or not scope:
            raise JudgeTranscriptIntegrityError("entry scope is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise JudgeTranscriptIntegrityError("entry occurrence count is invalid")
    return dict(entry)


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeTranscriptIntegrityError("transcript is not an object")
    required = {
        "schema_version",
        "complete",
        "record_pythonhashseed",
        "protocol_fingerprint_sha256",
        "protocols",
        "scopes",
        "entries",
        "statistics",
        "transcript_sha256",
    }
    if set(payload) != required:
        raise JudgeTranscriptIntegrityError("transcript top-level fields differ")
    if payload["schema_version"] != TRANSCRIPT_SCHEMA_VERSION:
        raise JudgeTranscriptIntegrityError("transcript schema version drift")
    if payload["complete"] is not True:
        raise JudgeTranscriptIntegrityError("transcript is not finalized")
    if payload["record_pythonhashseed"] != "42":
        raise JudgeTranscriptIntegrityError(
            "external Judge transcript must be recorded under PYTHONHASHSEED=42"
        )
    protocols_raw = payload["protocols"]
    if not isinstance(protocols_raw, list) or not protocols_raw:
        raise JudgeTranscriptIntegrityError("transcript has no protocols")
    protocols_list = [_validate_protocol(protocol) for protocol in protocols_raw]
    fingerprints = [str(protocol["protocol_fingerprint_sha256"]) for protocol in protocols_list]
    if fingerprints != sorted(fingerprints) or len(fingerprints) != len(set(fingerprints)):
        raise JudgeTranscriptIntegrityError("protocols are duplicate or not canonical")
    if payload["protocol_fingerprint_sha256"] != fingerprints:
        raise JudgeTranscriptIntegrityError("protocol fingerprint index mismatch")
    protocols = dict(zip(fingerprints, protocols_list, strict=True))

    scopes_raw = payload["scopes"]
    if not isinstance(scopes_raw, dict) or not scopes_raw:
        raise JudgeTranscriptIntegrityError("transcript has no declared scopes")
    if list(scopes_raw) != sorted(scopes_raw):
        raise JudgeTranscriptIntegrityError("transcript scopes are not canonical")
    scopes: dict[str, list[str]] = {}
    for scope, scope_protocols in scopes_raw.items():
        if not isinstance(scope, str) or not scope:
            raise JudgeTranscriptIntegrityError("transcript scope is invalid")
        if (
            not isinstance(scope_protocols, list)
            or not scope_protocols
            or scope_protocols != sorted(scope_protocols)
            or len(scope_protocols) != len(set(scope_protocols))
            or any(fingerprint not in protocols for fingerprint in scope_protocols)
        ):
            raise JudgeTranscriptIntegrityError(
                "transcript scope protocol declarations are invalid"
            )
        scopes[scope] = list(scope_protocols)

    entries_raw = payload["entries"]
    if not isinstance(entries_raw, list):
        raise JudgeTranscriptIntegrityError("transcript entries are not a list")
    entries = [_validate_entry(entry, protocols=protocols) for entry in entries_raw]
    keys = [str(entry["input_key_sha256"]) for entry in entries]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise JudgeTranscriptIntegrityError("entries are duplicate or not canonical")
    for entry in entries:
        fingerprint = str(entry["protocol_fingerprint_sha256"])
        for scope in entry["scope_occurrences"]:
            if scope not in scopes or fingerprint not in scopes[scope]:
                raise JudgeTranscriptIntegrityError(
                    "entry uses an undeclared scope/protocol pair"
                )

    total_occurrences = sum(
        sum(int(value) for value in entry["scope_occurrences"].values())
        for entry in entries
    )
    true_occurrences = sum(
        sum(int(value) for value in entry["scope_occurrences"].values())
        for entry in entries
        if entry["result"] is True
    )
    expected_statistics = {
        "protocol_count": len(protocols),
        "unique_input_count": len(entries),
        "actual_judge_call_count": len(entries),
        "coordinate_count": total_occurrences,
        "true_coordinate_count": true_occurrences,
        "false_coordinate_count": total_occurrences - true_occurrences,
        "scope_count": len(scopes),
    }
    if payload["statistics"] != expected_statistics:
        raise JudgeTranscriptIntegrityError("transcript statistics mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "transcript_sha256"}
    computed = _canonical_sha256(unsigned)
    if payload["transcript_sha256"] != computed:
        raise JudgeTranscriptIntegrityError("transcript content SHA mismatch")
    return dict(payload)


def load_judge_transcript(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> JudgeTranscriptArtifact:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise JudgeTranscriptIntegrityError(
            f"cannot read Judge transcript: {type(exc).__name__}"
        ) from exc
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeTranscriptIntegrityError("Judge transcript is invalid JSON") from exc
    payload = _validate_payload(decoded)
    semantic_sha = str(payload["transcript_sha256"])
    if expected_sha256 is not None and semantic_sha != str(expected_sha256).lower():
        raise JudgeTranscriptIntegrityError(
            f"Judge transcript SHA {semantic_sha} != expected {expected_sha256}"
        )
    canonical = _canonical_json_bytes(payload) + b"\n"
    if raw != canonical:
        raise JudgeTranscriptIntegrityError("Judge transcript is not canonical JSON")
    return JudgeTranscriptArtifact(
        path=str(path.resolve()),
        sha256=semantic_sha,
        payload=payload,
    )


def _exclusive_file_lock(lock_path: Path) -> Iterable[None]:
    """A tiny dependency-free cross-process lock used only during publication."""

    class _LockContext:
        def __enter__(self) -> None:
            deadline = time.monotonic() + _LOCK_WAIT_SECONDS
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                except FileExistsError:
                    try:
                        stale = time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS
                    except FileNotFoundError:
                        continue
                    if stale:
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
                        continue
                    if time.monotonic() >= deadline:
                        raise JudgeTranscriptError("timed out waiting for transcript lock")
                    time.sleep(0.01)
                    continue
                try:
                    os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                finally:
                    os.close(descriptor)
                return None

        def __exit__(self, *_args: object) -> None:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    return _LockContext()  # type: ignore[return-value]


def persist_judge_transcript_immutable(
    path: Path,
    payload: Mapping[str, object],
) -> JudgeTranscriptArtifact:
    validated = _validate_payload(dict(payload))
    canonical = _canonical_json_bytes(validated) + b"\n"
    lock_path = path.with_name(f".{path.name}.lock")
    with _exclusive_file_lock(lock_path):
        if path.exists():
            existing = load_judge_transcript(path)
            if path.read_bytes() != canonical:
                raise JudgeTranscriptIntegrityError(
                    "refusing to overwrite a different immutable Judge transcript"
                )
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)
    return load_judge_transcript(path)


class JudgeTranscriptRecorder:
    """Shared, thread-safe builder spanning all source tasks in one run."""

    def __init__(self, path: Path) -> None:
        if str(os.environ.get("PYTHONHASHSEED") or "") != "42":
            raise JudgeTranscriptUsageError(
                "external Judge recording requires interpreter-start "
                "PYTHONHASHSEED=42"
            )
        self.path = path
        self._protocols: dict[str, dict[str, object]] = {}
        self._scopes: dict[str, set[str]] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._finalized: JudgeTranscriptArtifact | None = None
        self._existing_artifact = (
            load_judge_transcript(path) if path.exists() else None
        )
        self._existing_replayer = (
            JudgeTranscriptReplayer(self._existing_artifact)
            if self._existing_artifact is not None
            else None
        )

    def wrap(
        self,
        delegate: _JudgeLike,
        *,
        scope: object,
        endpoint_url: object | None = None,
    ) -> "RecordingJudge":
        scope_text = str(scope).strip()
        if not scope_text:
            raise JudgeTranscriptUsageError("recording scope cannot be empty")
        endpoint = endpoint_url or _delegate_endpoint(delegate)
        if self._existing_replayer is not None:
            self._existing_replayer._register(
                delegate.config,
                scope=scope_text,
                endpoint_url=endpoint,
            )
        else:
            self._register(
                delegate.config,
                scope=scope_text,
                endpoint_url=endpoint,
            )
        return RecordingJudge(
            delegate,
            recorder=self,
            scope=scope_text,
            endpoint_url=endpoint,
        )

    def _register(
        self,
        config: LLMJudgeConfig,
        *,
        scope: str,
        endpoint_url: object | None,
    ) -> dict[str, object]:
        with self._lock:
            protocol = _protocol_payload(config, endpoint_url=endpoint_url)
            fingerprint = str(protocol["protocol_fingerprint_sha256"])
            existing = self._protocols.get(fingerprint)
            if existing is not None and existing != protocol:
                raise JudgeTranscriptIntegrityError("protocol SHA collision")
            self._protocols[fingerprint] = protocol
            self._scopes.setdefault(scope, set()).add(fingerprint)
            return protocol

    def _judge(
        self,
        delegate: _JudgeLike,
        *,
        scope: str,
        endpoint_url: object | None,
        items: list[tuple[str, str, str]],
    ) -> list[bool]:
        with self._lock:
            if self._existing_replayer is not None:
                return self._existing_replayer._judge(
                    delegate.config,
                    scope=scope,
                    endpoint_url=endpoint_url or _delegate_endpoint(delegate),
                    items=items,
                )
            if self._finalized is not None:
                raise JudgeTranscriptUsageError("recording continued after finalization")
            protocol = self._register(
                delegate.config,
                scope=scope,
                endpoint_url=endpoint_url or _delegate_endpoint(delegate),
            )
            protocol_fingerprint = str(protocol["protocol_fingerprint_sha256"])

            requests = [
                _request_identity(delegate.config, protocol, item) for item in items
            ]
            unseen: dict[str, tuple[str, str, str]] = {}
            identities: dict[str, dict[str, object]] = {}
            for item, (key, identity) in zip(items, requests, strict=True):
                identities[key] = identity
                if key not in self._entries:
                    normalized = identity["input"]
                    assert isinstance(normalized, dict)
                    unseen.setdefault(
                        key,
                        (
                            str(normalized["question"]),
                            str(normalized["reference"]),
                            str(normalized["prediction"]),
                        ),
                    )
            if unseen:
                unseen_keys = sorted(unseen)
                unique_items = [unseen[key] for key in unseen_keys]
                unique_results = delegate.judge(unique_items)
                if len(unique_results) != len(unique_items) or any(
                    not isinstance(result, bool) for result in unique_results
                ):
                    raise JudgeTranscriptIntegrityError(
                        "live Judge returned a malformed result vector"
                    )
                stats = delegate.last_run_stats
                if stats is None:
                    raise JudgeTranscriptIntegrityError("live Judge omitted run statistics")
                stats_payload = stats.as_dict()
                if (
                    int(stats_payload.get("total") or 0) != len(unique_items)
                    or int(stats_payload.get("parsed_count") or 0) != len(unique_items)
                    or int(stats_payload.get("error_count") or 0) != 0
                ):
                    raise JudgeTranscriptIntegrityError(
                        "live Judge did not parse every unique request successfully"
                    )
                protocol_reasons = llm_judge_protocol_stats_reasons(stats_payload)
                if protocol_reasons:
                    raise JudgeTranscriptIntegrityError(
                        "live Judge statistics protocol mismatch: "
                        + ",".join(protocol_reasons)
                    )
                for key, result in zip(unseen_keys, unique_results, strict=True):
                    identity = identities[key]
                    self._entries[key] = {
                        "input_key_sha256": key,
                        **identity,
                        "result": result,
                        "response_sha256": _response_sha256(
                            result,
                            protocol_fingerprint_sha256=protocol_fingerprint,
                        ),
                        "scope_occurrences": {},
                    }

            results: list[bool] = []
            for key, _identity in requests:
                entry = self._entries[key]
                occurrences = entry["scope_occurrences"]
                occurrences[scope] = int(occurrences.get(scope, 0)) + 1
                results.append(bool(entry["result"]))
            return results

    def _payload(self) -> dict[str, object]:
        if not self._protocols or not self._scopes:
            raise JudgeTranscriptUsageError(
                "cannot finalize a Judge transcript without declared scopes"
            )
        protocols = [self._protocols[key] for key in sorted(self._protocols)]
        entries: list[dict[str, object]] = []
        for key in sorted(self._entries):
            entry = dict(self._entries[key])
            entry["scope_occurrences"] = {
                scope: entry["scope_occurrences"][scope]
                for scope in sorted(entry["scope_occurrences"])
            }
            entries.append(entry)
        coordinate_count = sum(
            sum(int(value) for value in entry["scope_occurrences"].values())
            for entry in entries
        )
        true_coordinate_count = sum(
            sum(int(value) for value in entry["scope_occurrences"].values())
            for entry in entries
            if entry["result"] is True
        )
        unsigned: dict[str, object] = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "complete": True,
            "record_pythonhashseed": "42",
            "protocol_fingerprint_sha256": sorted(self._protocols),
            "protocols": protocols,
            "scopes": {
                scope: sorted(self._scopes[scope]) for scope in sorted(self._scopes)
            },
            "entries": entries,
            "statistics": {
                "protocol_count": len(protocols),
                "unique_input_count": len(entries),
                "actual_judge_call_count": len(entries),
                "coordinate_count": coordinate_count,
                "true_coordinate_count": true_coordinate_count,
                "false_coordinate_count": coordinate_count - true_coordinate_count,
                "scope_count": len(self._scopes),
            },
        }
        return {**unsigned, "transcript_sha256": _canonical_sha256(unsigned)}

    def persist(self) -> JudgeTranscriptArtifact:
        with self._lock:
            if self._existing_artifact is not None:
                assert self._existing_replayer is not None
                self._existing_replayer.assert_consumed()
                return self._existing_artifact
            if self._finalized is None:
                self._finalized = persist_judge_transcript_immutable(
                    self.path, self._payload()
                )
            return self._finalized


class RecordingJudge:
    """Judge-compatible adapter that records decisions from one live delegate."""

    def __init__(
        self,
        delegate: _JudgeLike,
        *,
        recorder: JudgeTranscriptRecorder,
        scope: object,
        endpoint_url: object | None = None,
    ) -> None:
        scope_text = str(scope).strip()
        if not scope_text:
            raise JudgeTranscriptUsageError("recording scope cannot be empty")
        self.config = delegate.config
        self.delegate = delegate
        self.recorder = recorder
        self.scope = scope_text
        self.endpoint_url = endpoint_url
        self.last_run_stats: LLMJudgeStats | None = None

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        results = self.recorder._judge(
            self.delegate,
            scope=self.scope,
            endpoint_url=self.endpoint_url,
            items=items,
        )
        self.last_run_stats = _stats_for(self.config, len(items))
        return results


class JudgeTranscriptReplayer:
    """Shared replay ledger that requires exact, full transcript consumption."""

    def __init__(self, artifact: JudgeTranscriptArtifact) -> None:
        self.artifact = artifact
        self._entries = {
            str(entry["input_key_sha256"]): entry
            for entry in artifact.payload["entries"]
        }
        self._protocols = {
            str(protocol["protocol_fingerprint_sha256"]): protocol
            for protocol in artifact.payload["protocols"]
        }
        self._expected_scopes = {
            str(scope): set(protocols)
            for scope, protocols in artifact.payload["scopes"].items()
        }
        self._wrapped_scopes: dict[str, set[str]] = {}
        self._consumed: Counter[tuple[str, str]] = Counter()
        self._lock = threading.RLock()

    def wrap(
        self,
        config: LLMJudgeConfig,
        *,
        scope: object,
        endpoint_url: object | None = None,
    ) -> "ReplayJudge":
        return ReplayJudge(
            config,
            artifact=self.artifact,
            replayer=self,
            scope=scope,
            endpoint_url=endpoint_url,
        )

    def _register(
        self,
        config: LLMJudgeConfig,
        *,
        scope: str,
        endpoint_url: object | None,
    ) -> dict[str, object]:
        protocol = _protocol_payload(config, endpoint_url=endpoint_url)
        fingerprint = str(protocol["protocol_fingerprint_sha256"])
        if self._protocols.get(fingerprint) != protocol:
            raise JudgeTranscriptUsageError(
                "Judge replay protocol or endpoint does not match transcript"
            )
        if fingerprint not in self._expected_scopes.get(scope, set()):
            raise JudgeTranscriptUsageError(
                "Judge replay scope/protocol pair is absent from transcript"
            )
        with self._lock:
            self._wrapped_scopes.setdefault(scope, set()).add(fingerprint)
        return protocol

    def _judge(
        self,
        config: LLMJudgeConfig,
        *,
        scope: str,
        endpoint_url: object | None,
        items: list[tuple[str, str, str]],
    ) -> list[bool]:
        protocol = self._register(
            config,
            scope=scope,
            endpoint_url=endpoint_url,
        )
        requests = [_request_identity(config, protocol, item) for item in items]
        with self._lock:
            additions: Counter[tuple[str, str]] = Counter()
            results: list[bool] = []
            for key, identity in requests:
                entry = self._entries.get(key)
                if entry is None or any(
                    entry.get(field) != identity[field]
                    for field in (
                        "protocol_fingerprint_sha256",
                        "input",
                        "input_sha256",
                        "rendered_prompt_sha256",
                    )
                ):
                    raise JudgeTranscriptUsageError(
                        f"unrecorded or drifted Judge input: {key}"
                    )
                expected = int(entry["scope_occurrences"].get(scope, 0))
                coordinate = (scope, key)
                additions[coordinate] += 1
                if self._consumed[coordinate] + additions[coordinate] > expected:
                    raise JudgeTranscriptUsageError(
                        f"duplicate/excess Judge coordinate for scope {scope}: {key}"
                    )
                results.append(bool(entry["result"]))
            self._consumed.update(additions)
            return results

    def assert_consumed(self) -> None:
        with self._lock:
            if self._wrapped_scopes != self._expected_scopes:
                raise JudgeTranscriptUsageError(
                    "Judge transcript scope/protocol declarations were not consumed "
                    "exactly"
                )
            expected = Counter(
                {
                    (scope, key): int(count)
                    for key, entry in self._entries.items()
                    for scope, count in entry["scope_occurrences"].items()
                }
            )
            if self._consumed != expected:
                missing = expected - self._consumed
                extra = self._consumed - expected
                raise JudgeTranscriptUsageError(
                    "Judge transcript was not consumed exactly; "
                    f"missing={sum(missing.values())},extra={sum(extra.values())}"
                )


class ReplayJudge:
    """Network-free Judge-compatible adapter backed by an immutable artifact."""

    def __init__(
        self,
        config: LLMJudgeConfig,
        *,
        artifact: JudgeTranscriptArtifact,
        replayer: JudgeTranscriptReplayer | None = None,
        scope: object,
        endpoint_url: object | None = None,
    ) -> None:
        scope_text = str(scope).strip()
        if not scope_text:
            raise JudgeTranscriptUsageError("replay scope cannot be empty")
        self.config = config
        self.artifact = artifact
        self.replayer = replayer or JudgeTranscriptReplayer(artifact)
        self.scope = scope_text
        self.endpoint_url = endpoint_url
        self.last_run_stats: LLMJudgeStats | None = None
        self.replayer._register(
            config,
            scope=scope_text,
            endpoint_url=endpoint_url,
        )

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        results = self.replayer._judge(
            self.config,
            scope=self.scope,
            endpoint_url=self.endpoint_url,
            items=items,
        )
        self.last_run_stats = _stats_for(self.config, len(items))
        return results

    def assert_consumed(self) -> None:
        self.replayer.assert_consumed()


__all__ = [
    "JudgeTranscriptArtifact",
    "JudgeTranscriptError",
    "JudgeTranscriptIntegrityError",
    "JudgeTranscriptRecorder",
    "JudgeTranscriptReplayer",
    "JudgeTranscriptUsageError",
    "RecordingJudge",
    "ReplayJudge",
    "TRANSCRIPT_SCHEMA_VERSION",
    "load_judge_transcript",
    "normalize_judge_text",
    "persist_judge_transcript_immutable",
    "sanitize_endpoint_identity",
]
