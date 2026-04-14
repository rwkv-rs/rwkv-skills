from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import copy


class TokenizerProtocol(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str:  # pragma: no cover - protocol
        ...


@dataclass(slots=True, frozen=True)
class TokenConstraintCache:
    token_bytes: tuple[bytes, ...]
    candidate_token_ids: tuple[int, ...]
    tokens_by_first_byte: dict[int, tuple[int, ...]]


class DecodeConstraint(Protocol):
    def clone(self) -> "DecodeConstraint":
        ...

    def feed_text(self, text: str) -> bool:
        ...

    def allowed_first_bytes(self) -> set[int] | None:
        ...

    def is_complete(self) -> bool:
        ...

    def finish_reason(self) -> str:
        ...


@dataclass(slots=True)
class ConstraintRuntime:
    constraint: DecodeConstraint
    cache: TokenConstraintCache
    pending_utf8: bytes = b""

    def clone(self) -> "ConstraintRuntime":
        return ConstraintRuntime(
            constraint=self.constraint.clone(),
            cache=self.cache,
            pending_utf8=bytes(self.pending_utf8),
        )

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.constraint.is_complete():
            return ()
        first_bytes = None if self.pending_utf8 else self.constraint.allowed_first_bytes()
        candidate_ids = self.cache.candidate_token_ids
        if first_bytes:
            gathered: list[int] = []
            seen: set[int] = set()
            for byte in first_bytes:
                for token_id in self.cache.tokens_by_first_byte.get(int(byte) & 0xFF, ()):
                    if token_id not in seen:
                        seen.add(token_id)
                        gathered.append(token_id)
            candidate_ids = tuple(gathered)

        allowed: list[int] = []
        for token_id in candidate_ids:
            if self._accept_token_bytes(self.cache.token_bytes[token_id], dry_run=True):
                allowed.append(int(token_id))
        return tuple(allowed)

    def commit_token_bytes(self, token_bytes: bytes) -> bool:
        return self._accept_token_bytes(token_bytes, dry_run=False)

    def _accept_token_bytes(self, token_bytes: bytes, *, dry_run: bool) -> bool:
        if not token_bytes:
            return False
        target = self.clone() if dry_run else self
        split = split_valid_utf8_prefix(target.pending_utf8 + token_bytes)
        if split is None:
            return False
        decoded_text, tail = split
        if decoded_text and not target.constraint.feed_text(decoded_text):
            return False
        if target.constraint.is_complete() and tail:
            return False
        target.pending_utf8 = tail
        return True

    def is_complete(self) -> bool:
        return self.constraint.is_complete() and not self.pending_utf8

    def finish_reason(self) -> str:
        return self.constraint.finish_reason()


def clone_constraint(constraint: DecodeConstraint) -> DecodeConstraint:
    return copy.deepcopy(constraint)


def build_token_constraint_cache(
    tokenizer: TokenizerProtocol,
    *,
    vocab_size: int,
) -> TokenConstraintCache:
    token_bytes: list[bytes] = []
    candidate_ids: list[int] = []
    tokens_by_first_byte: dict[int, list[int]] = {}
    for token_id in range(max(0, int(vocab_size))):
        data = decode_token_bytes(tokenizer, token_id)
        token_bytes.append(data)
        if not data:
            continue
        candidate_ids.append(token_id)
        bucket = tokens_by_first_byte.setdefault(int(data[0]), [])
        bucket.append(token_id)
    return TokenConstraintCache(
        token_bytes=tuple(token_bytes),
        candidate_token_ids=tuple(candidate_ids),
        tokens_by_first_byte={key: tuple(value) for key, value in tokens_by_first_byte.items()},
    )


def decode_token_bytes(tokenizer: TokenizerProtocol, token_id: int) -> bytes:
    decode_bytes = getattr(tokenizer, "decodeBytes", None)
    if callable(decode_bytes):
        data = decode_bytes([int(token_id)])
        if isinstance(data, bytes):
            return data
    decode_bytes = getattr(tokenizer, "decode_bytes", None)
    if callable(decode_bytes):
        data = decode_bytes([int(token_id)])
        if isinstance(data, bytes):
            return data
    try:
        return tokenizer.decode([int(token_id)]).encode("utf-8")
    except Exception:
        return b""


def split_valid_utf8_prefix(data: bytes) -> tuple[str, bytes] | None:
    if not data:
        return "", b""
    try:
        return data.decode("utf-8"), b""
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(data):
            return None
        prefix_len = int(exc.start)
        prefix = data[:prefix_len]
        remainder = data[prefix_len:]
        try:
            return prefix.decode("utf-8"), remainder
        except UnicodeDecodeError:
            return None


__all__ = [
    "ConstraintRuntime",
    "DecodeConstraint",
    "TokenConstraintCache",
    "build_token_constraint_cache",
    "clone_constraint",
    "decode_token_bytes",
    "split_valid_utf8_prefix",
]
