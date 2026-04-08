from __future__ import annotations

"""State and prefix-cache primitives adapted for future lightning-style engines."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_PREFIX_CACHE_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _serialize_token_ids(tokens: tuple[int, ...]) -> str:
    return " ".join(str(token) for token in tokens)


def _hash_token_ids(tokens: tuple[int, ...]) -> str:
    payload = _serialize_token_ids(tokens).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _coerce_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


@dataclass(slots=True, frozen=True)
class StatePoolConfig:
    l1_capacity: int = 16
    l2_capacity: int = 64
    db_path: str = "rwkv_sessions.db"
    prefix_cache_buckets: tuple[int, ...] = DEFAULT_PREFIX_CACHE_BUCKETS
    prefix_bucket_capacity: int = 16

    def normalized_prefix_buckets(self) -> tuple[int, ...]:
        buckets = sorted({int(bucket) for bucket in self.prefix_cache_buckets if int(bucket) > 0})
        if not buckets:
            raise ValueError("prefix_cache_buckets must contain at least one positive integer")
        return tuple(buckets)


class _CompressedTrieNode:
    def __init__(self, label: tuple[int, ...] = ()) -> None:
        self.label = label
        self.children: dict[int, _CompressedTrieNode] = {}
        self.terminal_key: str | None = None


class _CompressedTrie:
    def __init__(self) -> None:
        self.root = _CompressedTrieNode()

    def clear(self) -> None:
        self.root = _CompressedTrieNode()

    def insert(self, tokens: tuple[int, ...], terminal_key: str) -> None:
        self._insert(self.root, tokens, terminal_key)

    def longest_prefix(self, tokens: tuple[int, ...]) -> tuple[str | None, int]:
        node = self.root
        idx = 0
        best_key = node.terminal_key
        best_len = 0 if best_key is not None else 0
        while idx < len(tokens):
            child = node.children.get(tokens[idx])
            if child is None:
                break
            label = child.label
            if tokens[idx : idx + len(label)] != label:
                break
            idx += len(label)
            node = child
            if node.terminal_key is not None:
                best_key = node.terminal_key
                best_len = idx
        return best_key, best_len

    def _insert(self, node: _CompressedTrieNode, tokens: tuple[int, ...], terminal_key: str) -> None:
        if not tokens:
            node.terminal_key = terminal_key
            return
        first = tokens[0]
        child = node.children.get(first)
        if child is None:
            new_child = _CompressedTrieNode(tokens)
            new_child.terminal_key = terminal_key
            node.children[first] = new_child
            return
        common = self._common_prefix_len(tokens, child.label)
        if common == len(child.label):
            self._insert(child, tokens[common:], terminal_key)
            return
        split_node = _CompressedTrieNode(child.label[:common])
        node.children[first] = split_node
        child.label = child.label[common:]
        split_node.children[child.label[0]] = child
        remaining = tokens[common:]
        if remaining:
            new_child = _CompressedTrieNode(remaining)
            new_child.terminal_key = terminal_key
            split_node.children[remaining[0]] = new_child
        else:
            split_node.terminal_key = terminal_key

    @staticmethod
    def _common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx


@dataclass(slots=True)
class PrefixCacheEntry:
    state_id: str
    bucket_len: int
    prefix_tokens: tuple[int, ...]
    state_cpu: list[torch.Tensor]
    logits_cpu: torch.Tensor | None
    last_updated: float


class StateCacheManager:
    def __init__(self, config: StatePoolConfig | None = None) -> None:
        self.config = config or StatePoolConfig()
        self._prefix_buckets = self.config.normalized_prefix_buckets()
        self._prefix_hash_columns = tuple(f"prefix_hash_{bucket}" for bucket in self._prefix_buckets)
        self._cache_lock = threading.RLock()
        self._db_lock = threading.Lock()
        self._closed = False

        self.l1_cache: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self.l2_cache: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self.prefix_l2_cache: dict[int, OrderedDict[str, PrefixCacheEntry]] = {
            bucket: OrderedDict() for bucket in self._prefix_buckets
        }
        self.prefix_entry_index: dict[str, PrefixCacheEntry] = {}
        self.prefix_trie = _CompressedTrie()

        self.db_path = Path(self.config.db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        self._init_db()

    def put_state(self, session_id: str, state: list[torch.Tensor]) -> None:
        self._ensure_open()
        if not session_id:
            return
        with self._cache_lock:
            self.l1_cache.pop(session_id, None)
            self.l2_cache.pop(session_id, None)
            self.l1_cache[session_id] = self._clone_state(state)
            if len(self.l1_cache) > int(self.config.l1_capacity):
                evicted_id, evicted_state = self.l1_cache.popitem(last=False)
                self.l2_cache[evicted_id] = self._clone_to_cpu_state(evicted_state)
            if len(self.l2_cache) > int(self.config.l2_capacity):
                evicted_id, evicted_state = self.l2_cache.popitem(last=False)
                self._persist_session(evicted_id, evicted_state)

    def get_state(
        self,
        session_id: str,
        *,
        device: str | torch.device = "cuda",
    ) -> list[torch.Tensor] | None:
        self._ensure_open()
        if not session_id:
            return None
        target_device = _coerce_device(device)
        with self._cache_lock:
            if session_id in self.l1_cache:
                self.l1_cache.move_to_end(session_id)
                return self._clone_to_device_state(self.l1_cache[session_id], target_device)
            if session_id in self.l2_cache:
                state_cpu = self.l2_cache.pop(session_id)
                promoted = self._clone_to_device_state(state_cpu, target_device)
                self.l1_cache[session_id] = self._clone_state(promoted)
                return self._clone_state(promoted)

        blob = self._load_session_blob(session_id)
        if blob is None:
            return None
        try:
            state_cpu = self._deserialize(blob)
        except Exception as exc:
            logger.warning("failed to deserialize session %s: %s", session_id, exc)
            return None
        state = self._clone_to_device_state(state_cpu, target_device)
        with self._cache_lock:
            self.l1_cache[session_id] = self._clone_state(state)
        return self._clone_state(state)

    def close_session(self, session_id: str) -> None:
        self._ensure_open()
        if not session_id:
            return
        with self._cache_lock:
            state = self.l1_cache.pop(session_id, None)
            if state is None:
                state = self.l2_cache.pop(session_id, None)
            if state is None:
                return
            self._persist_session(session_id, self._clone_to_cpu_state(state))

    def delete_state_from_any_level(self, session_id: str) -> bool:
        self._ensure_open()
        deleted = False
        with self._cache_lock:
            if self.l1_cache.pop(session_id, None) is not None:
                deleted = True
            if self.l2_cache.pop(session_id, None) is not None:
                deleted = True
        with self._db_lock:
            self.db_cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self.db_conn.commit()
            deleted = deleted or bool(self.db_cursor.rowcount)
        return deleted

    def put_prefix_state(
        self,
        prefix_tokens: list[int] | tuple[int, ...],
        state: list[torch.Tensor],
        logits: torch.Tensor | None = None,
    ) -> bool:
        self._ensure_open()
        token_tuple = tuple(int(token) for token in prefix_tokens)
        bucket_len = len(token_tuple)
        if bucket_len not in self._prefix_buckets:
            return False
        entry = PrefixCacheEntry(
            state_id=_serialize_token_ids(token_tuple),
            bucket_len=bucket_len,
            prefix_tokens=token_tuple,
            state_cpu=self._clone_to_cpu_state(state),
            logits_cpu=self._clone_optional_tensor(logits, torch.device("cpu")),
            last_updated=time.time(),
        )
        with self._cache_lock:
            self._store_prefix_entry_locked(entry)
        self._persist_prefix_entry(entry)
        return True

    def match_prefix_state(
        self,
        prompt_tokens: list[int] | tuple[int, ...],
        *,
        device: str | torch.device = "cuda",
    ) -> dict[str, object] | None:
        self._ensure_open()
        token_tuple = tuple(int(token) for token in prompt_tokens)
        if not token_tuple:
            return None
        target_device = _coerce_device(device)
        with self._cache_lock:
            state_id, matched_len = self.prefix_trie.longest_prefix(token_tuple)
            if state_id is not None:
                entry = self.prefix_entry_index.get(state_id)
                if entry is not None:
                    self.prefix_l2_cache[entry.bucket_len].move_to_end(state_id)
                    return self._build_prefix_match(entry, matched_len, "l2_ram", target_device)
        for bucket in reversed(self._prefix_buckets):
            if len(token_tuple) < bucket:
                continue
            entry = self._load_prefix_entry(token_tuple[:bucket], bucket)
            if entry is not None:
                with self._cache_lock:
                    self._store_prefix_entry_locked(entry)
                return self._build_prefix_match(entry, bucket, "disk", target_device)
        return None

    def clear_prefix_memory(self) -> None:
        self._ensure_open()
        with self._cache_lock:
            self.prefix_l2_cache = {bucket: OrderedDict() for bucket in self._prefix_buckets}
            self.prefix_entry_index.clear()
            self.prefix_trie.clear()

    def list_states_in_db(self) -> list[tuple[str, float]]:
        self._ensure_open()
        with self._db_lock:
            self.db_cursor.execute("SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC")
            rows = self.db_cursor.fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    def list_prefix_states_in_db(self) -> list[tuple[str, int, float]]:
        self._ensure_open()
        with self._db_lock:
            self.db_cursor.execute("SELECT state_id, bucket_len, last_updated FROM prefix_cache ORDER BY last_updated DESC")
            rows = self.db_cursor.fetchall()
        return [(str(row[0]), int(row[1]), float(row[2])) for row in rows]

    def list_all_states(self) -> dict[str, object]:
        self._ensure_open()
        with self._cache_lock:
            l1_states = list(self.l1_cache.keys())
            l2_states = list(self.l2_cache.keys())
            prefix_l2_counts = {str(bucket): len(cache) for bucket, cache in self.prefix_l2_cache.items()}
        db_states = self.list_states_in_db()
        prefix_db_states = self.list_prefix_states_in_db()
        return {
            "l1_cache": l1_states,
            "l2_cache": l2_states,
            "database": [row[0] for row in db_states],
            "total_count": len(l1_states) + len(l2_states) + len(db_states),
            "prefix_l2_counts": prefix_l2_counts,
            "prefix_database_count": len(prefix_db_states),
        }

    def flush_all(self) -> None:
        self._ensure_open()
        with self._cache_lock:
            session_items = list(self.l1_cache.items()) + list(self.l2_cache.items())
            self.l1_cache.clear()
            self.l2_cache.clear()
            prefix_entries = list(self.prefix_entry_index.values())
            self.clear_prefix_memory()
        for session_id, state in session_items:
            self._persist_session(session_id, self._clone_to_cpu_state(state))
        for entry in prefix_entries:
            self._persist_prefix_entry(entry)

    def close(self) -> None:
        if self._closed:
            return
        self.flush_all()
        with self._db_lock:
            self.db_conn.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("state cache manager is closed")

    def _init_db(self) -> None:
        with self._db_lock:
            self.db_cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_blob BLOB NOT NULL,
                    last_updated REAL NOT NULL
                )
                """
            )
            prefix_hash_sql = ", ".join(f"{column} TEXT" for column in self._prefix_hash_columns)
            self.db_cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS prefix_cache (
                    state_id TEXT PRIMARY KEY,
                    bucket_len INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    {prefix_hash_sql},
                    state_blob BLOB NOT NULL,
                    logits_blob BLOB,
                    last_updated REAL NOT NULL
                )
                """
            )
            for bucket in self._prefix_buckets:
                self.db_cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_prefix_cache_{bucket}
                    ON prefix_cache (bucket_len, prefix_hash_{bucket}, last_updated)
                    """
                )
            self.db_conn.commit()

    def _store_prefix_entry_locked(self, entry: PrefixCacheEntry) -> None:
        bucket_cache = self.prefix_l2_cache[entry.bucket_len]
        bucket_cache.pop(entry.state_id, None)
        bucket_cache[entry.state_id] = entry
        self.prefix_entry_index[entry.state_id] = entry
        if len(bucket_cache) > int(self.config.prefix_bucket_capacity):
            evicted_id, _evicted = bucket_cache.popitem(last=False)
            self.prefix_entry_index.pop(evicted_id, None)
        self._rebuild_prefix_trie_locked()

    def _rebuild_prefix_trie_locked(self) -> None:
        self.prefix_trie.clear()
        for entry in self.prefix_entry_index.values():
            self.prefix_trie.insert(entry.prefix_tokens, entry.state_id)

    def _persist_session(self, session_id: str, state_cpu: list[torch.Tensor]) -> None:
        blob = self._serialize(state_cpu)
        with self._db_lock:
            self.db_cursor.execute(
                "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                (session_id, blob, time.time()),
            )
            self.db_conn.commit()

    def _persist_prefix_entry(self, entry: PrefixCacheEntry) -> None:
        row: list[object] = [
            entry.state_id,
            entry.bucket_len,
            entry.bucket_len,
        ]
        prefix_hashes = {bucket: _hash_token_ids(entry.prefix_tokens[:bucket]) for bucket in self._prefix_buckets}
        row.extend(prefix_hashes[bucket] if entry.bucket_len >= bucket else None for bucket in self._prefix_buckets)
        row.append(self._serialize(entry.state_cpu))
        row.append(self._serialize(entry.logits_cpu) if entry.logits_cpu is not None else None)
        row.append(entry.last_updated)
        placeholders = ", ".join("?" for _ in row)
        columns = ", ".join(
            ["state_id", "bucket_len", "token_count", *self._prefix_hash_columns, "state_blob", "logits_blob", "last_updated"]
        )
        with self._db_lock:
            self.db_cursor.execute(
                f"INSERT OR REPLACE INTO prefix_cache ({columns}) VALUES ({placeholders})",
                row,
            )
            self.db_conn.commit()

    def _load_session_blob(self, session_id: str) -> bytes | None:
        with self._db_lock:
            self.db_cursor.execute("SELECT state_blob FROM sessions WHERE session_id = ?", (session_id,))
            row = self.db_cursor.fetchone()
        if row is None:
            return None
        return bytes(row[0])

    def _load_prefix_entry(self, prefix_tokens: tuple[int, ...], bucket_len: int) -> PrefixCacheEntry | None:
        state_id = _serialize_token_ids(prefix_tokens)
        hash_column = f"prefix_hash_{bucket_len}"
        hash_value = _hash_token_ids(prefix_tokens)
        with self._db_lock:
            self.db_cursor.execute(
                f"""
                SELECT state_blob, logits_blob, last_updated
                FROM prefix_cache
                WHERE state_id = ? AND bucket_len = ? AND {hash_column} = ?
                LIMIT 1
                """,
                (state_id, bucket_len, hash_value),
            )
            row = self.db_cursor.fetchone()
        if row is None:
            return None
        try:
            state_cpu = self._deserialize(row[0])
            logits_cpu = self._deserialize(row[1]) if row[1] is not None else None
        except Exception as exc:
            logger.warning("failed to deserialize prefix entry %s: %s", state_id, exc)
            return None
        return PrefixCacheEntry(
            state_id=state_id,
            bucket_len=bucket_len,
            prefix_tokens=prefix_tokens,
            state_cpu=state_cpu,
            logits_cpu=logits_cpu,
            last_updated=float(row[2]) if row[2] is not None else time.time(),
        )

    def _build_prefix_match(
        self,
        entry: PrefixCacheEntry,
        matched_len: int,
        cache_source: str,
        device: torch.device,
    ) -> dict[str, object]:
        return {
            "state_id": entry.state_id,
            "matched_tokens": matched_len,
            "bucket_len": entry.bucket_len,
            "state": self._clone_to_device_state(entry.state_cpu, device),
            "logits": self._clone_optional_tensor(entry.logits_cpu, device),
            "cache_source": cache_source,
        }

    @staticmethod
    def _serialize(value: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(blob: bytes) -> object:
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location="cpu")

    @staticmethod
    def _clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.detach().clone() for tensor in state]

    @staticmethod
    def _clone_to_cpu_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.detach().to("cpu").clone() for tensor in state]

    @staticmethod
    def _clone_to_device_state(state: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
        non_blocking = device.type == "cuda"
        return [tensor.detach().to(device, non_blocking=non_blocking).clone() for tensor in state]

    @staticmethod
    def _clone_optional_tensor(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().to(device, non_blocking=device.type == "cuda").clone()


__all__ = [
    "DEFAULT_PREFIX_CACHE_BUCKETS",
    "PrefixCacheEntry",
    "StateCacheManager",
    "StatePoolConfig",
]
