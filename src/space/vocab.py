from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


_VOCAB_PATH = Path(__file__).resolve().parents[1] / "infer" / "rwkv7" / "rwkv_vocab_v20230424.txt"


@lru_cache(maxsize=1)
def load_rwkv_vocab() -> dict[int, bytes]:
    """Load RWKV vocab file into a mapping from token id -> raw bytes.

    The vocab file format matches what TRIE_TOKENIZER expects:
        <id> <python-literal> <len>
    where python-literal is a repr of either bytes or str.
    """
    vocab: dict[int, bytes] = {}
    if not _VOCAB_PATH.exists():
        return vocab

    for line in _VOCAB_PATH.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        first_space = line.find(" ")
        last_space = line.rfind(" ")
        if first_space <= 0 or last_space <= first_space:
            continue
        try:
            token_id = int(line[:first_space])
            literal = line[first_space + 1 : last_space].strip()
            expected_len = int(line[last_space + 1 :].strip())
        except ValueError:
            continue

        try:
            value = ast.literal_eval(literal)
        except Exception:  # noqa: BLE001
            continue

        if isinstance(value, str):
            raw = value.encode("utf-8")
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            continue

        if expected_len != len(raw):
            continue
        vocab[token_id] = raw

    return vocab


def _escape_control_chars(text: str) -> str:
    # Keep output single-line and readable in UI.
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 32 or code == 127:
            out.append(f"\\x{code:02x}")
        else:
            out.append(ch)
    return "".join(out)


def token_id_to_display(token_id: int) -> str:
    """Convert a token id into a human-readable text representation."""
    if token_id == 0:
        # TRIE_TOKENIZER injects this special token id.
        return "<|endoftext|>"

    vocab = load_rwkv_vocab()
    raw = vocab.get(int(token_id))
    if raw is None:
        return f"<unk:{token_id}>"

    decoded = raw.decode("utf-8", errors="backslashreplace")
    return _escape_control_chars(decoded)


__all__ = [
    "load_rwkv_vocab",
    "token_id_to_display",
]

