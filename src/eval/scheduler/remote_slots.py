from __future__ import annotations

"""Remote inference slot parsing.

The scheduler treats remote model slots as launch resources.  A slot may point
at the same backend model as another slot, allowing multiple benchmark jobs to
feed one batching server while the DB identity stays the real model name.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from .dataset_utils import safe_slug


INFER_WORKER_PROFILE_CHOICES = ("fixed", "param-size")


@dataclass(frozen=True, slots=True)
class RemoteModelSlot:
    slot: str
    model: str

    @property
    def slot_slug(self) -> str:
        return safe_slug(self.slot)

    @property
    def model_slug(self) -> str:
        return safe_slug(self.model)


def parse_remote_model_slot(raw: str) -> RemoteModelSlot | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if "=" not in text:
        return RemoteModelSlot(slot=text, model=text)
    slot, model = (part.strip() for part in text.split("=", 1))
    if not slot or not model:
        raise ValueError(f"invalid infer model slot spec: {raw!r}")
    return RemoteModelSlot(slot=slot, model=model)


def parse_remote_model_slots(raw_slots: Sequence[str]) -> tuple[RemoteModelSlot, ...]:
    slots: list[RemoteModelSlot] = []
    seen: set[str] = set()
    for raw in raw_slots:
        slot = parse_remote_model_slot(raw)
        if slot is None:
            continue
        if slot.slot_slug in seen:
            raise ValueError(f"duplicate remote inference slot: {slot.slot!r}")
        seen.add(slot.slot_slug)
        slots.append(slot)
    return tuple(slots)


def remote_slot_map(raw_slots: Sequence[str]) -> dict[str, RemoteModelSlot]:
    return {slot.slot_slug: slot for slot in parse_remote_model_slots(raw_slots)}


def unique_remote_models(slots: Iterable[RemoteModelSlot]) -> tuple[str, ...]:
    models: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        model_slug = slot.model_slug
        if model_slug in seen:
            continue
        seen.add(model_slug)
        models.append(slot.model)
    return tuple(models)


_PARAM_RE = re.compile(r"(?P<params>\d+(?:[._]\d+)?)b", re.IGNORECASE)


def infer_workers_for_model(
    model_name: str,
    *,
    default_workers: int,
    profile: str = "fixed",
) -> int:
    workers = max(1, int(default_workers))
    if profile == "fixed":
        return workers
    if profile != "param-size":
        raise ValueError(f"unknown infer worker profile: {profile!r}")
    params = _extract_params_b(model_name)
    if params is None:
        return workers
    if params <= 1.6:
        return 256
    if params <= 3.1:
        return 128
    if params >= 13.0:
        return 48
    if params >= 7.0:
        return 96
    return workers


def _extract_params_b(model_name: str) -> float | None:
    match = _PARAM_RE.search(str(model_name or ""))
    if match is None:
        return None
    try:
        return float(match.group("params").replace("_", "."))
    except ValueError:
        return None


__all__ = [
    "INFER_WORKER_PROFILE_CHOICES",
    "RemoteModelSlot",
    "infer_workers_for_model",
    "parse_remote_model_slot",
    "parse_remote_model_slots",
    "remote_slot_map",
    "unique_remote_models",
]
