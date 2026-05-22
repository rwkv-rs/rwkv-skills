from __future__ import annotations

from typing import Protocol


class FunctionCallingSandbox(Protocol):
    def close(self) -> None:
        ...
