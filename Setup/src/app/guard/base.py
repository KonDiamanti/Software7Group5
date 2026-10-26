"""Interface every guard model must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Direction = Literal["input", "output"]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str = ""
    score: float | None = None  # optional risk score from the guard model


class Guard(ABC):
    """Decide whether a piece of text may pass."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check(self, text: str, direction: Direction) -> GuardResult:
        """Check user input (direction='input') or model output (direction='output')."""
