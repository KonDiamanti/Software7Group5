"""Placeholder guard: allows everything. Replace once a guard model is chosen."""

from __future__ import annotations

from .base import Direction, Guard, GuardResult


class AllowAllGuard(Guard):
    def check(self, text: str, direction: Direction) -> GuardResult:
        return GuardResult(allowed=True, reason="placeholder guard", score=0.0)
