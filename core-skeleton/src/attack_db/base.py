# base.py
# what every version of the shared attack database has to offer.
# hash version and embedding version are both still open, so the rest of the
# project only talks to this shape.

from typing import Optional, Protocol


class AttackDB(Protocol):
    def add(self, text: str, source: str = "") -> None:
        """store a prompt that is known to be an attack"""
        ...

    def lookup(self, text: str) -> Optional[str]:
        """return a short note about the match (like the source), or None if the prompt is not known"""
        ...

    def count(self) -> int:
        ...