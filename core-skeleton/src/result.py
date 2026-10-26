# result.py
# the one shape every detector returns (tier 1, guard models, anything we add later)

from dataclasses import dataclass
from typing import Optional, Protocol

BLOCK = "block"
PASS = "pass"
UNSURE = "unsure"   # tier 1 uses this to say "send it to the guard"


@dataclass
class Result:
    verdict: str                 # "block", "pass" or "unsure"
    score: Optional[float]       # 0..1 if the detector has a score, otherwise None
    reason: str                  # short note: "score", "tamper", "non-english", "empty"


class Detector(Protocol):
    # anything with a check(text) -> Result method can be plugged in
    def check(self, text: str) -> Result:
        ...