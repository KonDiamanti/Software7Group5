
"""Service that checks known attacks, tier 1, and then the guard."""

from src.attack_db.base import AttackDB
from src.result import BLOCK, PASS, UNSURE, Result


class PromptService:
    """Route prompts through the shared attack database and detectors."""

    def __init__(self, attack_db: AttackDB, tier1, guard):
        self.attack_db = attack_db
        self.tier1 = tier1
        self.guard = guard

    def check(self, text: str) -> Result:
        """Return BLOCK, PASS, or the result of the guard."""

        # 1. Check for an attack already known to the shared database.
        source = self.attack_db.lookup(text)
        if source is not None:
            return Result(
                verdict=BLOCK,
                score=1.0,
                reason="known_attack",
            )

        # 2. Use the inexpensive tier-1 detector.
        tier1_result = self.tier1.check(text)

        if tier1_result.verdict == BLOCK:
            return tier1_result

        if tier1_result.verdict == PASS:
            return tier1_result

        if tier1_result.verdict != UNSURE:
            raise ValueError(
                f"Unknown tier-1 verdict: {tier1_result.verdict!r}"
            )

        # 3. Only uncertain prompts reach the guard.
        guard_result = self.guard.check(text)

        if isinstance(guard_result, Result):
            return guard_result

        # Support simple guards that return True/False.
        if guard_result is True:
            return Result(PASS, None, "guard_pass")

        if guard_result is False:
            return Result(BLOCK, None, "guard_block")

        raise TypeError(
            "Guard.check(text) must return a Result or a boolean"
        )
