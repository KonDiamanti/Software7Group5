
"""Route uncertain tier-1 results to the guard."""

from src.result import BLOCK, PASS, UNSURE, Result


class GuardRouter:
    """Call the guard only when tier 1 returns UNSURE."""

    def __init__(self, guard):
        self.guard = guard

    def route(self, text: str, tier1_result: Result) -> Result:
        """Return tier 1's decisive result or ask the guard."""

        if tier1_result.verdict in (BLOCK, PASS):
            return tier1_result

        if tier1_result.verdict != UNSURE:
            raise ValueError(
                f"Unknown tier-1 verdict: {tier1_result.verdict!r}"
            )

        guard_result = self.guard.check(text)

        if isinstance(guard_result, Result):
            return guard_result

        if guard_result is True:
            return Result(PASS, None, "guard_pass")

        if guard_result is False:
            return Result(BLOCK, None, "guard_block")

        raise TypeError(
            "Guard.check(text) must return a Result or a boolean"
        )
