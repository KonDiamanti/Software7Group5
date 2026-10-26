
import unittest

from src.result import BLOCK, PASS, UNSURE, Result
from src.router import GuardRouter


class FakeGuard:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = 0

    def check(self, text):
        self.calls += 1
        return self.allowed


class GuardRouterTests(unittest.TestCase):
    def test_block_result_does_not_call_guard(self):
        guard = FakeGuard()
        router = GuardRouter(guard)
        tier1 = Result(BLOCK, 0.99, "attack")

        result = router.route("prompt", tier1)

        self.assertEqual(result, tier1)
        self.assertEqual(guard.calls, 0)

    def test_pass_result_does_not_call_guard(self):
        guard = FakeGuard()
        router = GuardRouter(guard)
        tier1 = Result(PASS, 0.01, "clean")

        result = router.route("prompt", tier1)

        self.assertEqual(result, tier1)
        self.assertEqual(guard.calls, 0)

    def test_unsure_calls_guard_and_blocks(self):
        guard = FakeGuard(allowed=False)
        router = GuardRouter(guard)

        result = router.route(
            "prompt",
            Result(UNSURE, None, "uncertain"),
        )

        self.assertEqual(result.verdict, BLOCK)
        self.assertEqual(result.reason, "guard_block")
        self.assertEqual(guard.calls, 1)

    def test_unsure_calls_guard_and_passes(self):
        guard = FakeGuard(allowed=True)
        router = GuardRouter(guard)

        result = router.route(
            "prompt",
            Result(UNSURE, None, "uncertain"),
        )

        self.assertEqual(result.verdict, PASS)
        self.assertEqual(result.reason, "guard_pass")
        self.assertEqual(guard.calls, 1)


if __name__ == "__main__":
    unittest.main()
