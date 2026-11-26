import unittest

from fastapi.testclient import TestClient

from src.api import Services, create_app
from src.attack_db.hash_db import HashAttackDB
from src.guards.stub import LookupGuard
from src.result import BLOCK, PASS, UNSURE, Result


class FakeTier1:
    """PASS for prompts starting with 'nice', BLOCK for 'evil', everything else goes to the guard."""

    def __init__(self):
        self.learned = []

    def check(self, text):
        if text.startswith("nice"):
            return Result(PASS, 0.01, "score")
        if text.startswith("evil"):
            return Result(BLOCK, 0.99, "score")
        return Result(UNSURE, 0.5, "score")

    def learn(self, text, is_attack):
        self.learned.append((text, is_attack))


LABELS = {"odd attack": True, "odd question": False}


def make_client(share=True, make_guard=None, tier1=None):
    services = Services(
        make_db=HashAttackDB,
        make_tier1=(lambda: tier1) if tier1 else FakeTier1,
        make_guard=make_guard or (lambda app: LookupGuard(LABELS)),
        share=share,
    )
    return TestClient(create_app(services)), services


def post(client, path, app, text):
    return client.post(path, json={"app": app, "text": text})


class CheckTests(unittest.TestCase):
    def test_tier1_decides_clear_prompts(self):
        with make_client()[0] as client:
            answer = post(client, "/check", "chatbot", "nice weather today").json()
            self.assertEqual((answer["verdict"], answer["decided_by"]), (PASS, "tier1"))
            answer = post(client, "/check", "chatbot", "evil prompt").json()
            self.assertEqual((answer["verdict"], answer["decided_by"]), (BLOCK, "tier1"))

    def test_guard_decides_unsure_prompts_then_memory_remembers(self):
        with make_client()[0] as client:
            first = post(client, "/check", "chatbot", "odd attack").json()
            self.assertEqual((first["verdict"], first["decided_by"]), (BLOCK, "guard"))
            second = post(client, "/check", "chatbot", "odd attack").json()
            self.assertEqual((second["verdict"], second["decided_by"]), (BLOCK, "memory"))

    def test_guard_pass_is_not_remembered(self):
        with make_client()[0] as client:
            post(client, "/check", "chatbot", "odd question")
            again = post(client, "/check", "chatbot", "odd question").json()
            self.assertEqual((again["verdict"], again["decided_by"]), (PASS, "guard"))

    def test_input_is_validated(self):
        with make_client()[0] as client:
            self.assertEqual(post(client, "/check", "chatbot", "").status_code, 422)
            self.assertEqual(client.post("/check", json={"text": "hello"}).status_code, 422)

    def test_a_broken_guard_fails_closed(self):
        with make_client(make_guard=lambda app: LookupGuard({}))[0] as client:
            answer = post(client, "/check", "chatbot", "never seen by the guard")
            self.assertEqual(answer.status_code, 200)
            self.assertEqual((answer.json()["verdict"], answer.json()["decided_by"]), (BLOCK, "error"))


class SharingTests(unittest.TestCase):
    def guard_calls(self, client, app):
        return client.get("/stats").json()[app].get("guard", 0)

    def test_sharing_on_second_app_profits(self):
        with make_client(share=True)[0] as client:
            post(client, "/check", "app_a", "odd attack")
            answer = post(client, "/check", "app_b", "odd attack").json()
            self.assertEqual(answer["decided_by"], "memory")
            self.assertEqual(self.guard_calls(client, "app_b"), 0)

    def test_sharing_off_second_app_asks_its_own_guard(self):
        with make_client(share=False)[0] as client:
            post(client, "/check", "app_a", "odd attack")
            answer = post(client, "/check", "app_b", "odd attack").json()
            self.assertEqual(answer["decided_by"], "guard")
            self.assertEqual(self.guard_calls(client, "app_b"), 1)


class FeedbackTests(unittest.TestCase):
    def test_feedback_stores_an_attack_for_all_apps(self):
        with make_client(share=True)[0] as client:
            stored = post(client, "/feedback", "app_a", "a missed attack").json()
            self.assertEqual(stored["status"], "stored")
            self.assertEqual(stored["attacks_stored"], 1)
            answer = post(client, "/check", "app_b", "a missed attack").json()
            self.assertEqual((answer["verdict"], answer["decided_by"]), (BLOCK, "memory"))

    def test_feedback_teaches_tier1_when_it_can_learn(self):
        tier1 = FakeTier1()
        with make_client(tier1=tier1)[0] as client:
            post(client, "/feedback", "app_a", "a missed attack")
        self.assertEqual(tier1.learned, [("a missed attack", True)])


class HealthAndStatsTests(unittest.TestCase):
    def test_health(self):
        with make_client()[0] as client:
            post(client, "/check", "app_a", "odd attack")
            health = client.get("/health").json()
            self.assertEqual(health, {"status": "ok", "share": True, "apps": 1, "attacks_stored": 1})

    def test_attacks_are_counted_once_per_database(self):
        with make_client(share=False)[0] as client:
            post(client, "/check", "app_a", "odd attack")
            post(client, "/check", "app_b", "odd attack")
            self.assertEqual(client.get("/health").json()["attacks_stored"], 2)

    def test_stats_count_by_step_and_verdict(self):
        with make_client()[0] as client:
            post(client, "/check", "app_a", "nice one")
            post(client, "/check", "app_a", "odd attack")
            post(client, "/check", "app_a", "odd attack")
            stats = client.get("/stats").json()["app_a"]
            self.assertEqual((stats["tier1"], stats["guard"], stats["memory"]), (1, 1, 1))
            self.assertEqual((stats["pass"], stats["block"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
