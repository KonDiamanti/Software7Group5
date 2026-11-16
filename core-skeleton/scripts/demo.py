"""Demo: three apps use the shared service through its API, with and without sharing.

Run from inside the core-skeleton folder (the model must be trained first, see src/tier1/train.py):

    python -m scripts.demo
    python -m scripts.demo --runs 10 --repeat 0.5
    python -m scripts.demo --apps 10            # more apps: the saving from sharing grows with the number of apps
    python -m scripts.demo --memory hash        # exact-match memory instead of the embedding memory
    python -m scripts.demo --learn              # tier 1 also learns from the guard (slow: it retrains)

Part 1 shows one attack travelling through the three apps. Part 2 sends the same traffic to the
three apps with and without sharing and counts the guard calls.

What is a stand-in here: the guard (always right, it looks the answer up) and the traffic
(prompts from the test sets of deepset and jackhhao, repeats made by us).
"""

import argparse
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import Services, create_app
from src.attack_db.hash_db import HashAttackDB
from src.guards.stub import LookupGuard
from src.result import UNSURE

APPS = ["chatbot", "support", "coding helper"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tier1_char_svm_more_data.joblib"
MAX_LENGTH = 20000   # longest prompt the API accepts


def app_names(count):
    """The three named apps first, then app 4, app 5 and so on."""
    if count <= len(APPS):
        return APPS[:count]
    return APPS + [f"app {i}" for i in range(len(APPS) + 1, count + 1)]


# ---- edited copies of an attack (the same kinds of edits as in the cascade simulation) ----
def add_filler(text, rng):
    before = ["Please help me with this. ", "Hi there! ", "Quick question first. "]
    after = [" Thanks a lot!", " Let me know when you are ready.", " This is important to me."]
    return rng.choice(before) + text + rng.choice(after)


def typos(text, rng):
    words = text.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) > 4]
    for i in rng.sample(long_words, min(3, len(long_words))):
        w = words[i]
        k = rng.randrange(1, len(w) - 2)
        words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)


def drop_words(text, rng):
    kept = [w for w in text.split(" ") if rng.random() > 0.10]
    return " ".join(kept) if kept else text


def edited(text, rng):
    result = rng.choice([add_filler, typos, drop_words])(text, rng)
    return result if result != text else add_filler(text, rng)


# ---- traffic ----
def make_traffic(attacks, harmless, rng, per_app=100, attack_share=0.20, repeat_rate=0.30, apps=APPS):
    """Prompts in the order the apps send them. A repeat is an attack that some app has sent before:
    a third exactly the same, two thirds edited. Returns (traffic, labels)."""
    unused = list(attacks)
    rng.shuffle(unused)
    history = []
    traffic, labels = [], {}
    for _ in range(per_app):
        for app in apps:
            if rng.random() < attack_share:
                if history and (rng.random() < repeat_rate or not unused):   # no new attacks left: repeat one
                    original = rng.choice(history)
                    if rng.random() < 1 / 3:
                        text, kind = original, "exact repeat"
                    else:
                        text, kind = edited(original, rng), "edited repeat"
                else:
                    text, kind = unused.pop(), "new attack"
                    history.append(text)
                is_attack = True
            else:
                text, kind, is_attack = rng.choice(harmless), "harmless", False
            labels[text] = is_attack
            traffic.append((app, text, is_attack, kind))
    return traffic, labels


# ---- one run: all traffic through the API ----
def run_condition(traffic, labels, make_tier1, make_db, share):
    services = Services(make_db=make_db, make_tier1=make_tier1,
                        make_guard=lambda app: LookupGuard(labels), share=share)
    steps = Counter()   # (kind, decided_by) -> how many prompts
    missed = false_blocks = 0
    with TestClient(create_app(services)) as client:
        for app, text, is_attack, kind in traffic:
            response = client.post("/check", json={"app": app, "text": text})
            response.raise_for_status()
            answer = response.json()
            steps[(kind, answer["decided_by"])] += 1
            missed += is_attack and answer["verdict"] == "pass"
            false_blocks += (not is_attack) and answer["verdict"] == "block"
        attacks_stored = client.get("/health").json()["attacks_stored"]
    attacks = sum(1 for t in traffic if t[2])
    return {
        "guard_calls": sum(n for (kind, by), n in steps.items() if by == "guard"),
        "prompts": len(traffic),
        "missed_pct": 100 * missed / max(attacks, 1),
        "false_block_pct": 100 * false_blocks / max(len(traffic) - attacks, 1),
        "steps": steps,
        "attacks_stored": attacks_stored,
    }


# ---- part 1: one attack through three apps ----
def find_unsure_attack(tier1, attacks):
    """An attack that tier 1 cannot decide alone, so the guard has to look at it."""
    for text in attacks:
        result = tier1.check(text)
        if result.verdict == UNSURE and result.reason == "score":
            return text
    return None


def story(tier1, make_db, attacks, rng):
    attack = find_unsure_attack(tier1, attacks)
    if attack is None:
        print("No attack found that tier 1 leaves to the guard, skipping part 1.")
        return
    steps = [("chatbot", attack, "the attack"),
             ("support", attack, "the same attack"),
             ("coding helper", edited(attack, rng), "the attack with small edits")]
    labels = {text: True for _, text, _ in steps}
    print("\nPART 1: one attack travels through three apps")
    print(f"the attack starts with: {attack[:90]!r}")
    for share in (True, False):
        services = Services(make_db=make_db, make_tier1=lambda: tier1,
                            make_guard=lambda app: LookupGuard(labels), share=share)
        print(f"\n  sharing {'ON' if share else 'OFF'}")
        with TestClient(create_app(services)) as client:
            for app, text, what in steps:
                answer = client.post("/check", json={"app": app, "text": text}).json()
                print(f"    {app:14} sends {what:30} -> {answer['verdict']:5} decided by {answer['decided_by']}")
            guard_calls = sum(s.get("guard", 0) for s in client.get("/stats").json().values())
        print(f"    guard calls: {guard_calls}")


# ---- part 2: the numbers ----
def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def numbers(attacks, harmless, make_tier1, make_db, args):
    apps = app_names(args.apps)
    print(f"\nPART 2: {len(apps)} apps, {args.per_app} prompts each, 20% attacks, "
          f"repeat rate {args.repeat:.0%}, {args.runs} runs")
    results = {True: [], False: []}
    for seed in range(args.runs):
        traffic, labels = make_traffic(attacks, harmless, random.Random(seed),
                                       per_app=args.per_app, repeat_rate=args.repeat, apps=apps)
        for share in (False, True):
            start = time.perf_counter()
            results[share].append(run_condition(traffic, labels, make_tier1, make_db, share))
            print(f"  run {seed + 1}/{args.runs}, sharing {'on ' if share else 'off'}: "
                  f"{results[share][-1]['guard_calls']} guard calls ({time.perf_counter() - start:.0f} s)")

    def per_1000(result):
        return 1000 * result["guard_calls"] / result["prompts"]

    print("\nGUARD CALLS per 1,000 prompts (mean +- standard deviation over the runs)")
    for share, name in ((False, "separate (no sharing)"), (True, "shared")):
        m, sd = mean_std([per_1000(r) for r in results[share]])
        print(f"  {name:22} {m:6.1f} +- {sd:5.1f}")
    saved = [per_1000(a) - per_1000(b) for a, b in zip(results[False], results[True])]
    m, sd = mean_std(saved)
    fewer = sum(1 for x in saved if x > 0)
    print(f"  saved by sharing, on the same traffic: {m:.1f} +- {sd:.1f}, fewer calls in {fewer} of {len(saved)} runs")

    print("\nMISTAKES (mean over the runs)")
    for share, name in ((False, "separate (no sharing)"), (True, "shared")):
        print(f"  {name:22} missed attacks {mean_std([r['missed_pct'] for r in results[share]])[0]:.1f}%"
              f"   harmless blocked {mean_std([r['false_block_pct'] for r in results[share]])[0]:.1f}%")

    print("\nWHO DECIDED THE REPEATED ATTACKS (all runs together)")
    for kind in ("exact repeat", "edited repeat"):
        for share, name in ((False, "separate"), (True, "shared")):
            total = Counter()
            for r in results[share]:
                for (k, by), n in r["steps"].items():
                    if k == kind:
                        total[by] += n
            n_all = sum(total.values()) or 1
            print(f"  {kind:14} {name:9} memory {100 * total['memory'] / n_all:4.0f}%   "
                  f"tier 1 {100 * total['tier1'] / n_all:4.0f}%   guard {100 * total['guard'] / n_all:4.0f}%")

    print("\nNote: the guard is a stand-in that is always right, the repeats in the traffic are made by us,"
          " and the sample is small, so read the differences as a direction.")


def load_prompts():
    """Attacks and harmless prompts from the test sets of deepset and jackhhao (tier 1 never trained on them)."""
    from src.tier1.train import load_test_sets
    attacks, harmless = [], []
    for texts, labels in load_test_sets().values():
        for text, label in zip(texts, labels):
            if text.strip() and len(text) <= MAX_LENGTH:
                (attacks if label == 1 else harmless).append(text)
    attacks = list(dict.fromkeys(attacks))
    harmless = [t for t in dict.fromkeys(harmless) if t not in set(attacks)]
    return attacks, harmless


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--apps", type=int, default=3, help="number of apps in part 2")
    parser.add_argument("--per-app", type=int, default=100, help="prompts per app")
    parser.add_argument("--repeat", type=float, default=0.30, help="share of the attacks that are repeats")
    parser.add_argument("--memory", choices=["embedding", "hash"], default="embedding")
    parser.add_argument("--learn", action="store_true", help="tier 1 learns from the guard (slow)")
    parser.add_argument("--retrain-every", type=int, default=50)
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    from src.tier1.char_svm import CharSvmTier1
    tier1 = CharSvmTier1.load(args.model)
    attacks, harmless = load_prompts()
    print(f"prompts for the traffic: {len(attacks)} attacks, {len(harmless)} harmless")

    if args.memory == "embedding":
        from src.attack_db.embedding_db import EmbeddingAttackDB, default_embedder
        embedder = default_embedder()
        make_db = lambda: EmbeddingAttackDB(embedder=embedder)
    else:
        make_db = HashAttackDB

    if args.learn:
        from src.tier1.learning import LearningTier1
        from src.tier1.train import prepare_data
        base_x, base_y = prepare_data(True)[:2]
        make_tier1 = lambda: LearningTier1(tier1, base_x, base_y, retrain_every=args.retrain_every)
    else:
        make_tier1 = lambda: tier1

    print(f"memory: {args.memory}, tier 1 learns from the guard: {args.learn}")
    story(tier1, make_db, attacks, random.Random(0))
    numbers(attacks, harmless, make_tier1, make_db, args)


if __name__ == "__main__":
    main()
