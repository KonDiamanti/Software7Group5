"""More tests for the prototype service: a bigger test (more runs, more prompts) and wrong reports.

Run from inside the core-skeleton folder, after tier 1 is trained (see src/tier1/train.py).
Three tests, each one is a sub-command:

  scale    The saving of sharing with more runs and (optionally) more prompts in the pool.
           Gives the mean, the standard deviation and a 95% confidence interval of the saving, so we can
           say if the saving is bigger than the noise.
      python -m scripts.scale_and_reports scale --apps 10 --repeat 0.7 --runs 20
      python -m scripts.scale_and_reports scale --guard prompt-guard-2 --apps 10 --repeat 0.7 --runs 20 --extra-pool 500

  reports  Correct reports at different rates and wrong (fake) reports of harmless prompts, with and
           without sharing. Shows how much a wrong report harms the apps.
      python -m scripts.scale_and_reports reports --guard prompt-guard-2 --apps 10 --repeat 0.7 --runs 5

  quarantine  The same report test, with three setups on the same traffic: separate, shared, and shared with the
           quarantine (a report counts only for the app that made it, until another app reports the same or a
           close prompt, or the guard blocks it). Shows what the quarantine keeps and what it costs.
      python -m scripts.scale_and_reports quarantine --guard prompt-guard-2 --apps 10 --repeat 0.7 --runs 5

  entry    One wrong entry in the database on its own: how many other harmless prompts does it block?
           Needs no guard and no tier 1.
      python -m scripts.scale_and_reports entry

What is a stand-in here: the traffic (repeats made by us), and, by default, the guard (always right).
Wrong reports are made by us too: harmless prompts sent to /feedback as if they were attacks.
"""

import argparse
import csv
import math
import os
import random
import statistics
import time
from collections import Counter

import numpy as np
from fastapi.testclient import TestClient

from scripts.demo import MAX_LENGTH, MODEL_PATH, app_names, load_prompts, make_traffic, mean_std
from src.api import Services, create_app
from src.attack_db.hash_db import HashAttackDB
from src.guards.stub import LookupGuard

EXTRA_DATASET = "Necent/llm-jailbreak-prompt-injection-dataset"   # the dataset of scripts/network_simulation.py


# ------------------------------------------------------------------ one run
def run_once(traffic, labels, make_tier1, make_db, share, make_guard=None,
             report_rate=0.0, fake_texts=(), seed=0):
    """All traffic through the API once.

    report_rate: share of the missed attacks that the apps report (correctly) through /feedback.
    fake_texts:  harmless prompts that are reported as attacks BEFORE the traffic starts, one app each
                 (the worst case: the wrong reports are there from the start).
    """
    make_guard = make_guard or (lambda app: LookupGuard(labels))
    services = Services(make_db=make_db, make_tier1=make_tier1, make_guard=make_guard, share=share)
    apps = list(dict.fromkeys(t[0] for t in traffic))
    steps = Counter()
    missed = false_blocks = reports = 0
    with TestClient(create_app(services)) as client:
        for i, text in enumerate(fake_texts):
            app = apps[i % len(apps)]
            client.post("/feedback", json={"app": app, "text": text}).raise_for_status()
        for index, (app, text, is_attack, kind) in enumerate(traffic):
            answer = client.post("/check", json={"app": app, "text": text}).json()
            steps[(kind, answer["decided_by"])] += 1
            passed = answer["verdict"] == "pass"
            missed += is_attack and passed
            if is_attack and passed and random.Random(f"feedback-{seed}-{index}").random() < report_rate:
                client.post("/feedback", json={"app": app, "text": text}).raise_for_status()
                reports += 1
            false_blocks += (not is_attack) and answer["verdict"] == "block"
        stored = client.get("/health").json()["attacks_stored"]
    attacks = sum(1 for t in traffic if t[2])
    harmless = len(traffic) - attacks
    return {
        "prompts": len(traffic),
        "guard_calls": sum(n for (kind, by), n in steps.items() if by == "guard"),
        "missed_pct": 100 * missed / max(attacks, 1),
        "false_blocks": false_blocks,
        "false_block_pct": 100 * false_blocks / max(harmless, 1),
        "by_memory_pct": 100 * steps[("harmless", "memory")] / max(harmless, 1),   # harmless blocked by the database
        "reports": reports,
        "attacks_stored": stored,
    }


def per_1000(result):
    return 1000 * result["guard_calls"] / result["prompts"]


def t_value(n):
    """Two-sided 95% value of the t distribution for n runs."""
    from scipy.stats import t
    return float(t.ppf(0.975, n - 1))


# ------------------------------------------------------------------ the bigger pool (test scale)
def drop_near_copies(candidates, reference_texts, threshold=0.9):
    """Keep only the candidates that are not a near copy (word TF-IDF cosine >= threshold) of a reference text.
    The same rule that removed near copies of test prompts from the training data of tier 1."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not candidates or not reference_texts:
        return list(candidates)
    vectorizer = TfidfVectorizer().fit(list(reference_texts) + list(candidates))
    reference = vectorizer.transform(reference_texts)
    keep = []
    for start in range(0, len(candidates), 500):
        chunk = candidates[start:start + 500]
        best = (vectorizer.transform(chunk) @ reference.T).max(axis=1).toarray().ravel()
        keep += [text for text, b in zip(chunk, best) if b < threshold]
    return keep


def load_extra_pool(count, avoid, train_texts, seed=0):
    """`count` more attacks and `count` more harmless prompts from a second dataset.
    Removed: prompts already in the pool, and near copies of anything tier 1 was trained on (so tier 1 has not seen them)."""
    from datasets import load_dataset
    rows = load_dataset(EXTRA_DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
    attacks, harmless, seen = [], [], set(avoid)
    scanned = 0
    for row in rows:
        scanned += 1
        text = row["prompt"]
        if not text or not text.strip() or len(text) > MAX_LENGTH or text in seen:
            continue
        seen.add(text)
        (attacks if row["prompt_adversarial"] == 1 else harmless).append(text)
        if (len(attacks) >= 3 * count and len(harmless) >= 3 * count) or scanned >= 60000:
            break   # we take more than we need, because the near copy filter removes some
    attacks = drop_near_copies(attacks, train_texts)[:count]
    harmless = drop_near_copies(harmless, train_texts)[:count]
    print(f"extra pool: {len(attacks)} attacks and {len(harmless)} harmless prompts "
          f"(scanned {scanned} rows, near copies of tier 1 training prompts removed)")
    return attacks, harmless


# ------------------------------------------------------------------ helpers shared by the tests
def cached_embedder(embedder):
    """The same text is embedded many times (every condition sends the same traffic), so remember it."""
    memory = {}

    def embed(texts):
        missing = [t for t in dict.fromkeys(texts) if t not in memory]
        if missing:
            for text, vec in zip(missing, embedder(missing)):
                memory[text] = vec
        return np.stack([memory[t] for t in texts])
    return embed


def make_parts(args):
    from src.tier1.char_svm import CharSvmTier1
    tier1 = CharSvmTier1.load(args.model)
    if args.memory == "embedding":
        from src.attack_db.embedding_db import EmbeddingAttackDB, default_embedder
        embedder = cached_embedder(default_embedder())
        make_db = lambda: EmbeddingAttackDB(embedder=embedder)
        min_apps = getattr(args, "min_apps", 2)
        make_db_quarantine = lambda: EmbeddingAttackDB(embedder=embedder, quarantine=True, min_reporters=min_apps)
    else:
        make_db = HashAttackDB
        make_db_quarantine = None   # the hash version has no quarantine
    guard = None
    if args.guard != "stub":
        from src.guards.hf_guard import MODELS, HFGuard
        guard = HFGuard(MODELS[args.guard], device=args.device)   # one model, answers are remembered per text
    return tier1, make_db, ((lambda app: guard) if guard else None), guard, make_db_quarantine


def get_pool(args):
    attacks, harmless = load_prompts()
    if args.extra_pool:
        from src.tier1.train import prepare_data
        train_texts = list(prepare_data(True)[0])
        more_a, more_h = load_extra_pool(args.extra_pool, set(attacks) | set(harmless), train_texts)
        attacks, harmless = attacks + more_a, harmless + more_h
    print(f"pool for the traffic: {len(attacks)} attacks, {len(harmless)} harmless prompts")
    return attacks, harmless


def add_common(p):
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--apps", type=int, default=10)
    p.add_argument("--per-app", type=int, default=100)
    p.add_argument("--repeat", type=float, default=0.70, help="share of the attacks that are repeats")
    p.add_argument("--memory", choices=["embedding", "hash"], default="embedding")
    p.add_argument("--guard", choices=["stub", "prompt-guard-2", "protectai"], default="stub")
    p.add_argument("--device", default="cpu", help="cpu, or cuda for the guard model")
    p.add_argument("--extra-pool", type=int, default=0,
                   help="add this many attacks and this many harmless prompts from a second dataset")
    p.add_argument("--start-seed", type=int, default=0, help="first seed (to continue an interrupted run)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--csv", default=None, help="append one row per run to this file")


def append_csv(path, header, row):
    if not path:
        return
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# ------------------------------------------------------------------ test scale
def scale(args):
    tier1, make_db, make_guard, guard, _ = make_parts(args)
    attacks, harmless = get_pool(args)
    apps = app_names(args.apps)
    print(f"\nSCALE TEST: {len(apps)} apps, {args.per_app} prompts each, 20% attacks, repeat rate {args.repeat:.0%}, "
          f"{args.runs} runs, guard {args.guard}, memory {args.memory}, tier 1 fixed")
    separate, shared = [], []
    for seed in range(args.start_seed, args.start_seed + args.runs):
        traffic, labels = make_traffic(attacks, harmless, random.Random(seed), per_app=args.per_app,
                                       repeat_rate=args.repeat, apps=apps)
        start = time.perf_counter()
        a = run_once(traffic, labels, lambda: tier1, make_db, False, make_guard, seed=seed)
        b = run_once(traffic, labels, lambda: tier1, make_db, True, make_guard, seed=seed)
        separate.append(per_1000(a))
        shared.append(per_1000(b))
        print(f"  run seed {seed}: separate {separate[-1]:.1f}, shared {shared[-1]:.1f}, saved {separate[-1] - shared[-1]:+.1f} "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
        append_csv(args.csv, ["seed", "separate", "shared", "missed_sep", "missed_shared", "fb_sep", "fb_shared"],
                   [seed, f"{separate[-1]:.2f}", f"{shared[-1]:.2f}", f"{a['missed_pct']:.2f}", f"{b['missed_pct']:.2f}",
                    f"{a['false_block_pct']:.2f}", f"{b['false_block_pct']:.2f}"])

    saved = [x - y for x, y in zip(separate, shared)]
    n = len(saved)
    m, sd = mean_std(saved)
    print("\nGUARD CALLS per 1,000 prompts (mean +- standard deviation over the runs)")
    print(f"  separate (no sharing)  {mean_std(separate)[0]:6.1f} +- {mean_std(separate)[1]:5.1f}")
    print(f"  shared                 {mean_std(shared)[0]:6.1f} +- {mean_std(shared)[1]:5.1f}")
    print(f"  saved by sharing, same traffic: {m:.1f} +- {sd:.1f} ({100 * m / mean_std(separate)[0]:.1f}% of the separate calls)")
    if n > 1:
        se = sd / math.sqrt(n)
        half = t_value(n) * se
        print(f"  standard error {se:.2f}, 95% interval for the mean saving: {m - half:.1f} to {m + half:.1f}")
        print(f"  saving is {'bigger than' if m - half > 0 else 'NOT clearly bigger than'} zero; "
              f"standard deviation is {'smaller' if sd < m else 'NOT smaller'} than the saving")
    print(f"  fewer calls with sharing in {sum(1 for x in saved if x > 0)} of {n} runs")
    if n >= 10:
        print("  the mean saving as the number of runs grows:")
        for k in range(5, n + 1, 5):
            part = saved[:k]
            print(f"    after {k:2d} runs: {statistics.mean(part):5.1f} +- {statistics.stdev(part) / math.sqrt(k):4.1f} (standard error)")
    if guard is not None:
        print(f"\nthe guard model really ran {guard.model_runs} times (the answer for a text is remembered)")


# ------------------------------------------------------------------ test reports
def condition_list(args):
    """(name, share of missed attacks reported correctly, number of wrong reports)"""
    conditions = [("no reports", 0.0, 0)]
    conditions += [(f"correct reports {int(r * 100)}%", r, 0) for r in args.rates if r > 0]
    conditions += [(f"{k} wrong reports", 0.0, k) for k in args.fakes if k > 0]
    if args.rates and args.fakes and max(args.fakes) > 0:
        conditions.append((f"correct {int(max(args.rates) * 100)}% + {max(args.fakes)} wrong", max(args.rates), max(args.fakes)))
    return conditions


def reports(args):
    tier1, make_db, make_guard, guard, _ = make_parts(args)
    attacks, harmless = get_pool(args)
    apps = app_names(args.apps)
    conditions = condition_list(args)
    print(f"\nREPORTS TEST: {len(apps)} apps, {args.per_app} prompts each, 20% attacks, repeat rate {args.repeat:.0%}, "
          f"{args.runs} runs, guard {args.guard}, memory {args.memory}, tier 1 fixed")
    print("a wrong report = a harmless prompt sent to /feedback as if it were an attack, before the traffic starts (worst case)")
    results = {(c[0], share): [] for c in conditions for share in (False, True)}
    for seed in range(args.start_seed, args.start_seed + args.runs):
        traffic, labels = make_traffic(attacks, harmless, random.Random(seed), per_app=args.per_app,
                                       repeat_rate=args.repeat, apps=apps)
        fake_pool = random.Random(f"fake-{seed}").sample(harmless, min(max([c[2] for c in conditions]), len(harmless)))
        start = time.perf_counter()
        for name, rate, k in conditions:
            for share in (False, True):
                r = run_once(traffic, labels, lambda: tier1, make_db, share, make_guard,
                             report_rate=rate, fake_texts=fake_pool[:k], seed=seed)
                results[(name, share)].append(r)
                append_csv(args.csv, ["seed", "condition", "share", "guard_per_1000", "missed_pct", "false_blocks",
                                      "false_block_pct", "by_memory_pct", "reports", "stored"],
                           [seed, name, share, f"{per_1000(r):.2f}", f"{r['missed_pct']:.2f}", r["false_blocks"],
                            f"{r['false_block_pct']:.2f}", f"{r['by_memory_pct']:.2f}", r["reports"], r["attacks_stored"]])
        print(f"  run seed {seed} done ({time.perf_counter() - start:.0f} s)", flush=True)

    base = {share: results[("no reports", share)] for share in (False, True)}
    print("\nPER CONDITION (mean over the runs)")
    print(f"{'condition':28} {'sharing':8} {'guard calls':>11} {'missed %':>9} {'harmless blocked %':>19} {'(by database)':>14}")
    for name, rate, k in conditions:
        for share in (False, True):
            rs = results[(name, share)]
            print(f"{name:28} {'shared' if share else 'separate':8} {mean_std([per_1000(r) for r in rs])[0]:11.1f} "
                  f"{mean_std([r['missed_pct'] for r in rs])[0]:9.2f} {mean_std([r['false_block_pct'] for r in rs])[0]:19.2f} "
                  f"{mean_std([r['by_memory_pct'] for r in rs])[0]:14.2f}")

    print("\nSAVED BY SHARING per condition (separate minus shared guard calls per 1,000, same traffic)")
    for name, rate, k in conditions:
        saved = [per_1000(a) - per_1000(b) for a, b in zip(results[(name, False)], results[(name, True)])]
        m, sd = mean_std(saved)
        print(f"  {name:28} {m:6.1f} +- {sd:4.1f}, fewer in {sum(1 for x in saved if x > 0)} of {len(saved)} runs")

    print("\nHARM OF THE WRONG REPORTS (against 'no reports' on the same traffic)")
    for name, rate, k in conditions:
        if k == 0:
            continue
        for share in (False, True):
            extra_pct = [a["false_block_pct"] - b["false_block_pct"] for a, b in zip(results[(name, share)], base[share])]
            extra_n = [(a["false_blocks"] - b["false_blocks"]) / k for a, b in zip(results[(name, share)], base[share])]
            print(f"  {name:28} {'shared' if share else 'separate':8} harmless blocked +{mean_std(extra_pct)[0]:.2f} points "
                  f"(+- {mean_std(extra_pct)[1]:.2f}), {mean_std(extra_n)[0]:.1f} more harmless prompts blocked per wrong report")
    if guard is not None:
        print(f"\nthe guard model really ran {guard.model_runs} times (the answer for a text is remembered)")


# ------------------------------------------------------------------ test quarantine
def quarantine(args):
    tier1, make_db, make_guard, guard, make_db_quarantine = make_parts(args)
    if make_db_quarantine is None:
        raise SystemExit("the quarantine needs the embedding database (use --memory embedding)")
    attacks, harmless = get_pool(args)
    apps = app_names(args.apps)
    conditions = condition_list(args)
    setups = [("separate", False, make_db), ("shared", True, make_db),
              (f"shared + quarantine ({args.min_apps} apps)", True, make_db_quarantine)]
    print(f"\nQUARANTINE TEST: {len(apps)} apps, {args.per_app} prompts each, 20% attacks, repeat rate {args.repeat:.0%}, "
          f"{args.runs} runs, guard {args.guard}, memory {args.memory}, tier 1 fixed")
    print("a wrong report = a harmless prompt sent to /feedback as if it were an attack, before the traffic starts (worst case)")
    print(f"quarantine: a report counts only for the app that made it, until {args.min_apps} different apps report the same "
          "or a close prompt, or the guard blocks it")
    results = {(c[0], name): [] for c in conditions for name, _, _ in setups}
    for seed in range(args.start_seed, args.start_seed + args.runs):
        traffic, labels = make_traffic(attacks, harmless, random.Random(seed), per_app=args.per_app,
                                       repeat_rate=args.repeat, apps=apps)
        fake_pool = random.Random(f"fake-{seed}").sample(harmless, min(max([c[2] for c in conditions]), len(harmless)))
        start = time.perf_counter()
        for cname, rate, k in conditions:
            for sname, share, factory in setups:
                r = run_once(traffic, labels, lambda: tier1, factory, share, make_guard,
                             report_rate=rate, fake_texts=fake_pool[:k], seed=seed)
                results[(cname, sname)].append(r)
                append_csv(args.csv, ["seed", "condition", "setup", "guard_per_1000", "missed_pct", "false_blocks",
                                      "false_block_pct", "by_memory_pct", "reports", "stored"],
                           [seed, cname, sname, f"{per_1000(r):.2f}", f"{r['missed_pct']:.2f}", r["false_blocks"],
                            f"{r['false_block_pct']:.2f}", f"{r['by_memory_pct']:.2f}", r["reports"], r["attacks_stored"]])
        print(f"  run seed {seed} done ({time.perf_counter() - start:.0f} s)", flush=True)

    print("\nPER CONDITION (mean over the runs)")
    print(f"{'condition':28} {'setup':28} {'guard calls':>11} {'missed %':>9} {'harmless blocked %':>19}")
    for cname, rate, k in conditions:
        for sname, share, factory in setups:
            rs = results[(cname, sname)]
            print(f"{cname:28} {sname:28} {mean_std([per_1000(r) for r in rs])[0]:11.1f} "
                  f"{mean_std([r['missed_pct'] for r in rs])[0]:9.2f} {mean_std([r['false_block_pct'] for r in rs])[0]:19.2f}")

    shared_name, quarantine_name = setups[1][0], setups[2][0]
    print("\nWHAT THE QUARANTINE CHANGES (shared + quarantine minus shared, same traffic, mean +- standard deviation)")
    print(f"{'condition':28} {'guard calls':>16} {'missed (points)':>18} {'harmless blocked (points)':>27}")
    for cname, rate, k in conditions:
        a, b = results[(cname, shared_name)], results[(cname, quarantine_name)]
        calls = mean_std([per_1000(y) - per_1000(x) for x, y in zip(a, b)])
        missed = mean_std([y["missed_pct"] - x["missed_pct"] for x, y in zip(a, b)])
        blocked = mean_std([y["false_block_pct"] - x["false_block_pct"] for x, y in zip(a, b)])
        print(f"{cname:28} {calls[0]:+9.1f} +- {calls[1]:4.1f} {missed[0]:+11.2f} +- {missed[1]:4.2f} {blocked[0]:+18.2f} +- {blocked[1]:4.2f}")
    print("\nnegative harmless blocked = the quarantine blocks fewer harmless prompts; positive missed = it misses more attacks")
    if guard is not None:
        print(f"\nthe guard model really ran {guard.model_runs} times (the answer for a text is remembered)")


# ------------------------------------------------------------------ test entry
def entry_effect(texts, embedder, cutoffs=(0.90, 0.85, 0.80)):
    """For every harmless text on its own as a wrong entry: how many OTHER harmless texts would the database block?
    Uses the same cleaning as the database. Returns {cutoff: array with one count per text}."""
    from src.attack_db.embedding_db import normalise
    vectors = np.asarray(embedder([normalise(t) for t in texts]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)
    return {c: (sims >= c).sum(axis=1) for c in cutoffs}


def entry(args):
    from src.attack_db.embedding_db import default_embedder
    attacks, harmless = load_prompts()
    print(f"\nENTRY TEST: {len(harmless)} harmless prompts, each one on its own as a wrong entry in the database")
    counts = entry_effect(harmless, default_embedder())
    for cutoff, c in counts.items():
        none = 100 * (c == 0).mean()
        print(f"  cutoff {cutoff:.2f}: one wrong entry blocks itself and on average {c.mean():.2f} other harmless prompts "
              f"(most: {c.max()}); {none:.0f}% of the entries block no other prompt")
    print("  with sharing, a wrong entry reaches all apps; without sharing only the app that made it")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="test", required=True)
    p = sub.add_parser("scale")
    add_common(p)
    p.set_defaults(runs=20)
    p = sub.add_parser("reports")
    add_common(p)
    p.add_argument("--rates", type=float, nargs="*", default=[0.1, 0.25, 0.5, 1.0],
                   help="shares of the missed attacks that are reported correctly")
    p.add_argument("--fakes", type=int, nargs="*", default=[5, 20, 50], help="numbers of wrong reports")
    p = sub.add_parser("quarantine")
    add_common(p)
    p.add_argument("--rates", type=float, nargs="*", default=[0.1, 0.25, 0.5, 1.0],
                   help="shares of the missed attacks that are reported correctly")
    p.add_argument("--fakes", type=int, nargs="*", default=[5, 20, 50], help="numbers of wrong reports")
    p.add_argument("--min-apps", type=int, default=2, help="different apps that must agree before a report counts for all")
    sub.add_parser("entry")
    args = parser.parse_args()
    {"scale": scale, "reports": reports, "quarantine": quarantine, "entry": entry}[args.test](args)


if __name__ == "__main__":
    main()
