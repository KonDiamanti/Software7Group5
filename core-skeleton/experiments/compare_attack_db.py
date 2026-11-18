# compare_attack_db.py
# compares three ways to store known attacks in the shared database:
#   SHA-256 (exact match), SimHash (letters/words close), embedding (meaning close)
#
# run from core-skeleton:   python -m experiments.compare_attack_db
# (first time: python -m experiments.make_variants  to build data/variants_seed*.csv)
#
# what it measures, for each method:
#   1. detection   share of each attack-variant set that the database blocks
#   2. false pos.  share of harmless prompts it blocks (benign_eval and hard_benign)
#   3. speed       time to add a prompt and to look one up, plus the scan time at 100k rows
#   4. storage     bytes per row in the sqlite file
#   5. leakage     how many of a stored attack's words someone can guess back from the stored fingerprint
#
# fair threshold: each method gets the loosest threshold that blocks at most --fp-target of benign_tune.
# benign_tune is never used again, so the false positive numbers below come from prompts the threshold has not seen.

import argparse
import os
import random
import re
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.attack_db.embedding_db import EmbeddingAttackDB, load_default_model
from src.attack_db.hash_db import HashAttackDB
from src.attack_db.simhash_db import SimHashAttackDB, _hash64, _popcount, exact_hash, normalise

HERE = Path(__file__).parent
ATTACK_SETS = ["copy", "format", "typo", "synonym", "wrapped", "unseen"]
BENIGN_SETS = ["benign_eval", "hard_benign"]
STOP = set("""the and for with from that this you your are was were have has had not but can will would could
should about into over all any its our their they them what when where which who how than then also just only
very more most some such other each been being does did doing here there these those she him her his""".split())


# ---------------------------------------------------------------- one wrapper per method
# score(text) = how close the prompt is to the closest stored attack, higher = closer, 1.0 = exact copy

class Method:
    name = ""
    def score(self, text): ...
    def set_threshold(self, t): ...


class Sha(Method):
    name = "SHA-256 (exact)"
    def __init__(self, path=":memory:"):
        self.db = HashAttackDB(path)
    def score(self, text):
        return 1.0 if self.db.lookup(text) is not None else 0.0
    def set_threshold(self, t):
        pass   # nothing to tune, it either matches or it does not


class SimHash(Method):
    def __init__(self, mode, n, path=":memory:"):
        self.name = f"SimHash 64-bit ({n}-{'letter' if mode == 'char' else 'word'} pieces)"
        self.db = SimHashAttackDB(path, mode=mode, n=n)
    def score(self, text):
        if self.db.con.execute("SELECT 1 FROM attacks WHERE hash=?", (exact_hash(text),)).fetchone():
            return 1.0
        d, _ = self.db.nearest(text)
        return 1 - d / 64
    def set_threshold(self, t):
        self.db.max_distance = int(round((1 - t) * 64))
    def threshold_text(self):
        return f"max {self.db.max_distance} of 64 bits different"


class Embedding(Method):
    def __init__(self, model, path=":memory:"):
        self.name = f"Embedding ({model.name})"
        self.db = EmbeddingAttackDB(path, model=model)
    def score(self, text):
        if self.db.con.execute("SELECT 1 FROM attacks WHERE hash=?", (exact_hash(text),)).fetchone():
            return 1.0
        s, _ = self.db.nearest(text)
        return s
    def set_threshold(self, t):
        self.db.min_similarity = t
    def threshold_text(self):
        return f"cosine >= {self.db.min_similarity:.3f}"


# ---------------------------------------------------------------- helpers

def pick_threshold(benign_scores, fp_target, step):
    """loosest threshold that blocks at most fp_target of the tuning prompts"""
    s = np.sort(np.asarray(benign_scores))[::-1]
    allowed = int(np.floor(fp_target * len(s)))
    edge = s[allowed] if allowed < len(s) else -1.0
    return min(1.0, edge + step)   # just above the closest harmless prompt we must not block


def content_words(text):
    return {w for w in re.findall(r"[a-z]{3,}", normalise(text)) if w not in STOP}


def leakage_recall(method, stored_texts, vocab, model=None, sample=100, seed=0):
    """
    attacker has the stored fingerprint and a list of candidate words (all words in the dataset).
    they rank the words by how well each fits the fingerprint and keep the top k (k = words really in the prompt).
    recall = share of the real words they got back. random guessing gives about k / len(vocab).
    """
    rng = random.Random(seed)
    targets = [t for t in stored_texts if len(content_words(t)) >= 5]
    targets = rng.sample(targets, min(sample, len(targets)))
    vocab = sorted(vocab)
    recalls, baselines = [], []

    if isinstance(method, Embedding):
        word_vecs = method.db.model.encode(vocab)
        def ranked(text):
            return np.argsort(-(word_vecs @ method.db.embed(text)))
    elif isinstance(method, SimHash) and method.db.mode == "char":
        n = method.db.n
        shifts = np.arange(64, dtype=np.uint64)
        word_bits = []
        for w in vocab:
            grams = [w[i:i + n] for i in range(len(w) - n + 1)] or [w]
            h = np.array([_hash64(g) for g in grams], dtype=np.uint64)
            word_bits.append((((h[:, None] >> shifts) & np.uint64(1)).mean(axis=0)))
        word_bits = np.vstack(word_bits)            # share of each word's pieces with a 1 in each bit
        def ranked(text):
            fp = method.db.fingerprint(text)
            fp_bits = np.array([(fp >> i) & 1 for i in range(64)], dtype=float)
            agree = (word_bits * fp_bits + (1 - word_bits) * (1 - fp_bits)).mean(axis=1)
            return np.argsort(-agree)
    else:
        return None, None

    for t in targets:
        real = content_words(t)
        k = len(real)
        top = {vocab[i] for i in ranked(t)[:k]}
        recalls.append(len(top & real) / k)
        baselines.append(k / len(vocab))
    return float(np.mean(recalls)), float(np.mean(baselines))


def scan_time_at(method, n_rows, repeats=200, seed=0):
    """time of the nearest-neighbour scan alone (no model, no sqlite) when the table has n_rows"""
    rng = np.random.default_rng(seed)
    if isinstance(method, SimHash):
        fps = rng.integers(0, 2**63, size=n_rows, dtype=np.uint64)
        q = np.uint64(12345)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _popcount(fps ^ q).argmin()
    elif isinstance(method, Embedding):
        dim = method.db._vecs.shape[1]
        vecs = rng.standard_normal((n_rows, dim)).astype(np.float32)
        q = vecs[0]
        t0 = time.perf_counter()
        for _ in range(repeats):
            (vecs @ q).argmax()
    else:
        return None   # SHA-256 is an index lookup in sqlite, it does not scan
    return (time.perf_counter() - t0) / repeats * 1e3


def file_bytes_per_row(make, texts):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.db")
        m = make(path)
        for t in texts:
            m.db.add(t)
        m.db.con.execute("VACUUM")
        m.db.con.close()
        return os.path.getsize(path) / len(texts)


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------- main

def run_split(make, v, fp_target, measure_costs, vocab):
    m = make()
    stored = v[v.set == "stored"].text.tolist()
    t0 = time.perf_counter()
    for t in stored:
        m.db.add(t)
    add_ms = (time.perf_counter() - t0) / len(stored) * 1e3

    q = v[v.set != "stored"].copy()
    q["score"] = [m.score(t) for t in q.text]
    thr = pick_threshold(q[q.set == "benign_tune"].score, fp_target,
                         step=1 / 64 if isinstance(m, SimHash) else 1e-4)
    m.set_threshold(thr)
    q["blocked"] = q.score >= thr

    row = {"method": m.name, "threshold": getattr(m, "threshold_text", lambda: "exact only")()}
    for s in ATTACK_SETS:
        row[s] = q[q.set == s].blocked.mean()
    for s in BENIGN_SETS:
        row["FP " + s] = q[q.set == s].blocked.mean()

    if measure_costs:
        # speed through the real lookup() path, with the chosen threshold
        sample = q[q.set.isin(ATTACK_SETS + BENIGN_SETS)].text.sample(300, random_state=0).tolist()
        times = []
        for t in sample:
            t0 = time.perf_counter()
            m.db.lookup(t)
            times.append((time.perf_counter() - t0) * 1e3)
        row["add ms"] = add_ms
        row["lookup ms (mean)"] = statistics.mean(times)
        row["lookup ms (p95)"] = float(np.percentile(times, 95))
        row["scan ms @100k rows"] = scan_time_at(m, 100_000)
        row["fingerprint bytes"] = (32 if isinstance(m, Sha) else 8 if isinstance(m, SimHash)
                                    else m.db._vecs.shape[1] * 4)
        row["bytes/row (sqlite)"] = file_bytes_per_row(make, stored)
        row["word recovery"], row["random guess"] = leakage_recall(m, stored, vocab)

    fps = [{"method": m.name, "set": s, "score": round(r.score, 3), "text": r.text[:160]}
           for s in BENIGN_SETS for _, r in q[(q.set == s) & q.blocked].iterrows()]

    # sweep: what each threshold would give, so the trade-off is visible beyond one chosen point
    grid = [1 - d / 64 for d in range(0, 25)] if isinstance(m, SimHash) else (
        np.round(np.arange(0.50, 1.0001, 0.02), 2) if isinstance(m, Embedding) else [1.0])
    sweep = []
    for g in grid:
        b = q.score >= g
        sr = {"method": m.name, "threshold": g}
        for s in ATTACK_SETS:
            sr[s] = b[q.set == s].mean()
        for s in ["benign_tune"] + BENIGN_SETS:
            sr["FP " + s] = b[q.set == s].mean()
        sweep.append(sr)
    return row, fps, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-target", type=float, default=0.0,
                    help="share of benign_tune each method may block (default 0: none)")
    ap.add_argument("--model", choices=["minilm", "wordllama"], default="minilm",
                    help="embedding model; minilm falls back to wordllama when it is not installed")
    args = ap.parse_args()

    files = sorted((HERE / "data").glob("variants_seed*.csv"))
    if not files:
        raise SystemExit("no data/variants_seed*.csv yet, run: python -m experiments.make_variants")
    splits = [pd.read_csv(f, keep_default_na=False) for f in files]
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    model = load_default_model(args.model)
    makers = {
        "sha": lambda p=":memory:": Sha(p),
        "sim_char": lambda p=":memory:": SimHash("char", 4, p),
        "sim_word": lambda p=":memory:": SimHash("word", 3, p),
        "emb": lambda p=":memory:": Embedding(model, p),
    }
    print("embedding model:", model.name, "| splits:", len(splits))

    vocab = set()
    for t in splits[0].text:
        vocab |= content_words(t)

    rows, fp_examples, sweeps = [], [], []
    for i, v in enumerate(splits):
        for make in makers.values():
            row, fps, sweep = run_split(make, v, args.fp_target, measure_costs=(i == 0), vocab=vocab)
            rows.append(dict(row, split=i))
            fp_examples += [dict(e, split=i) for e in fps]
            sweeps += [dict(r, split=i) for r in sweep]
        print(f"split {i} done")

    per_split = pd.DataFrame(rows)
    sweep = pd.DataFrame(sweeps).groupby(["method", "threshold"], sort=False).mean(numeric_only=True).drop(columns="split").reset_index()
    per_split.to_csv(out_dir / "attack_db_per_split.csv", index=False)
    sweep.to_csv(out_dir / "attack_db_sweep.csv", index=False)
    pd.DataFrame(fp_examples).to_csv(out_dir / "attack_db_false_positives.csv", index=False)
    write_report(per_split, fp_examples, model.name, args, splits[0], out_dir / "attack_db_comparison.md")
    print(f"results in {out_dir}")


def write_report(per_split, fp_examples, model_name, args, v, path):
    n = v.groupby("set").size()
    k = per_split.split.nunique()
    methods = list(dict.fromkeys(per_split.method))
    g = per_split.groupby("method", sort=False)
    ms = lambda m, col: f"{100 * g.get_group(m)[col].mean():.1f} ± {100 * g.get_group(m)[col].std(ddof=0):.1f}"
    thr = lambda m: " / ".join(list(g.get_group(m).threshold.str.replace(" of 64 bits different", "").str.replace("cosine >= ", "")))

    L = ["# Attack database: SHA-256 vs SimHash vs embedding", "",
         f"Embedding model: {model_name}. {k} random splits; numbers are mean ± spread (std) over the splits, in %. "
         f"Each split stores {n['stored']} attacks and tunes each threshold so it blocks at most {pct(args.fp_target)} "
         f"of {n['benign_tune']} harmless prompts (benign_tune), which are not used again.", "",
         "## Detection (% blocked, higher is better)", "",
         "| method | " + " | ".join(ATTACK_SETS) + " |", "|---|" + "---|" * len(ATTACK_SETS)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(ms(m, s) for s in ATTACK_SETS) + " |")
    L += ["", "Set sizes (split 0): " + ", ".join(f"{s} {n[s]}" for s in ATTACK_SETS), "",
          "## False positives (% of harmless prompts blocked, lower is better)", "",
          "| method | benign_eval | hard_benign |", "|---|---|---|"]
    for m in methods:
        L.append(f"| {m} | {ms(m, 'FP benign_eval')} | {ms(m, 'FP hard_benign')} |")
    L += ["", f"benign_eval {n['benign_eval']} prompts, hard_benign {n['hard_benign']} prompts per split. "
          "With this few prompts, 0% still allows a real rate of a few %.", "",
          "Thresholds chosen per split: " + "; ".join(f"{m}: {thr(m)}" for m in methods if not m.startswith("SHA")), "",
          "## Cost and privacy (split 0)", "",
          "| method | add ms | lookup ms mean | lookup ms p95 | scan ms @100k | fingerprint bytes | sqlite bytes/row | word recovery | random guess |",
          "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=2: "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"
    for _, r in per_split[per_split.split == 0].iterrows():
        sha = r.method.startswith("SHA")
        rec = "0 (exact guesses only)" if sha else ("n/a" if pd.isna(r["word recovery"]) else pct(r["word recovery"]))
        base = "" if sha or rec == "n/a" else pct(r["random guess"])
        L.append(f"| {r.method} | {f(r['add ms'])} | {f(r['lookup ms (mean)'], 3)} | {f(r['lookup ms (p95)'], 3)} | "
                 f"{f(r['scan ms @100k rows'], 3)} | {int(r['fingerprint bytes'])} | {f(r['bytes/row (sqlite)'], 0)} | {rec} | {base} |")
    L += ["", "Word recovery: from the stored fingerprint alone, rank every word in the dataset by fit and keep the "
          "top k. The number is the share of the attack's real words recovered. Random guess is the same with no "
          "fingerprint. SQLite bytes/row includes the SHA-256 column every version keeps for step 2 and dedup. "
          "scan ms @100k: the nearest-match scan alone over 100,000 synthetic rows (no index).", "",
          "## Harmless prompts that got blocked (all splits)", ""]
    if fp_examples:
        for e in fp_examples:
            L.append(f"- split {e['split']}, **{e['method']}**, {e['set']}, score {e['score']}: {e['text']!r}")
    else:
        L.append("None.")
    L += ["", "Per-split numbers: attack_db_per_split.csv. Threshold sweep (averaged over splits): attack_db_sweep.csv."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
