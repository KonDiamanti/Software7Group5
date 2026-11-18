# experiments/cascade_sim.py
# the whole cascade for N apps sending traffic, with and without sharing:
#   1. memory of known attacks (hash or embedding)  -> block
#   2. tier 1 (logistic regression on embeddings)   -> block / pass / unsure
#   3. guard (only when unsure), here always right  -> block / pass
# every guard answer is learned: attacks go into the memory, all answers train tier 1
# tier 1 cutoffs follow the 2% rule from Section 4, measured on a calibration set
# that is never used for training

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.attack_db.hash_db import fingerprint

SEEDS = 10
T_PER_APP = 100          # prompts each app sends
ATTACK_SHARE = 0.20      # share of attacks in the traffic
SEED_SIZE = 100          # attacks and harmless prompts tier 1 starts with (public data)
CAL_SIZE = 60            # attacks and harmless prompts only used to set the cutoffs
CAL_RATE = 0.02          # block cutoff: 2% of harmless above it, pass cutoff: 2% of attacks below it
MEMORY_CUTOFF = 0.90     # embedding memory: similarity needed to call it a known attack
RETRAIN_EVERY = 20       # retrain tier 1 after this many new guard answers
MAIN_REPEAT = 0.30       # share of attacks that are repeats (we measured about 0.28)
APPS = [1, 2, 5, 10]
REPEATS = [0.0, 0.1, 0.3, 0.5, 0.7]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load():
    attacks, harmless = [], []
    ds = load_dataset("jackhhao/jailbreak-classification")
    for split in ds:
        for r in ds[split]:
            (attacks if r["type"] == "jailbreak" else harmless).append(r["prompt"])
    ds = load_dataset("deepset/prompt-injections")
    for split in ds:
        for r in ds[split]:
            (attacks if r["label"] == 1 else harmless).append(r["text"])
    return list(dict.fromkeys(attacks)), list(dict.fromkeys(harmless))


# ---- edited copies of attacks (for repeats that are not exact) ----------------
edit_rng = random.Random(0)


def add_filler(t):
    before = ["Please help me with this. ", "Hi there! ", "Quick question first. "]
    after = [" Thanks a lot!", " Let me know when you are ready.", " This is important to me."]
    return edit_rng.choice(before) + t + edit_rng.choice(after)


def typos(t):
    words = t.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4]
    for i in edit_rng.sample(idx, min(3, len(idx))):
        w = words[i]
        k = edit_rng.randrange(1, len(w) - 2)
        words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)


def drop_words(t):
    kept = [w for w in t.split(" ") if edit_rng.random() > 0.10]
    return " ".join(kept) if kept else t


def edited(t):
    e = edit_rng.choice([add_filler, typos, drop_words])(t)
    return e if e != t else add_filler(t)


attacks, harmless = load()
na, nh = len(attacks), len(harmless)
variants = [[edited(a), edited(a)] for a in attacks]      # two edited copies of every attack
print(f"attacks {na}, harmless {nh}")

model = SentenceTransformer("all-MiniLM-L6-v2")
path = os.path.join(CACHE_DIR, "emb_sim_MiniLM.npz")
alltext = attacks + harmless + [v for vs in variants for v in vs]
cached = np.load(path)["E"] if os.path.exists(path) else None
if cached is not None and len(cached) == len(alltext):
    E = cached
else:
    print("embedding everything once (about 1-2 minutes)...")
    E = model.encode(alltext, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    np.savez(path, E=E)
EA, EH = E[:na], E[na:na + nh]
EV = E[na + nh:].reshape(na, 2, -1)
DIM = E.shape[1]


# ---- one "brain": memory + tier 1 ----------------------------------------------
class Brain:
    def __init__(self, seedX, seedy, cal, memory, learn):
        self.memory, self.learn = memory, learn
        self.cal_att, self.cal_har = cal
        self.X, self.y = list(seedX), list(seedy)
        self.fp, self.mem = set(), np.zeros((0, DIM))
        self.new = 0
        self.fit()

    def fit(self):
        self.clf = LogisticRegression(max_iter=300, class_weight="balanced").fit(np.array(self.X), self.y)
        # cutoffs from the calibration prompts: tier 1 blocks above hi, passes below lo
        self.hi = np.quantile(self.clf.predict_proba(self.cal_har)[:, 1], 1 - CAL_RATE)
        self.lo = np.quantile(self.clf.predict_proba(self.cal_att)[:, 1], CAL_RATE)
        if self.lo > self.hi:
            self.lo = self.hi = (self.lo + self.hi) / 2

    def decide(self, vec, text):
        if self.memory == "hash" and fingerprint(text) in self.fp:
            return "block"
        if self.memory == "embedding" and len(self.mem) and (self.mem @ vec).max() >= MEMORY_CUTOFF:
            return "block"
        p = self.clf.predict_proba(vec[None])[0, 1]
        if p >= self.hi:
            return "block"
        if p <= self.lo:
            return "pass"
        return "guard"

    def confirm(self, vec, text, label):
        if label == 1 and self.memory == "hash":
            self.fp.add(fingerprint(text))
        if label == 1 and self.memory == "embedding":
            self.mem = np.vstack([self.mem, vec])
        if self.learn:
            self.X.append(vec)
            self.y.append(label)
            self.new += 1
            if self.new >= RETRAIN_EVERY:
                self.fit()
                self.new = 0


CONDS = [  # name, memory, learns from the guard, one brain per app?
    ("static cascade", None, False, False),
    ("separate", "embedding", True, True),
    ("shared hash", "hash", True, False),
    ("shared embedding", "embedding", True, False),
    ("shared emb. memory only", "embedding", False, False),
]


def make_traffic(rs, n_apps, repeat_rate):
    perm_a, perm_h = list(rs.permutation(na)), list(rs.permutation(nh))
    seed_a, seed_h = perm_a[:SEED_SIZE], perm_h[:SEED_SIZE]
    cal_a, cal_h = perm_a[SEED_SIZE:SEED_SIZE + CAL_SIZE], perm_h[SEED_SIZE:SEED_SIZE + CAL_SIZE]
    free_a, free_h = perm_a[SEED_SIZE + CAL_SIZE:], perm_h[SEED_SIZE + CAL_SIZE:]
    seedX = [EA[i] for i in seed_a] + [EH[j] for j in seed_h]
    seedy = [1] * len(seed_a) + [0] * len(seed_h)
    cal = (EA[cal_a], EH[cal_h])
    seen, traffic = [], []
    for _ in range(T_PER_APP):
        for app in rs.permutation(n_apps):
            if rs.rand() < ATTACK_SHARE:
                if seen and rs.rand() < repeat_rate:
                    i, v = seen[rs.randint(len(seen))], rs.randint(3)   # 0 = exact copy
                    vec, text = (EA[i], attacks[i]) if v == 0 else (EV[i, v - 1], variants[i][v - 1])
                else:
                    i = free_a.pop()
                    seen.append(i)
                    vec, text = EA[i], attacks[i]
                traffic.append((int(app), vec, text, 1))
            else:
                j = free_h.pop() if free_h else perm_h[SEED_SIZE + CAL_SIZE + rs.randint(nh - SEED_SIZE - CAL_SIZE)]
                traffic.append((int(app), EH[j], harmless[j], 0))
    return seedX, seedy, cal, traffic


def run(seedX, seedy, cal, traffic, n_apps, memory, learn, separate):
    brains = [Brain(seedX, seedy, cal, memory, learn) for _ in range(n_apps if separate else 1)]
    guard = missed = falseb = 0
    for app, vec, text, label in traffic:
        b = brains[app if separate else 0]
        d = b.decide(vec, text)
        if d == "guard":
            guard += 1
            b.confirm(vec, text, label)
        elif d == "block" and label == 0:
            falseb += 1
        elif d == "pass" and label == 1:
            missed += 1
    n_att = sum(t[3] for t in traffic)
    return (1000 * guard / len(traffic), 100 * missed / max(n_att, 1),
            100 * falseb / max(len(traffic) - n_att, 1))


combos = [(n, MAIN_REPEAT) for n in APPS] + [(max(APPS), r) for r in REPEATS if r != MAIN_REPEAT]
res = {c: {name: [] for name, *_ in CONDS} for c in combos}
for seed in range(SEEDS):
    for n, r in combos:
        rs = np.random.RandomState(seed * 100 + n)
        seedX, seedy, cal, traffic = make_traffic(rs, n, r)
        for name, memory, learn, separate in CONDS:
            res[(n, r)][name].append(run(seedX, seedy, cal, traffic, n, memory, learn, separate))
    print(f"seed {seed + 1}/{SEEDS} done")

names = [c[0] for c in CONDS]
head = "".join(f"{nm:>25}" for nm in names)


def table(title, metric, rows, label):
    print("\n" + title)
    print(f"{label:>8}" + head)
    for key in rows:
        vals = "".join(f"{np.mean([x[metric] for x in res[key][nm]]):25.1f}" for nm in names)
        print(f"{key[0] if label == 'apps' else key[1]:>8}" + vals)


main_rows = [(n, MAIN_REPEAT) for n in APPS]
rep_rows = [(max(APPS), r) for r in REPEATS]
print(f"\nguard only (every prompt to the guard): 1000.0 guard calls per 1,000 prompts")
print(f"traffic: {T_PER_APP} prompts per app, {int(100 * ATTACK_SHARE)}% attacks, repeat rate {int(100 * MAIN_REPEAT)}%, "
      f"{SEEDS} runs, guard assumed always right")
table("GUARD CALLS per 1,000 prompts (lower is better), by number of apps", 0, main_rows, "apps")
table("MISSED ATTACKS in % of attacks (lower is better)", 1, main_rows, "apps")
table("HARMLESS PROMPTS BLOCKED in % of harmless (lower is better)", 2, main_rows, "apps")
print(f"\nSAME, with {max(APPS)} apps and different repeat rates")
table("GUARD CALLS per 1,000 prompts, by repeat rate", 0, rep_rows, "repeat")
table("MISSED ATTACKS %, by repeat rate", 1, rep_rows, "repeat")
table("HARMLESS BLOCKED %, by repeat rate", 2, rep_rows, "repeat")