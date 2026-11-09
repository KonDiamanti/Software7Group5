# experiments/real_paraphrases.py
# part 1: hand-written rewordings of attacks the database knows
# part 2: harmless prompts that sound like attacks
# part 3: does turning leetspeak back into letters help (and does it hurt)

import os
import re
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.attack_db.hash_db import fingerprint
from paraphrase_set import ATTACKS, HARMLESS

CUTOFFS = [0.95, 0.90, 0.85, 0.80]
rng = random.Random(1)


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


def short(t, n=110):
    return " ".join(t.split())[:n]


# turns 1gn0r3 back into ignore, only in words that mix letters with digits
LEET_BACK = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})


def unleet(t):
    def fix(m):
        w = m.group(0)
        return w.translate(LEET_BACK) if re.search(r"[A-Za-z]", w) else w
    return re.sub(r"\S+", fix, t)


LEET = str.maketrans("aeio", "4310")


def leetspeak(t, p):
    return " ".join(w.translate(LEET) if rng.random() < p else w for w in t.split(" "))


def pct(x):
    return f"{100 * x:5.1f}%"


data_attacks, data_harmless = load()
originals = [o for o, _ in ATTACKS]
db_texts = list(dict.fromkeys(data_attacks + originals))
print(f"database: {len(db_texts)} attacks ({len(originals)} of them are our hand-written ones)")

model = SentenceTransformer("all-MiniLM-L6-v2")


def enc(texts):
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


print("encoding the database...")
EA = enc(db_texts)
db_hashes = {fingerprint(t) for t in db_texts}

# ---- part 1
rewrites = [(g, r) for g, (_, rs) in enumerate(ATTACKS) for r in rs]
S = enc([r for _, r in rewrites]) @ EA.T
best_any = S.max(axis=1)
own_idx = [db_texts.index(ATTACKS[g][0]) for g, _ in rewrites]
best_own = np.array([S[k, own_idx[k]] for k in range(len(rewrites))])
hash_hits = sum(fingerprint(r) in db_hashes for _, r in rewrites)

print(f"\nPART 1: {len(rewrites)} hand-written rewordings of {len(ATTACKS)} known attacks")
print(f"hash recognises {hash_hits} of {len(rewrites)}")
print("cutoff | any stored attack | its own original")
for c in CUTOFFS:
    print(f" {c:.2f}  |      {pct((best_any >= c).mean())}      |     {pct((best_own >= c).mean())}")

print("\nthe 8 rewordings furthest from their original:")
for k in np.argsort(best_own)[:8]:
    g, r = rewrites[k]
    print(f"\n[own {best_own[k]:.2f}, any {best_any[k]:.2f}]")
    print(f" original: {short(ATTACKS[g][0])}")
    print(f" reworded: {short(r)}")

# ---- part 2
HV = enc(HARMLESS)
hb = (HV @ EA.T).max(axis=1)
print(f"\nPART 2: {len(HARMLESS)} harmless prompts that sound like attacks")
print("cutoff | wrongly matched")
for c in CUTOFFS:
    print(f" {c:.2f}  |  {pct((hb >= c).mean())}")
print("\nclosest ones:")
for k in np.argsort(-hb)[:5]:
    print(f"\n[{hb[k]:.2f}] harmless: {short(HARMLESS[k])}")
    print(f"       attack:   {short(db_texts[int((HV[k] @ EA.T).argmax())])}")

# ---- part 3
print("\nPART 3: leetspeak, before and after turning digits back into letters")
print("encoding the database again with the fix...")
EA_fix = enc([unleet(t) for t in db_texts])
sample = rng.sample(range(len(data_attacks)), min(200, len(data_attacks)))
for p in [0.15, 0.40]:
    edited = []
    for i in sample:
        e = leetspeak(data_attacks[i], p)
        if e != data_attacks[i]:
            edited.append(e)
    raw = (enc(edited) @ EA.T).max(axis=1)
    fixed = (enc([unleet(t) for t in edited]) @ EA_fix.T).max(axis=1)
    lens = np.array([len(t) for t in edited])
    print(f"\nleetspeak on {int(p * 100)}% of the words, {len(edited)} prompts")
    print("group              n   raw>=0.90  fixed>=0.90  raw>=0.85  fixed>=0.85")
    for name, mask in [("short (<=200 ch)", lens <= 200), ("long (>200 ch)", lens > 200), ("all", lens > 0)]:
        if mask.sum() == 0:
            continue
        print(f"{name:<17}{int(mask.sum()):>4}   {pct((raw[mask] >= .90).mean())}    {pct((fixed[mask] >= .90).mean())}"
              f"    {pct((raw[mask] >= .85).mean())}    {pct((fixed[mask] >= .85).mean())}")

print("\nharmless dataset prompts matched by mistake, without and with the fix")
HD = enc(data_harmless)
HD_fix = enc([unleet(t) for t in data_harmless])
a = (HD @ EA.T).max(axis=1)
b = (HD_fix @ EA_fix.T).max(axis=1)
for c in [0.90, 0.85]:
    print(f" cutoff {c:.2f}: raw {pct((a >= c).mean())}   fixed {pct((b >= c).mean())}")