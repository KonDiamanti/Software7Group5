# experiments/repeat_rate.py
# Question: if the database already held every OTHER attack, how many attacks
# would a hash recognise, and how many would an embedding recognise?
# Also: how many harmless prompts would the embedding wrongly match?

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.attack_db.hash_db import fingerprint


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
    return attacks, harmless


attacks, harmless = load()
print(f"attacks: {len(attacks)}, harmless: {len(harmless)}")

# --- hash: attack i is recognised if another attack has the same fingerprint
counts = Counter(fingerprint(a) for a in attacks)
hash_hits = sum(1 for a in attacks if counts[fingerprint(a)] > 1)
print(f"\nHASH recognises {hash_hits}/{len(attacks)} attacks "
      f"({100 * hash_hits / len(attacks):.1f}%), harmless matched: 0")

# --- embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
t0 = time.time()
EA = model.encode(attacks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
EH = model.encode(harmless, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
print(f"embedding time: {1000 * (time.time() - t0) / (len(attacks) + len(harmless)):.1f} ms per prompt")

S = EA @ EA.T
np.fill_diagonal(S, -1)          # an attack must not match itself
best_attack = S.max(axis=1)      # closest OTHER attack
best_harmless = (EH @ EA.T).max(axis=1)   # closest attack to each harmless prompt

print("\nEMBEDDING (cosine similarity cutoff)")
print("cutoff | attacks recognised | harmless wrongly matched")
for t in [0.95, 0.90, 0.85, 0.80, 0.75]:
    a = (best_attack >= t).mean() * 100
    h = (best_harmless >= t).mean() * 100
    print(f" {t:.2f}  | {a:5.1f}%            | {h:5.1f}%")

# --- families: attacks joined if similarity >= cutoff
def families(cutoff):
    parent = list(range(len(attacks)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(*np.where(np.triu(S >= cutoff, 1))):
        parent[find(i)] = find(j)
    return Counter(find(i) for i in range(len(attacks)))

print("\nFAMILIES")
for t in [0.90, 0.80]:
    sizes = families(t)
    in_family = sum(s for s in sizes.values() if s >= 2)
    print(f"cutoff {t:.2f}: {len(sizes)} families, "
          f"{100 * in_family / len(attacks):.1f}% of attacks have a close relative, "
          f"largest sizes: {sorted(sizes.values(), reverse=True)[:5]}")