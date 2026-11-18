# experiments/show_examples.py
import os
import sys
import random

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


def short(t):
    return " ".join(t.split())[:160]


attacks, harmless = load()
model = SentenceTransformer("all-MiniLM-L6-v2")
EA = model.encode(attacks, normalize_embeddings=True, batch_size=64)
EH = model.encode(harmless, normalize_embeddings=True, batch_size=64)

S = EA @ EA.T
np.fill_diagonal(S, -1)

print("PAIRS the hash misses but the embedding finds (similarity 0.90 to 0.95)")
random.seed(1)
pairs = [(i, j) for i, j in zip(*np.where(np.triu(S >= 0.90, 1)))
         if fingerprint(attacks[i]) != fingerprint(attacks[j]) and S[i, j] < 0.95]
print(f"({len(pairs)} such pairs, showing 8)")
for i, j in random.sample(pairs, min(8, len(pairs))):
    print(f"\n[{S[i, j]:.2f}]\n A: {short(attacks[i])}\n B: {short(attacks[j])}")

print("\n\nHARMLESS prompts wrongly matched at 0.90")
H = EH @ EA.T
for k in np.where(H.max(axis=1) >= 0.90)[0]:
    j = H[k].argmax()
    print(f"\n[{H[k, j]:.2f}]\n harmless: {short(harmless[k])}\n attack:   {short(attacks[j])}")