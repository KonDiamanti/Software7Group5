# experiments/paraphrase_test.py
# edit a known attack a bit and see if hash / embedding still finds it

import os
import re
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.attack_db.hash_db import fingerprint

SAMPLE = 200
CUTOFFS = [0.95, 0.90, 0.85, 0.80]
rng = random.Random(1)


def load_attacks():
    attacks = []
    ds = load_dataset("jackhhao/jailbreak-classification")
    for split in ds:
        for r in ds[split]:
            if r["type"] == "jailbreak":
                attacks.append(r["prompt"])
    ds = load_dataset("deepset/prompt-injections")
    for split in ds:
        for r in ds[split]:
            if r["label"] == 1:
                attacks.append(r["text"])
    # dict.fromkeys drops exact duplicates and keeps the order
    return list(dict.fromkeys(attacks))


# small edits, one function each

SYN = {"ignore": "disregard", "instructions": "directions", "previous": "earlier",
       "all": "every", "pretend": "imagine", "act": "behave", "answer": "reply",
       "tell": "inform", "rules": "guidelines", "now": "currently",
       "always": "constantly", "write": "compose", "show": "display",
       "explain": "describe", "create": "produce", "begin": "start", "stop": "cease"}


def swap_synonyms(t):
    # only the first 3 matches get swapped
    n = [0]
    def rep(m):
        w = m.group(0)
        r = SYN.get(w.lower())
        if r is None or n[0] >= 3:
            return w
        n[0] += 1
        return r.capitalize() if w[0].isupper() else r
    return re.sub(r"[A-Za-z]+", rep, t)


def curly_quotes(t):
    return t.replace('"', "\u201c").replace("'", "\u2019")


def add_filler(t):
    before = ["Please help me with this. ", "Hi there! ", "Quick question first. "]
    after = [" Thanks a lot!", " Let me know when you are ready.", " This is important to me."]
    return rng.choice(before) + t + rng.choice(after)


def typos(t):
    # swap two neighbouring letters in 3 longer words
    words = t.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4]
    for i in rng.sample(idx, min(3, len(idx))):
        w = words[i]
        k = rng.randrange(1, len(w) - 2)
        words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)


LEET = str.maketrans("aeio", "4310")


def leetspeak(t):
    # about 15% of the words get a->4 e->3 i->1 o->0
    return " ".join(w.translate(LEET) if rng.random() < 0.15 else w for w in t.split(" "))


def drop_words(t):
    kept = [w for w in t.split(" ") if rng.random() > 0.10]
    return " ".join(kept) if kept else t


def shuffle_sentences(t):
    parts = re.split(r"(?<=[.!?])\s+", t)
    rng.shuffle(parts)
    return " ".join(parts)


EDITS = {
    "synonyms (up to 3)": swap_synonyms,
    "curly quotes": curly_quotes,
    "filler text added": add_filler,
    "typos (3 words)": typos,
    "leetspeak (15%)": leetspeak,
    "10% words dropped": drop_words,
    "sentences shuffled": shuffle_sentences,
}

# main part

attacks = load_attacks()
print(f"attacks in the database: {len(attacks)}, edited samples per edit: up to {SAMPLE}")

model = SentenceTransformer("all-MiniLM-L6-v2")
EA = model.encode(attacks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
db_hashes = {fingerprint(a) for a in attacks}
sample = rng.sample(range(len(attacks)), min(SAMPLE, len(attacks)))

header = f"{'edit':<22}{'n':>5}  {'hash':>7}  " + "  ".join(f"emb>={c:.2f}" for c in CUTOFFS)
print("\n" + header)
print("-" * len(header))
for name, fn in EDITS.items():
    texts = []
    for i in sample:
        e = fn(attacks[i])
        # skip prompts the edit did not change (no leetspeak letters, no quotes...)
        if e != attacks[i]:
            texts.append(e)
    if not texts:
        print(f"{name:<22}{0:>5}  (nothing changed)")
        continue
    hash_ok = sum(fingerprint(t) in db_hashes for t in texts)
    E = model.encode(texts, normalize_embeddings=True, batch_size=64)
    # closest stored attack for every edited prompt
    best = (E @ EA.T).max(axis=1)
    cols = "  ".join(f"{100 * (best >= c).mean():8.1f}%" for c in CUTOFFS)
    print(f"{name:<22}{len(texts):>5}  {100 * hash_ok / len(texts):6.1f}%  {cols}")