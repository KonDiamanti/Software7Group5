# experiments/learning_curve.py
# does the service get better at catching unseen attacks when more apps share
# the prompts their guard models confirmed?
#   shared classifier: logistic regression on embeddings, trained on N apps' data
#   shared memory:     closest-stored-attack lookup, built from the same N apps' data
# N = 1 means no sharing, one app with only its own data

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from paraphrase_set import ATTACKS, EXTRA, HARMLESS

MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
APPS = [1, 2, 5, 10, 20]
SEEDS = 20
MEMORY_CUTOFF = 0.90
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


data_attacks, data_harmless = load()
rewordings = [v for (_, rs), ex in zip(ATTACKS, EXTRA) for v in rs + ex]
texts = data_attacks + data_harmless + HARMLESS + rewordings
na, nh, nl, nr = len(data_attacks), len(data_harmless), len(HARMLESS), len(rewordings)

# how many confirmed attacks (and harmless prompts) each app brings
PER_APP = min(30, int(0.7 * na) // max(APPS), int(0.6 * nh) // max(APPS))
print(f"attacks {na}, harmless {nh}, look-alikes {nl}, rewordings {nr}")
print(f"each app brings {PER_APP} confirmed attacks and {PER_APP} harmless prompts, {SEEDS} random runs")


def embed_all(name):
    model = SentenceTransformer(name)
    path = os.path.join(CACHE_DIR, "emb_" + name.replace("/", "_") + ".npz")
    if os.path.exists(path):
        E = np.load(path)["E"]
        if len(E) == len(texts):
            return model, E
    E = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    np.savez(path, E=E)
    return model, E


def ms_per_prompt(model):
    sample = random.Random(0).sample(texts[:na + nh], 30)
    model.encode(sample[:2])                       # warm up
    t0 = time.time()
    for t in sample:
        model.encode([t], normalize_embeddings=True)
    return 1000 * (time.time() - t0) / len(sample)


for name in MODELS:
    print("\n" + "=" * 78)
    print("MODEL:", name)
    model, E = embed_all(name)
    print(f"time to embed one prompt on this machine: {ms_per_prompt(model):.0f} ms")

    EA, EH = E[:na], E[na:na + nh]
    EL, ER = E[na + nh:na + nh + nl], E[na + nh + nl:]

    rows = {n: {"clf_r": [], "clf_a": [], "auc_r": [], "mem_r": [], "mem_a": [], "mem_fm": []} for n in APPS}

    for seed in range(SEEDS):
        rs = np.random.RandomState(seed)
        pa, ph = rs.permutation(na), rs.permutation(nh)
        test_a, train_a = pa[:int(0.3 * na)], pa[int(0.3 * na):]
        test_h, train_h = ph[:int(0.4 * nh)], ph[int(0.4 * nh):]
        neg = np.vstack([EH[test_h], EL])          # harmless prompts nobody trained on

        for n in APPS:
            att = train_a[:n * PER_APP]
            har = train_h[:n * PER_APP]
            X = np.vstack([EA[att], EH[har]])
            y = [1] * len(att) + [0] * len(har)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)

            neg_s = clf.decision_function(neg)
            thr = np.quantile(neg_s, 0.99)         # 1% of harmless prompts flagged
            r_s = clf.decision_function(ER)
            a_s = clf.decision_function(EA[test_a])
            rows[n]["clf_r"].append((r_s > thr).mean())
            rows[n]["clf_a"].append((a_s > thr).mean())
            rows[n]["auc_r"].append(roc_auc_score([1] * len(r_s) + [0] * len(neg_s),
                                                  np.concatenate([r_s, neg_s])))

            db = EA[att]                           # memory built from the same attacks
            rows[n]["mem_r"].append(((ER @ db.T).max(axis=1) >= MEMORY_CUTOFF).mean())
            rows[n]["mem_a"].append(((EA[test_a] @ db.T).max(axis=1) >= MEMORY_CUTOFF).mean())
            rows[n]["mem_fm"].append(((neg @ db.T).max(axis=1) >= MEMORY_CUTOFF).mean())

    def ms(v):
        return f"{100 * np.mean(v):5.1f}% +-{100 * np.std(v):4.1f}"

    print(f"\nclassifier: at 1% harmless flagged. memory: cutoff {MEMORY_CUTOFF}. mean over {SEEDS} runs")
    print("apps | CLASSIFIER rewordings | CLASSIFIER held-out | rewording AUC | MEMORY rewordings | MEMORY held-out | MEMORY false matches")
    for n in APPS:
        r = rows[n]
        print(f"{n:>4} | {ms(r['clf_r']):<21} | {ms(r['clf_a']):<19} | {np.mean(r['auc_r']):13.3f} | "
              f"{100 * np.mean(r['mem_r']):16.1f}% | {100 * np.mean(r['mem_a']):14.1f}% | {100 * np.mean(r['mem_fm']):19.1f}%")