# experiments/compare_models.py
# 1. which embedding model separates reworded attacks from harmless prompts best
# 2. nearest-neighbour memory vs a classifier trained on the embeddings
# 3. does the memory get better when more variants of an attack are known

import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from paraphrase_set import ATTACKS, EXTRA, HARMLESS

MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "all-mpnet-base-v2"]
assert len(EXTRA) == len(ATTACKS)


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
originals = [o for o, _ in ATTACKS]
db_texts = list(dict.fromkeys(data_attacks + originals))
neg_texts = data_harmless + HARMLESS          # dataset harmless first, then our look-alikes
nd = len(data_harmless)

# 5 rewordings per attack: 3 first set, 2 extra
variants = [rs + ex for (_, rs), ex in zip(ATTACKS, EXTRA)]
per = len(variants[0])
flat = [v for vs in variants for v in vs]

print(f"database: {len(db_texts)} attacks, harmless: {nd} dataset + {len(HARMLESS)} look-alikes")
print(f"rewordings: {len(flat)} ({per} for each of {len(ATTACKS)} attacks)")


def detect_at_1pct(pos, neg):
    thr = np.quantile(neg, 0.99)
    return thr, (pos > thr).mean()


for name in MODELS:
    print("\n" + "=" * 70)
    print("MODEL:", name)
    model = SentenceTransformer(name)

    def enc(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    EA, EN, EV = enc(db_texts), enc(neg_texts), enc(flat)
    labels = [1] * len(flat)

    # nearest-neighbour memory: score = similarity to the closest stored attack
    pos_knn = (EV @ EA.T).max(axis=1)
    neg_knn = (EN @ EA.T).max(axis=1)
    auc = roc_auc_score(labels + [0] * len(neg_knn), np.concatenate([pos_knn, neg_knn]))
    thr, det = detect_at_1pct(pos_knn, neg_knn)
    print(f"memory (closest attack): AUC {auc:.3f}, at 1% false matches the cutoff is {thr:.2f} "
          f"and it recognises {100 * det:.1f}% of the rewordings")

    # classifier on the same embeddings, harmless scores come from cross-validation
    Xtr = np.vstack([EA, EN[:nd]])
    ytr = [1] * len(EA) + [0] * nd
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    oof = cross_val_predict(clf, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
    clf.fit(Xtr, ytr)
    pos_lr = clf.predict_proba(EV)[:, 1]
    neg_lr = np.concatenate([oof[len(EA):], clf.predict_proba(EN[nd:])[:, 1]])
    auc2 = roc_auc_score(labels + [0] * len(neg_lr), np.concatenate([pos_lr, neg_lr]))
    thr2, det2 = detect_at_1pct(pos_lr, neg_lr)
    print(f"classifier on embeddings: AUC {auc2:.3f}, at 1% false alarms it catches {100 * det2:.1f}% of the rewordings")

    # does the memory improve when more variants of the same attack are known?
    hits = {k: [] for k in range(per)}
    for g in range(len(ATTACKS)):
        Vg = EV[g * per:(g + 1) * per]
        within = Vg @ Vg.T
        base = pos_knn[g * per:(g + 1) * per]
        for i in range(per):
            others = [j for j in range(per) if j != i]
            for k in range(per):
                for known in itertools.combinations(others, k):
                    score = base[i] if not known else max(base[i], max(within[i, j] for j in known))
                    hits[k].append(score > thr)
    print("memory with k known rewordings of the same attack (same cutoff as above):")
    print("  " + "   ".join(f"k={k}: {100 * np.mean(hits[k]):5.1f}%" for k in range(per)))