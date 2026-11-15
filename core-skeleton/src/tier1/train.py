# train.py
# trains tier 1 and saves it. Run from inside the core-skeleton folder:
#   python -m src.tier1.train               same training data as classifier-testing (safe-guard only)
#   python -m src.tier1.train --more-data   also uses the train splits of deepset and jackhhao
# (the test splits of deepset and jackhhao are never used for training)
# It needs internet once (Hugging Face datasets) and takes about a minute on a laptop.

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.result import BLOCK, PASS, UNSURE
from src.tier1.language import is_english
from src.tier1.char_svm import CharSvmTier1, choose_cutoffs, make_model

NEAR_COPY = 0.9   # cosine similarity of TF-IDF vectors from which two prompts count as near copies
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tier1_char_svm.joblib"


def load_training():
    # same training data as in classifier-testing (the safe-guard train split)
    data = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    return list(data["text"]), [int(x) for x in data["label"]]


def load_extra_training():
    # the train splits of deepset and jackhhao. The cheap check only knew safe-guard,
    # which looks different from these two.
    deepset = load_dataset("deepset/prompt-injections", split="train")
    jackhhao = load_dataset("jackhhao/jailbreak-classification", split="train")
    texts = [r["text"] for r in deepset] + [r["prompt"] for r in jackhhao]
    labels = [int(r["label"]) for r in deepset] + [int(r["type"] == "jailbreak") for r in jackhhao]
    return texts, labels


def remove_near_copies(texts, labels, test_texts):
    """Drop training prompts that are near copies of a test prompt (same rule as classifier-testing/prompt_data.py).

    Without this the numbers on the test sets would look better than they are.
    Repeats until none is left, because removing prompts changes the TF-IDF weights a little.
    """
    while True:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
        closest = cosine_similarity(vectorizer.transform(texts), vectorizer.transform(test_texts)).max(axis=1)
        keep = [i for i, similarity in enumerate(closest) if similarity < NEAR_COPY]
        if len(keep) == len(texts):
            return texts, labels
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]


def load_test_sets():
    # the two public test sets from classifier-testing, label 1 = attack
    deepset = load_dataset("deepset/prompt-injections", split="test")
    jackhhao = load_dataset("jackhhao/jailbreak-classification", split="test")
    return {
        "deepset": ([r["text"] for r in deepset], [int(r["label"]) for r in deepset]),
        "jackhhao": ([r["prompt"] for r in jackhhao], [int(r["type"] == "jailbreak") for r in jackhhao]),
    }


def report(tier1, name, texts, labels):
    counts = {0: {BLOCK: 0, PASS: 0, UNSURE: 0}, 1: {BLOCK: 0, PASS: 0, UNSURE: 0}}
    for text, label in zip(texts, labels):
        counts[label][tier1.check(text).verdict] += 1
    attacks, harmless = sum(labels), len(labels) - sum(labels)
    print(f"\n{name}: {attacks} attacks, {harmless} harmless prompts")
    for label, word, total in ((1, "attacks", attacks), (0, "harmless", harmless)):
        c = counts[label]
        print(f"  {word:9} blocked {c[BLOCK]:4}  passed {c[PASS]:4}  unsure {c[UNSURE]:4}"
              f"   ({100 * c[UNSURE] / max(total, 1):.0f}% go to the guard)")
    print(f"  mistakes: {counts[1][PASS]} attacks passed, {counts[0][BLOCK]} harmless blocked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--more-data", action="store_true",
                        help="also use the train splits of deepset and jackhhao (half to train, half to set the cutoffs)")
    args = parser.parse_args()

    test_sets = load_test_sets()
    test_texts = [t for x, _ in test_sets.values() for t in x]

    texts, labels = load_training()
    before = len(texts)
    texts, labels = remove_near_copies(texts, labels, test_texts)
    print(f"safe-guard: removed {before - len(texts)} near copies of test prompts, {len(texts)} prompts left")

    if args.more_data:
        # cutoffs are set on deepset and jackhhao prompts that the model has not seen,
        # because the cutoffs set on safe-guard were too optimistic for other kinds of prompts
        extra_x, extra_y = load_extra_training()
        extra_x, extra_y = remove_near_copies(extra_x, extra_y, test_texts)
        ex_fit_x, ex_val_x, ex_fit_y, ex_val_y = train_test_split(
            extra_x, extra_y, test_size=0.5, random_state=0, stratify=extra_y)
        keep = [i for i, t in enumerate(ex_val_x) if is_english(t)]   # other languages go to the guard anyway
        fit_x, fit_y = texts + ex_fit_x, labels + ex_fit_y
        val_x, val_y = [ex_val_x[i] for i in keep], [ex_val_y[i] for i in keep]
        val_name = "validation (deepset + jackhhao train parts, English)"
    else:
        fit_x, val_x, fit_y, val_y = train_test_split(texts, labels, test_size=0.2, random_state=0, stratify=labels)
        val_name = "validation (safe-guard)"
    print(f"training on {len(fit_x)} prompts, setting the cutoffs on {len(val_x)} other prompts")

    model = make_model().fit(fit_x, fit_y)
    scores = model.predict_proba(val_x)[:, 1]
    pass_at, block_at = choose_cutoffs(list(scores), val_y)
    print(f"pass_at {pass_at:.3f}, block_at {block_at:.3f}")

    tier1 = CharSvmTier1(model, pass_at, block_at)
    report(tier1, val_name, val_x, val_y)
    for name, (x, y) in test_sets.items():
        report(tier1, name, x, y)

    path = MODEL_PATH.with_name("tier1_char_svm_more_data.joblib") if args.more_data else MODEL_PATH
    path.parent.mkdir(exist_ok=True)
    tier1.save(path)
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    sys.exit(main())
