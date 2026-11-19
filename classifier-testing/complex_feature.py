"""The complex feature: linguistic markers as features for a learned check.

The rule-based check blocks a prompt as soon as one of its nine rules fires. Here each rule becomes an
on/off feature, and a model learns from the training data how much each one counts. Eight text-shape
features from GuardChain (Majhi et al. 2026) are added: length, punctuation, uppercase letters and so on.
"""
import re
import string

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from classifiers import COMPILED_RULES, LETTERS

RULE_NAMES = list(COMPILED_RULES)
SHAPE_NAMES = ["length", "words", "word length", "punctuation", "uppercase", "digits", "special characters",
               "repeated words"]
PUNCTUATION = set(string.punctuation)


def rule_features(texts):
    """1 if the rule fires on the prompt, else 0. One column per rule."""
    return np.array([[int(bool(rule.search(text))) for rule in COMPILED_RULES.values()] for text in texts])


def shape(text):
    """GuardChain's eight text-shape features. Length and word count are on a log scale here,
    because GuardChain divides them by the longest prompt, which needs all prompts at once."""
    words = re.findall(r"\w+", text.lower())
    characters = max(len(text), 1)
    return [
        np.log1p(len(text)),                                                   # length
        np.log1p(len(words)),                                                  # words
        np.mean([len(word) for word in words]) if words else 0.0,              # word length
        sum(c in PUNCTUATION for c in text) / characters,                      # punctuation
        sum("A" <= c <= "Z" for c in text) / characters,                       # uppercase
        sum("0" <= c <= "9" for c in text) / characters,                       # digits
        sum(not (c.isalnum() or c.isspace() or c in PUNCTUATION) for c in text) / characters,  # special characters, emoji
        1 - len(set(words)) / len(words) if words else 0.0,                    # repeated words
    ]


def shape_features(texts):
    return np.array([shape(text) for text in texts])


def markers(rules=True, text_shape=True):
    """The marker features as one block. The rules stay 0/1. The shape features are scaled to mean 0 and
    standard deviation 1 on the training data, so that no feature counts more just because its numbers are bigger."""
    blocks = []
    if rules:
        blocks.append(FunctionTransformer(rule_features))
    if text_shape:
        blocks.append(make_pipeline(FunctionTransformer(shape_features), StandardScaler()))
    return make_union(*blocks)


def svm():
    """Calibrated like in classifiers.py, so that it gives a score. With the shape features the SVM needs more
    steps to finish learning; without them it finishes early, so the baseline is the same as in classifiers.py."""
    return CalibratedClassifierCV(LinearSVC(max_iter=100000), cv=5)


def marker_models():
    """Char TF-IDF + SVM (our tier 1 pick) with and without the markers, and the markers on their own."""
    def chars_plus(**which):
        return make_pipeline(make_union(TfidfVectorizer(**LETTERS), markers(**which)), svm())

    return {
        "Char TF-IDF + SVM": make_pipeline(TfidfVectorizer(**LETTERS), svm()),
        "Char TF-IDF + SVM + rules": chars_plus(rules=True, text_shape=False),
        "Char TF-IDF + SVM + text shape": chars_plus(rules=False, text_shape=True),
        "Char TF-IDF + SVM + rules + text shape": chars_plus(rules=True, text_shape=True),
        "Markers only (rules + text shape) + LR": make_pipeline(markers(), LogisticRegression(max_iter=1000)),
    }


def marker_weights(model):
    """The weight logistic regression gives each marker. Positive = points to an attack. The shape features
    are scaled, so their weight is per standard deviation."""
    weights = model[-1].coef_[0]
    table = pd.DataFrame({"marker": RULE_NAMES + SHAPE_NAMES,
                          "kind": ["rule"] * len(RULE_NAMES) + ["text shape"] * len(SHAPE_NAMES),
                          "weight": weights})
    return table.sort_values("weight", ascending=False, key=abs).reset_index(drop=True)


def summary(table):
    """One row per model: attacks caught and false alarms per dataset, then F1, AUC and time on deepset + jackhhao."""
    counts = table.pivot(index="classifier", columns="dataset", values=["attacks caught", "false alarms"])
    rows = pd.DataFrame(index=table["classifier"].unique())
    for dataset in ["jackhhao", "deepset", "hackaprompt"]:
        if ("attacks caught", dataset) in counts:
            rows[f"{dataset} caught"] = counts[("attacks caught", dataset)]
            if dataset != "hackaprompt":   # HackAPrompt has only attacks
                rows[f"{dataset} false alarms"] = counts[("false alarms", dataset)]
    combined = table[table["dataset"] == "all"].set_index("classifier")
    for column in ["F1", "AUC", "ms per prompt"]:
        rows[column + " (all)"] = combined[column].round(3)
    return rows


def changed(predictions, before, after, show=5):
    """Prompts where adding the markers changed the answer, as examples for the report."""
    groups = {
        "newly caught attacks": (predictions["label"] == 1) & (predictions[before] == 0) & (predictions[after] == 1),
        "attacks no longer caught": (predictions["label"] == 1) & (predictions[before] == 1) & (predictions[after] == 0),
        "new false alarms": (predictions["label"] == 0) & (predictions[before] == 0) & (predictions[after] == 1),
        "false alarms fixed": (predictions["label"] == 0) & (predictions[before] == 1) & (predictions[after] == 0),
    }
    for title, rows in groups.items():
        rows = predictions[rows]
        print(f"{after} vs {before}: {len(rows)} {title}" + (f", the first {show}:" if len(rows) > show else ":"))
        for _, row in rows.head(show).iterrows():
            print(f"- [{row['dataset']}] {row['text'][:200]!r}")
        print()
