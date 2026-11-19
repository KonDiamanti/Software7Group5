"""Three-tier adversarial-prompt cascade prototype.

Tier 1 uses high-specificity rules. Tier 2 uses Char TF-IDF + calibrated SVM.
Tier 3 calls a guard model only for uncertain prompts.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from classifiers import COMPILED_RULES, LETTERS, hf_classifier, pick_device, train_sklearn


HIGH_SPECIFICITY_RULES = (
    "ignore instructions",
    "no limits",
    "jailbreak names",
)


@dataclass
class CascadeResult:
    text: str
    decision: str
    stage: str
    score: float
    guard_used: bool
    milliseconds: float


class Cascade:
    def __init__(self, char_model=None, guard_check=None,
                 block_cutoff=0.80, pass_cutoff=0.20):
        self.char_model = char_model
        self.guard_check = guard_check
        self.block_cutoff = block_cutoff
        self.pass_cutoff = pass_cutoff
        self.training_seconds = None

        if pass_cutoff > block_cutoff:
            raise ValueError("pass_cutoff must be no greater than block_cutoff")

    def train(self, train):
        model = make_pipeline(
            TfidfVectorizer(**LETTERS),
            CalibratedClassifierCV(LinearSVC(max_iter=100000), cv=5),
        )
        check, seconds = train_sklearn(model, train)
        self.char_model = check
        self.training_seconds = seconds

        labels = np.asarray(train["label"])
        scores = np.asarray([check(text)[1] for text in train["text"]])
        benign = scores[labels == 0]
        attacks = scores[labels == 1]

        if len(benign) and len(attacks):
            self.block_cutoff = float(np.quantile(benign, 0.98))
            self.pass_cutoff = float(np.quantile(attacks, 0.02))
            if self.pass_cutoff > self.block_cutoff:
                midpoint = (self.pass_cutoff + self.block_cutoff) / 2
                self.pass_cutoff = midpoint
                self.block_cutoff = midpoint
        return seconds

    def rule_hits(self, text):
        return [
            name for name, rule in COMPILED_RULES.items()
            if rule.search(text)
        ]

    def classify(self, text):
        start = time.perf_counter()
        elapsed = lambda: (time.perf_counter() - start) * 1000

        # TODO: Add the shared signature/embedding database lookup here.
        # A known attack should block immediately and return before the rule check.

        hits = self.rule_hits(text)
        high_confidence_hits = [
            name for name in hits if name in HIGH_SPECIFICITY_RULES
        ]

        if high_confidence_hits:
            return CascadeResult(text, "block", "rule", 1.0, False, elapsed())

        if self.char_model is None:
            raise RuntimeError("Train the cascade or provide char_model first")

        _, char_score = self.char_model(text)
        char_score = float(char_score)
        guard_used = False

        if char_score >= self.block_cutoff:
            decision = "block"
            stage = "char_tier"
        elif char_score <= self.pass_cutoff:
            decision = "pass"
            stage = "char_tier"
        elif self.guard_check is None:
            decision = "uncertain"
            stage = "guard_not_loaded"
            char_score = char_score
        else:
            guard_answer, guard_score = self.guard_check(text)
            decision = "block" if int(guard_answer) == 1 else "pass"
            stage = "guard"
            guard_used = True
            if guard_score is not None:
                char_score = float(guard_score)
            if decision == "block":
                # TODO: Once the shared database is implemented, store this prompt
                # in it as an attack confirmed by Prompt Guard 2.
                pass

        return CascadeResult(
            text, decision, stage, char_score, guard_used, elapsed()
        )


def load_prompt_guard(device=None):
    if device is None:
        device, _ = pick_device()
    return hf_classifier("meta-llama/Llama-Prompt-Guard-2-86M", device)


def lazy_prompt_guard(device=None):
    guard = None

    def check(text):
        nonlocal guard
        if guard is None:
            guard = load_prompt_guard(device)
        return guard(text)

    check.loaded = lambda: guard is not None
    return check


def evaluate(cascade, prompts):
    rows = []
    for _, row in prompts.iterrows():
        result = cascade.classify(row["text"])
        rows.append({
            "text": row["text"],
            "dataset": row["dataset"],
            "label": int(row["label"]),
            "decision": result.decision,
            "stage": result.stage,
            "score": result.score,
            "guard_used": result.guard_used,
            "milliseconds": result.milliseconds,
        })
    return pd.DataFrame(rows)


def summary(predictions):
    rows = []
    for dataset, group in predictions.groupby("dataset", sort=False):
        rows.append(_summary_row(dataset, group))

    combined = predictions[predictions["dataset"].isin(["deepset", "jackhhao"])]
    if not combined.empty:
        rows.append(_summary_row("all", combined))
    return pd.DataFrame(rows)


def _summary_row(dataset, group):
    attacks = group["label"] == 1
    harmless = group["label"] == 0
    blocked = group["decision"] == "block"
    caught = int((attacks & blocked).sum())
    false_alarms = int((harmless & blocked).sum())
    missed = int((attacks & ~blocked).sum())
    attack_count = int(attacks.sum())
    harmless_count = int(harmless.sum())
    f1 = (
        2 * caught / (2 * caught + false_alarms + missed)
        if 2 * caught + false_alarms + missed
        else float("nan")
    )
    return {
        "dataset": dataset,
        "prompts": len(group),
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "detection_rate": caught / attack_count if attack_count else float("nan"),
        "false_alarm_rate": false_alarms / harmless_count if harmless_count else float("nan"),
        "F1": f1,
        "guard_calls": int(group["guard_used"].sum()),
        "guard_call_rate": group["guard_used"].mean(),
        "mean_ms": group["milliseconds"].mean(),
    }
