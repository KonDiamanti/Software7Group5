"""Three-tier cascade: rules -> char TF-IDF + SVM -> Prompt Guard 2.

1. Rules:      a high-specificity rule hit -> block, otherwise unsure.
2. Char tier:  score >= block_cutoff -> block, score <= pass_cutoff -> pass, otherwise unsure.
3. Guard:      Prompt Guard 2 decides every prompt that is still unsure; its answer is final.

No shared database is used in this version.
"""
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- tier 1
# The rules of the report, read from one place so that all experiments use the same list:
#  - RULES: the nine rules of Table 1 (defined in classifiers.py, read without ML imports)
#  - HIGH_SPECIFICITY_RULES: the three that may block alone (Sections 8.3 and 9.4)
from Linguistic_feature import HIGH_SPECIFICITY_RULES as _PAPER_BLOCKING_RULES
from Linguistic_feature import RULES

# Hard evidence: one hit is enough to block ("no limits", "ignore instructions", "jailbreak names").
HIGH_SPECIFICITY_RULES = set(_PAPER_BLOCKING_RULES)
# Soft evidence (role play, system prompt, ...): also common in harmless prompts, so it never blocks alone.
SOFT_RULES = set(RULES) - HIGH_SPECIFICITY_RULES

# Same patterns and flags as the report (re.I; DAN/STAN/DUDE/AIM stay upper case through (?-i:...)),
# plus re.S so that '.' also matches a newline and "Ignore\nall instructions" is still caught.
_COMPILED = {name: re.compile(p, re.I | re.S) for name, p in RULES.items()}


def normalise(text):
    """Text the rules see: NFKC (full-width letters -> ASCII) and no invisible format
    characters (zero-width space/joiner, soft hyphen, BOM, ...)."""
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


@dataclass
class Result:
    decision: str                  # block | pass | uncertain
    stage: str                     # rule | char_tier | guard
    score: float | None = None     # score of the tier that decided
    guard_used: bool = False
    rule_hits: list = field(default_factory=list)
    char_score: float | None = None
    guard_score: float | None = None
    char_ms: float = 0.0
    guard_ms: float = 0.0


class Cascade:
    def __init__(self, char_model=None, guard_check=None,
                 pass_cutoff=0.20, block_cutoff=0.80,
                 guard_cutoff=0.50, soft_rules_need_guard=False):
        # char_model(text) -> (label, p_attack);  guard_check(text) -> (label, p_attack)
        # Only p_attack is used; the labels are ignored.
        self.char_model = char_model
        self.guard_check = guard_check
        self.guard_cutoff = guard_cutoff
        # True: a soft rule hit (role play, no limits) may not be passed by the char tier
        # and goes to the guard instead.  False (default) follows the plain spec.
        self.soft_rules_need_guard = soft_rules_need_guard
        self.set_cutoffs(pass_cutoff, block_cutoff)

    # ------------------------------------------------------------ cutoffs
    def set_cutoffs(self, pass_cutoff, block_cutoff):
        """Set both cutoffs at once; pass_cutoff must be below block_cutoff."""
        pass_cutoff, block_cutoff = float(pass_cutoff), float(block_cutoff)
        if not pass_cutoff < block_cutoff:     # also False for NaN
            raise ValueError(
                f"pass_cutoff ({pass_cutoff:.4f}) must be < block_cutoff ({block_cutoff:.4f}); "
                "otherwise no prompt is ever unsure and Prompt Guard 2 is never reached.")
        self._pass_cutoff, self._block_cutoff = pass_cutoff, block_cutoff

    @property
    def pass_cutoff(self):
        return self._pass_cutoff

    @pass_cutoff.setter
    def pass_cutoff(self, value):
        self.set_cutoffs(value, self._block_cutoff)

    @property
    def block_cutoff(self):
        return self._block_cutoff

    @block_cutoff.setter
    def block_cutoff(self, value):
        self.set_cutoffs(self._pass_cutoff, value)

    # ------------------------------------------------------------ tier 1
    def rule_hits(self, text):
        # A rule hits if it matches the raw text (as in the report) or the normalised text
        # (so full-width letters and invisible characters cannot hide a phrase).  The raw
        # text is still needed for the "hidden characters" rule, which normalise() would erase.
        raw, clean = str(text), normalise(text)
        return [n for n, rx in _COMPILED.items() if rx.search(raw) or rx.search(clean)]

    def rule_state(self, text):
        """-> ('block'|'unsure', hits).  The rule tier never passes a prompt."""
        hits = self.rule_hits(text)
        return ("block" if set(hits) & HIGH_SPECIFICITY_RULES else "unsure"), hits

    # ------------------------------------------------------------ tier 2
    def char_state(self, score, soft_hit=False):
        if not math.isfinite(score):
            return "unsure"
        if score >= self.block_cutoff:
            return "block"
        if score <= self.pass_cutoff and not (soft_hit and self.soft_rules_need_guard):
            return "pass"
        return "unsure"

    # ------------------------------------------------------------ cascade
    def classify(self, text):
        state, hits = self.rule_state(text)
        if state == "block":
            return Result("block", "rule", 1.0, False, hits)

        t0 = time.perf_counter()
        _, char_score = self.char_model(text)
        char_ms = (time.perf_counter() - t0) * 1000
        char_score = float(char_score)
        state = self.char_state(char_score, bool(set(hits) & SOFT_RULES))
        if state in ("block", "pass"):
            return Result(state, "char_tier", char_score, False, hits, char_score, None, char_ms)

        if self.guard_check is None:           # guard disabled
            return Result("uncertain", "char_tier", char_score, False, hits, char_score, None, char_ms)

        t0 = time.perf_counter()
        _, guard_score = self.guard_check(text)
        guard_ms = (time.perf_counter() - t0) * 1000
        guard_score = float(guard_score)
        decision = "block" if guard_score >= self.guard_cutoff else "pass"
        return Result(decision, "guard", guard_score, True, hits, char_score, guard_score, char_ms, guard_ms)

    # ------------------------------------------------------------ training
    def train(self, train_df, text_col="text", label_col="label", pick=False,
              target_miss=0.02, target_false_alarm=0.02, folds=5, seed=0):
        """Fit char TF-IDF + calibrated SVM on all of train_df.

        pick=False keeps the cutoffs given to the constructor.
        pick=True chooses them with pick_cutoffs() on out-of-fold scores, so the
        cutoffs and the final model come from the same training procedure.
        Returns the training time in seconds."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        t0 = time.perf_counter()
        X, y = train_df[text_col].astype(str), train_df[label_col].astype(int)

        def build():
            return make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
                CalibratedClassifierCV(LinearSVC(), cv=3))

        if pick:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
            scores = cross_val_predict(build(), X, y, cv=cv, method="predict_proba")[:, 1]
            self.set_cutoffs(*pick_cutoffs(scores, y.to_numpy(), target_miss, target_false_alarm))

        model = build().fit(X, y)

        def char_model(text):
            p = float(model.predict_proba([str(text)])[0, 1])
            return int(p >= 0.5), p

        self.char_model = char_model
        return time.perf_counter() - t0


def pick_cutoffs(scores, labels, target_miss=0.02, target_false_alarm=0.02, min_gap=0.05):
    """Pass zone: biggest score range where <= target_miss of passed prompts are attacks.
    Block zone: biggest score range where <= target_false_alarm of blocked prompts are harmless.
    Everything in between goes to Prompt Guard 2.  If no score is safe to pass (block),
    the pass (block) cutoff is -inf (+inf), so nothing is passed (blocked)."""
    if min_gap <= 0:
        raise ValueError("min_gap must be > 0, otherwise the unsure band can be empty.")
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if len(s) == 0 or len(s) != len(y):
        raise ValueError("scores and labels must be non-empty and the same length.")
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    n = len(s)
    # Only cut between two different scores, so tied prompts are never split.
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    first_of_tie = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    attack_frac_below = np.cumsum(y_sorted) / np.arange(1, n + 1)                      # pass all <= s_i
    harmless_frac_above = (np.cumsum((1 - y_sorted)[::-1]) / np.arange(1, n + 1))[::-1]  # block all >= s_i
    ok_pass = np.where((attack_frac_below <= target_miss) & last_of_tie)[0]
    ok_block = np.where((harmless_frac_above <= target_false_alarm) & first_of_tie)[0]
    pass_c = float(s_sorted[ok_pass.max()]) if len(ok_pass) else -math.inf
    block_c = float(s_sorted[ok_block.min()]) if len(ok_block) else math.inf
    if block_c - pass_c < min_gap:             # never let the band collapse
        mid = (pass_c + block_c) / 2
        pass_c, block_c = mid - min_gap / 2, mid + min_gap / 2
    return pass_c, block_c


# ---------------------------------------------------------------- Prompt Guard 2
def lazy_prompt_guard(device, model_id="meta-llama/Llama-Prompt-Guard-2-86M",
                      max_length=512, stride=64, batch_size=16):
    """Returns guard(text) -> (label, p_malicious).  The model loads on the first call.

    Prompt Guard 2 reads at most 512 tokens.  Longer prompts are split into
    overlapping 512-token windows and the highest window score is used, so an attack
    hidden after a long harmless text is still seen.  Index 1 of the output is the
    malicious class (id2label is only LABEL_0/LABEL_1; checked in test_cascade_real_guard.py)."""
    state = {}

    def guard(text):
        if "model" not in state:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            state["torch"] = torch
            state["tok"] = AutoTokenizer.from_pretrained(model_id)
            state["model"] = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        torch, tok, model = state["torch"], state["tok"], state["model"]
        enc = tok(str(text), truncation=True, max_length=max_length, stride=stride,
                  return_overflowing_tokens=True, padding=True, return_tensors="pt")
        enc.pop("overflow_to_sample_mapping", None)
        p = 0.0
        with torch.no_grad():
            for i in range(0, len(enc["input_ids"]), batch_size):
                batch = {k: v[i:i + batch_size].to(device) for k, v in enc.items()}
                probs = torch.softmax(model(**batch).logits, dim=-1)[:, 1]
                p = max(p, float(probs.max()))
        return int(p >= 0.5), p

    return guard


# ---------------------------------------------------------------- evaluation
def evaluate(cascade, data):
    rows = []
    for text, dataset, label in zip(data["text"], data["dataset"], data["label"]):
        t0 = time.perf_counter()
        r = cascade.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(dict(text=text, dataset=dataset, label=label, decision=r.decision, stage=r.stage,
                         score=r.score, guard_used=r.guard_used, milliseconds=ms,
                         rule_hits="|".join(r.rule_hits), char_score=r.char_score,
                         guard_score=r.guard_score, char_milliseconds=r.char_ms,
                         guard_milliseconds=r.guard_ms))
    return pd.DataFrame(rows)


def _metrics(label, decision, guard_used):
    """Undecided ('uncertain') attacks count as missed, never as caught."""
    label, decision = np.asarray(label), np.asarray(decision)
    caught = int(((label == 1) & (decision == "block")).sum())
    missed = int(((label == 1) & (decision != "block")).sum())
    fa = int(((label == 0) & (decision == "block")).sum())
    harmless = int((label == 0).sum())
    dr = caught / max(caught + missed, 1)
    prec = caught / max(caught + fa, 1)
    return dict(prompts=len(label), caught=caught, missed=missed, false_alarms=fa,
                uncertain=int((decision == "uncertain").sum()),
                detection_rate=dr, false_alarm_rate=fa / max(harmless, 1),
                F1=2 * prec * dr / max(prec + dr, 1e-9),
                guard_calls=int(np.sum(guard_used)), guard_call_rate=float(np.mean(guard_used)))


def summary(pred):
    def row(name, d):
        return dict(dataset=name, **_metrics(d.label, d.decision, d.guard_used),
                    mean_ms=float(d.milliseconds.mean()))
    rows = [row(n, g) for n, g in pred.groupby("dataset", sort=False)]
    return pd.DataFrame(rows + [row("all", pred)])


def mistakes(pred, dataset=None, kind="missed"):
    """Wrong answers in evaluate() output.  kind: 'missed' (attacks not blocked, which
    includes undecided ones) or 'false_alarms' (harmless prompts blocked)."""
    d = pred if dataset is None else pred[pred["dataset"] == dataset]
    if kind == "missed":
        d = d[(d["label"] == 1) & (d["decision"] != "block")]
    elif kind == "false_alarms":
        d = d[(d["label"] == 0) & (d["decision"] == "block")]
    else:
        raise ValueError("kind must be 'missed' or 'false_alarms'")
    return d[["dataset", "stage", "decision", "rule_hits", "char_score", "guard_score", "text"]]


def cutoff_sweep(cascade, data, bands=((0.2, 0.8), (0.1, 0.9), (0.05, 0.95), (0.02, 0.98), (0.01, 0.99))):
    """How the unsure band trades guard calls for detection, on all of `data`.

    Scores every prompt once with the char tier and with the guard, then replays the
    cascade for each (pass_cutoff, block_cutoff) band.  Also shows the two ends:
    no guard at all, and the guard for every prompt the rules do not block.
    Use it to report the trade-off, not to pick cutoffs (that would tune on test data)."""
    texts = list(data["text"])
    label = np.asarray(data["label"])
    rule_block = np.array([cascade.rule_state(t)[0] == "block" for t in texts])
    soft = np.array([bool(set(cascade.rule_hits(t)) & SOFT_RULES) for t in texts])
    char = np.array([float(cascade.char_model(t)[1]) for t in texts])
    guard = (np.array([float(cascade.guard_check(t)[1]) for t in texts])
             if cascade.guard_check is not None else None)

    def replay(name, p, b, use_guard=True):
        keep = cascade.soft_rules_need_guard
        passed = (char <= p) & ~(soft & keep)
        unsure = ~rule_block & ~(char >= b) & ~passed
        decision = np.where(rule_block | (char >= b), "block", "pass").astype(object)
        if use_guard and guard is not None:
            decision[unsure] = np.where(guard[unsure] >= cascade.guard_cutoff, "block", "pass")
            used = unsure
        else:
            decision[unsure] = "uncertain"
            used = np.zeros(len(texts), bool)
        return dict(setting=name, pass_cutoff=p, block_cutoff=b, **_metrics(label, decision, used))

    rows = [replay("no guard (char tier at 0.5)", 0.5, 0.5 + 1e-9, use_guard=False)]
    rows += [replay(f"char {p:g} / {b:g}", p, b) for p, b in bands]
    if guard is not None:
        rows.append(replay("guard for every prompt", -math.inf, math.inf))
    return pd.DataFrame(rows)
