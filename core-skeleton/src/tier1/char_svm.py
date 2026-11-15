# char_svm.py
# tier 1: character TF-IDF + SVM. This was the best cheap check in
# classifier-testing/RESULTS.md (93.5% of the jackhhao jailbreaks, 1 false alarm, about 6 ms).
# The SVM gives a score from 0 to 1. Two cutoffs turn it into three answers:
#   score <= pass_at   -> PASS   (almost sure it is harmless)
#   score >= block_at  -> BLOCK  (almost sure it is an attack)
#   in between         -> UNSURE (the guard decides)

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.result import BLOCK, PASS, UNSURE, Result
from src.tier1.cleaning import suspicious
from src.tier1.language import is_english

# same settings as LETTERS in classifier-testing/classifiers.py (groups of 2 to 5 letters)
LETTERS = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, max_features=200000)


def make_model():
    # the SVM is calibrated so that it gives a score (5-fold)
    return make_pipeline(TfidfVectorizer(**LETTERS), CalibratedClassifierCV(LinearSVC(), cv=5))


def choose_cutoffs(scores, labels, max_error=0.02, min_count=20):
    """Pick pass_at and block_at from scores of prompts the model has not seen.

    pass_at: the highest score up to which at most max_error of the prompts are attacks.
    block_at: the lowest score from which at most max_error of the prompts are harmless.
    min_count stops a handful of prompts from deciding the cutoff.
    If the model cannot reach max_error on one side, that side is switched off
    (pass_at = -1 or block_at = 2), so everything on that side goes to the guard.
    """
    pairs = sorted(zip(scores, labels))
    n = len(pairs)

    pass_at = -1.0
    attacks = 0
    for i, (score, label) in enumerate(pairs):
        attacks += label
        if i + 1 >= min_count and attacks / (i + 1) <= max_error:
            pass_at = score

    block_at = 2.0
    harmless = 0
    for i in range(n - 1, -1, -1):
        score, label = pairs[i]
        harmless += 1 - label
        if n - i >= min_count and harmless / (n - i) <= max_error:
            block_at = score

    if pass_at >= block_at:   # the two zones must not overlap
        middle = (pass_at + block_at) / 2
        pass_at, block_at = middle - 1e-6, middle + 1e-6
    return pass_at, block_at


class CharSvmTier1:
    def __init__(self, model, pass_at, block_at):
        self.model = model
        self.pass_at = pass_at
        self.block_at = block_at

    def score(self, text):
        return float(self.model.predict_proba([text])[0][1])

    def check(self, text):
        if not text.strip():
            return Result(UNSURE, None, "empty")
        if suspicious(text):   # invisible characters or mixed alphabets: let the guard look
            return Result(UNSURE, None, "tamper")
        if not is_english(text):   # the model only knows English
            return Result(UNSURE, None, "non-english")

        score = self.score(text)
        if score >= self.block_at:
            return Result(BLOCK, score, "score")
        if score <= self.pass_at:
            return Result(PASS, score, "score")
        return Result(UNSURE, score, "score")

    def save(self, path):
        joblib.dump({"model": self.model, "pass_at": self.pass_at, "block_at": self.block_at}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        return cls(data["model"], data["pass_at"], data["block_at"])
