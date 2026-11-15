# learning.py
# tier 1 that keeps learning from the guard. Every answer of the guard is a new labelled
# prompt. After every `retrain_every` answers, the model is trained again on the base
# training data plus everything the guard has answered. The cutoffs stay the same.
# In our network simulation this was where most of the saved guard calls came from.
#
# The retraining takes a few seconds with the full training data (about 20 s), so a
# service would do it in the background. Here it simply happens inside learn().

from src.result import BLOCK
from src.tier1.char_svm import CharSvmTier1, make_model


class LearningTier1:
    def __init__(self, tier1, base_texts, base_labels, retrain_every=100):
        self.tier1 = tier1
        self.base_texts = list(base_texts)
        self.base_labels = list(base_labels)
        self.new_texts = []
        self.new_labels = []
        self.retrain_every = retrain_every
        self.retrains = 0

    def check(self, text):
        return self.tier1.check(text)

    def learn(self, text, is_attack):
        self.new_texts.append(text)
        self.new_labels.append(int(is_attack))
        if len(self.new_texts) % self.retrain_every == 0:
            self.retrain()

    def on_guard_answer(self, text, result):
        # this is what PromptService calls after every guard answer
        self.learn(text, result.verdict == BLOCK)

    def retrain(self):
        model = make_model().fit(self.base_texts + self.new_texts, self.base_labels + self.new_labels)
        self.tier1 = CharSvmTier1(model, self.tier1.pass_at, self.tier1.block_at)
        self.retrains += 1
