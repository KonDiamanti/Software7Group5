# hf_guard.py
# a real guard: a small trained classifier from Hugging Face (for example Prompt Guard 2
# or the ProtectAI model). It has the same check(text) method as the stub guard, so it
# can replace it anywhere. It follows hf_classifier in classifier-testing/classifiers.py.
#
# Prompt Guard 2 needs a Hugging Face login and approved access to the model
# (https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M). The ProtectAI model needs neither.

from src.result import BLOCK, PASS, Result

MODELS = {
    "prompt-guard-2": "meta-llama/Llama-Prompt-Guard-2-86M",
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
}

# Prompt Guard 2 calls the attack label LABEL_1
ATTACK_LABELS = {"INJECTION", "JAILBREAK", "MALICIOUS", "LABEL_1"}


class HFGuard:
    def __init__(self, model_id=MODELS["prompt-guard-2"], device="cpu", classifier=None):
        # classifier: a function like a transformers text-classification pipeline.
        # tests pass a fake one, so they need no download.
        if classifier is None:
            from transformers import pipeline   # imported here so the rest works without transformers
            classifier = pipeline("text-classification", model=model_id, device=device)
        self.classifier = classifier
        self.calls = 0          # how often the guard was asked
        self.model_runs = 0     # how often the model really ran (the answer for a text is remembered)
        self._answers = {}

    def check(self, text):
        self.calls += 1
        if text not in self._answers:
            self._answers[text] = self._run_model(text)
        return self._answers[text]

    def _run_model(self, text):
        self.model_runs += 1
        labels = self.classifier(text, top_k=None, truncation=True, max_length=512)
        if labels and isinstance(labels[0], list):   # some versions wrap the list in another list
            labels = labels[0]
        top = max(labels, key=lambda label: label["score"])
        attack_score = sum(label["score"] for label in labels if label["label"].upper() in ATTACK_LABELS)
        if top["label"].upper() in ATTACK_LABELS:
            return Result(BLOCK, attack_score, "guard_block")
        return Result(PASS, attack_score, "guard_pass")
