# stub.py
# a stand-in guard for the prototype and the demo. It looks the answer up in a table of
# known labels, so it is always right (the same assumption as in the network simulation).
# A real guard model (for example Prompt Guard 2) replaces it later: anything with
# check(text) that returns a Result or True/False fits.

from src.result import BLOCK, PASS, Result


class LookupGuard:
    def __init__(self, labels, default=None):
        # labels: dict from prompt text to True (attack) or False (harmless)
        # default: answer for prompts that are not in the table (None = raise an error)
        self.labels = dict(labels)
        self.default = default
        self.calls = 0   # how many times the guard was asked, this is what the demo counts

    def check(self, text):
        self.calls += 1
        if text in self.labels:
            attack = self.labels[text]
        elif self.default is not None:
            attack = self.default
        else:
            raise KeyError(f"the stub guard does not know this prompt: {text[:60]!r}")
        if attack:
            return Result(BLOCK, None, "guard_block")
        return Result(PASS, None, "guard_pass")
