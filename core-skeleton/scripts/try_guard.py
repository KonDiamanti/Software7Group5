"""Try a real guard on the test prompts of deepset and jackhhao, before using it in the demo.

    python -m scripts.try_guard --guard protectai          # no login needed
    python -m scripts.try_guard --guard prompt-guard-2     # needs login and approved access

Prints how many attacks the guard catches, how many harmless prompts it blocks, and the time per prompt.
"""

import argparse
import time

from src.guards.hf_guard import MODELS, HFGuard
from src.result import BLOCK
from src.tier1.train import load_test_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", choices=sorted(MODELS), default="protectai")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    guard = HFGuard(MODELS[args.guard], device=args.device)
    for name, (texts, labels) in load_test_sets().items():
        caught = blocked = attacks = harmless = 0
        start = time.perf_counter()
        for text, label in zip(texts, labels):
            is_block = guard.check(text).verdict == BLOCK
            if label == 1:
                attacks += 1
                caught += is_block
            else:
                harmless += 1
                blocked += is_block
        seconds = time.perf_counter() - start
        print(f"{name}: caught {caught} of {attacks} attacks, blocked {blocked} of {harmless} harmless prompts, "
              f"{1000 * seconds / len(texts):.0f} ms per prompt")


if __name__ == "__main__":
    main()
