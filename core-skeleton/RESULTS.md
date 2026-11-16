# Results: tier 1 and the demo

Numbers from our runs on 15 and 16 November 2026 (laptop, CPU only). The raw output of every run is in [`results/`](results/), copied from the terminal, with the command on the first line. [README.md](README.md) says how to run them.

Terms: **guard calls** are counted per 1,000 prompts (without any cheap check it is 1,000). **Saved by sharing** is the guard calls of the apps without sharing minus the guard calls with sharing, on the same traffic in every run, so it is a paired difference. **Missed attacks** are attacks that were passed, **harmless blocked** are harmless prompts that were blocked.

## Tier 1 (`python -m src.tier1.train --more-data`)

Char TF-IDF + SVM, the same settings as in `classifier-testing`. It is trained on the safe-guard train split (without 63 near copies of test prompts) and on half of the deepset and jackhhao train splits. The other half sets the two cutoffs, on English prompts only. The test splits of deepset and jackhhao are never used for training.

- **Cutoffs:** pass at a score of 0.033 or lower, block at 0.420 or higher, everything in between goes to the guard. They were set on 627 prompts the model had not seen.
- **jackhhao:** 129 of 139 attacks blocked, 0 passed, 10 sent to the guard. 1 of 123 harmless prompts blocked, 58 sent to the guard.
- **deepset:** 18 of 60 attacks blocked, 7 passed, 35 sent to the guard (most of the German prompts go to the guard because the model only knows English). 0 of 56 harmless prompts blocked, 31 sent to the guard.
- **The price of being careful:** tier 1 sends many harmless prompts to the guard (47% of the harmless jackhhao prompts, 55% of the harmless deepset prompts). This is what drives most of the guard calls in the demo.
- **First version was too optimistic:** with the cutoffs set on safe-guard prompts, both cutoffs ended at 0.549, because safe-guard is easy for this model. On deepset it then passed 33 of 60 attacks (with `--more-data` and cutoffs from deepset and jackhhao prompts: 7). See `results/tier1_train_safeguard_only.txt`.
- **Not comparable with `classifier-testing/RESULTS.md`:** that table trains on safe-guard only. Here part of deepset and jackhhao is also used for training.

## Demo (`python -m scripts.demo`)

Apps send prompts to the service through the API. The traffic comes only from the test splits of deepset and jackhhao (199 attacks, 179 harmless prompts). Every app sends 100 prompts, 20% attacks. A repeat is an attack that some app has sent before: a third exactly the same, two thirds edited by a script (filler text, typos or dropped words). The shared memory is the embedding database (cutoff 0.90). Every run is made with sharing off (each app has its own database and tier 1) and on (one database and one tier 1), on the same traffic.

- **One attack through three apps (part 1):** with sharing, the guard is asked once. The second app is blocked by memory, and so is a slightly edited copy sent to the third app. Without sharing, the guard is asked three times.
- **Sharing saves guard calls, and the saving grows with the number of apps and with tier 1 learning:**
  - 3 apps, tier 1 fixed, 30% repeats (5 runs): 387.3 without sharing, 383.3 with sharing. Saved 4.0 (std 4.3), fewer in 3 of 5 runs.
  - 3 apps, tier 1 fixed, 70% repeats (10 runs): 427.7 and 417.7. Saved 10.0 (std 7.4), fewer in 8 of 10 runs.
  - 3 apps, tier 1 learns, 30% repeats (3 runs): 387.8 and 372.2. Saved 15.6 (std 10.7), fewer in 3 of 3 runs.
  - 10 apps, tier 1 fixed, 30% repeats (5 runs): 421.4 and 411.8. Saved 9.6 (std 2.2), fewer in 5 of 5 runs.
  - 10 apps, tier 1 learns, 30% repeats (3 runs): 415.0 and 370.7. Saved 44.3 (std 1.5), about 11%, fewer in 3 of 3 runs.
- **The memory alone saves little, because tier 1 already decides about 80% of the repeats by itself.** With 10 apps and a fixed tier 1, 17% of the exact repeats and 22% of the edited repeats still reached the guard without sharing. With sharing it was 0% and 9%.
- **Most of the saving comes from tier 1 learning from the guard answers of all apps.** With learning and 10 apps, edited repeats reaching the guard fell from 20% to 5% and the total guard calls by about 11%.
- **This agrees with the cascade simulation** (`embedding-testing/RESULTS.md`): memory alone saved about 11 calls per 1,000 at 10 apps, and the saving with tier 1 learning was larger.
- **Mistakes did not change clearly.** Missed attacks were 3.0% without sharing and 3.5% with sharing at 10 apps with learning (3.3% and 3.2% with a fixed tier 1). Harmless blocked went from 0.4% to 0.6%. With 3 runs these differences are about one attack per run, so we cannot separate them from noise.
- **Most guard calls are not about attacks.** About 37% to 43% of all prompts reach the guard, most of them harmless prompts that tier 1 is unsure about. Sharing attack memory cannot change that.

## Real guard (Prompt Guard 2)

The wrapper `src/guards/hf_guard.py` runs Prompt Guard 2 (86M) instead of the stub. It runs on the laptop CPU.

- **Alone on the test sets** (`python -m scripts.try_guard --guard prompt-guard-2`): jackhhao 130 of 139 attacks caught, 1 of 123 harmless prompts blocked. deepset 8 of 60 attacks caught, 0 of 56 harmless blocked. These are the same numbers as in `classifier-testing/RESULTS.md` (93.5% and 0.8% on jackhhao, 13.3% and 0.0% on deepset). It needs 249 ms per prompt on deepset and 761 ms on jackhhao (longer prompts) on our CPU, against 33 ms in the table (a GPU).
- **In the demo with 10 apps, tier 1 fixed, 30% repeats, 3 runs:** sharing saved only 1.3 guard calls per 1,000 prompts (std 1.2, fewer in 2 of 3 runs), against 9.6 with the stub guard. 20.0% of the attacks were missed in both conditions (3.3% with the stub). Exact repeats decided by memory with sharing: 7% (with the stub: 20%).
- **The reason:** the shared database only stores what the guard blocks. Prompt Guard 2 passes most of the deepset injections, so those attacks are never stored and the guard sees them again when they come back. The network effect is only as good as the guard's detection.
- **With feedback** (`--feedback 0.5`, the apps report half of the attacks that were passed through `/feedback`; on average 15.7 reports per run): 419.0 guard calls without sharing, 412.3 with sharing. Saved 6.7 (std 1.2), fewer in 3 of 3 runs. Exact repeats decided by memory 18% with sharing (3% without), edited repeats 10% (1%).
- **Sharing now also lowers the missed attacks:** 19.7% without sharing, 17.2% with sharing (means of 3 runs). With the always-right stub there was nothing to protect, with an imperfect guard a report from one app protects the others. About 17% are still missed, because the guard's weakness on injections is larger than what the reports fix. Harmless blocked went from 0.4% to 0.6%.
- **The reporting rate of 50% is our assumption** about how often an app notices an attack that got through.

## Limits

- Most runs use a stand-in guard that is always right. The runs with Prompt Guard 2 are only 3 runs each, with a fixed tier 1 (not with tier 1 learning).
- The repeats in the traffic are made by us (a third exact copies, two thirds script-edited), not real attacker behaviour. Only about 6% of the prompts are repeats.
- Small scale: 100 prompts per app, 3 to 10 runs. The spread between runs is large for the small settings, so read those as a direction. Only the 10-app runs have a small spread.
- The runs with tier 1 learning are only 3 runs each. The retraining happens after every 20 guard answers, so an app that is alone (no sharing) retrains on its own few answers.
- The harmless prompts are drawn with replacement from 179 prompts, so the same harmless prompt can reach several apps.
- Tier 1 was trained on part of deepset and jackhhao, so the traffic is not unseen for the kind of prompts, only for the exact prompts.
- The cutoffs come from one split of 627 prompts and are not tuned further.
