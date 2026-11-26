# SW7: cheap prompt check in front of guard models

A shared service that checks a prompt before it reaches a guard model:
cheap checks first, guard model only when they are unsure.

## Flow

    prompt -> 1. known attack in the shared database?        -> block
           -> 2. tier 1 (Char TF-IDF + SVM)                  -> block / pass / unsure
           -> 3. guard model (only if unsure)                -> block / pass
           -> 4. learn: a guard-confirmed attack goes into the shared database,
                 every guard answer can teach tier 1

Apps talk to the service through a small API. With sharing on, all apps use one attack database and one tier 1, so an attack caught for one app is known to all of them.

## What is built

- src/result.py: the one answer shape every detector returns (verdict, score, reason)
- src/tier1/cleaning.py: text cleaning and the tamper rule (invisible characters, mixed alphabets)
- src/tier1/language.py: English check (the tier-1 model only knows English)
- src/tier1/char_svm.py: tier 1, character TF-IDF + SVM (the best cheap check in `classifier-testing/RESULTS.md`). Two cutoffs turn its score into block, pass or unsure. Empty, tampered and non-English prompts go to the guard.
- src/tier1/train.py: trains tier 1, picks the cutoffs on prompts the model has not seen, saves the model to `models/` (not committed)
- src/tier1/learning.py: tier 1 that retrains after every N guard answers
- src/attack_db/: shared attack database
  - base.py: the shape every database version has to offer (add, lookup, count)
  - hash_db.py: exact match on a hash of the cleaned prompt, stored in sqlite
  - embedding_db.py: finds attacks that are close in meaning (near-copies, small edits, leetspeak) with a cosine similarity cutoff (default 0.90), stored in sqlite. It needs sentence-transformers.
- src/guards/stub.py: a stand-in guard that looks the answer up in a table of known labels, so it is always right (the default in the demo)
- src/guards/hf_guard.py: a real guard, a small trained classifier from Hugging Face (Prompt Guard 2 or the ProtectAI model). Same `check(text)` as the stub. Prompt Guard 2 needs a Hugging Face login and approved access, ProtectAI needs neither. The answer for a text is remembered, so the demo does not run the model twice for the same text.
- src/router.py: sends a prompt to the guard only when tier 1 is unsure
- src/service.py: the whole flow for one app (database, tier 1, guard). After a guard answer it stores confirmed attacks in the database and calls an optional `on_guard_answer(text, result)`.
- src/api.py: the web layer (FastAPI), see below
- scripts/demo.py: apps using the service through the API, with and without sharing
- scripts/try_guard.py: tries a real guard alone on the deepset and jackhhao test prompts
- scripts/: import_attacks.py and network_simulation.py (the first network effect experiment, hash database only)
- embedding-testing/: experiments and results for the embedding database (see RESULTS.md in that folder)
- tests/: 81 unit tests
- RESULTS.md: results for tier 1 and the demo, raw outputs in `results/`

## API

    POST /check     {"app": "chatbot", "text": "..."}   ->  {"verdict": "block", "reason": "known_attack", "score": 1.0, "decided_by": "memory"}
    POST /feedback  {"app": "chatbot", "text": "..."}   ->  stores a missed attack in the database (and teaches tier 1)
    GET  /stats     how each app's prompts were decided (memory, tier1, guard)
    GET  /health    status, number of apps, number of stored attacks

`decided_by` is `memory`, `tier1`, `guard` or `error`. If the check fails (for example the guard is down), the service blocks the prompt (fail closed) and logs the error.

`create_app(Services(make_db, make_tier1, make_guard, share=True))` builds the API. `share=False` gives every app its own database and tier 1. The endpoints are async on purpose: sqlite connections may only be used in the thread that made them.

## Not built yet

- The group has not chosen the real guard yet. The wrapper works for Prompt Guard 2 and ProtectAI. Qwen3Guard and Llama Guard 3 (in `classifier-testing/classifiers.py`) would need their own wrapper.
- A file that starts the real service (embedding database, trained tier 1, real guard) as a web server. Only the demo and the tests build the API for now.
- Decisions about what the shared database stores: in the prototype it stores the prompt text.

## Run (from inside the core-skeleton folder)

    pip install -r requirements.txt
    python -m unittest discover -s tests -t . -v

Train tier 1 (needs internet once, about a minute) and run the demo:

    python -m src.tier1.train --more-data
    python -m scripts.demo
    python -m scripts.demo --apps 10
    python -m scripts.demo --apps 10 --learn --retrain-every 20 --runs 3

Try a real guard (Prompt Guard 2 needs `hf auth login` and approved access first), then use it in the demo:

    python -m scripts.try_guard --guard prompt-guard-2
    python -m scripts.demo --apps 10 --guard prompt-guard-2
    python -m scripts.demo --apps 10 --guard prompt-guard-2 --feedback 0.5

`--more-data` also trains on the train splits of deepset and jackhhao. The model is saved to `models\tier1_char_svm_more_data.joblib`, and the demo expects that file. `--learn` lets tier 1 learn from the guard's answers, which makes the demo slow because it retrains. `--memory hash` uses the exact-match database instead of the embedding database. `--feedback 0.5` lets the apps report half of the attacks that were passed through `/feedback`.

## Known limits

- By default the guard in the demo is a stand-in that is always right. Real guards make mistakes, and the shared database only stores what the guard blocks, so the network effect depends on how many attacks the guard catches (see RESULTS.md).
- The hash database only finds exact repeats (after cleaning). One changed word gives no match.
- The embedding database finds near-copies, not attacks reworded completely. Results and experiments: embedding-testing/RESULTS.md.
- The tier-1 model only knows English, so other languages go to the guard.
- The tier-1 cutoffs are set on one split of about 600 prompts, so they are a bit noisy.
- The trained model is not in git. Everyone has to train it once.
- Tier 1 retrains inside the request that triggers it. A real service would do that in the background.
- Results and limits of the demo: RESULTS.md.
