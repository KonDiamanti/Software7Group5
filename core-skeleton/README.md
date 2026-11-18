# SW7: cheap prompt check in front of guard models

Skeleton for a shared service that checks a prompt before it reaches a guard model:
cheap checks first, guard model only when they are unsure.

## Idea of the flow (the order is our current assumption)

    prompt -> 1. known attack in the shared database?  -> block
           -> 2. tier 1 (cheap check, not chosen yet)  -> block / pass / unsure
           -> 3. guard model (only if unsure)          -> block / pass

## What is built

- src/result.py: the one answer shape every detector returns (verdict, score, reason)
- src/tier1/cleaning.py: text cleaning and the tamper rule (invisible characters, mixed alphabets)
- src/tier1/language.py: English check (the tier-1 model only knows English)
- src/attack_db/: shared attack database
  - base.py: the shape every database version has to offer (add, lookup, count)
  - hash_db.py: exact match on a hash of the cleaned prompt, stored in sqlite (uses the cleaning from tier1/cleaning.py)
  - embedding_db.py: finds attacks that are close in meaning (near-copies, small edits, leetspeak) with a cosine similarity cutoff (default 0.90), stored in sqlite. It needs sentence-transformers.
- src/router.py: sends a prompt to the guard only when tier 1 is unsure
- src/service.py: the whole flow (database, tier 1, guard). Tier 1 and the guard are passed in, so any detector with a check(text) method fits.
- tests/: unit tests for the databases (test_attack_db.py, 18 tests), the router (test_router.py) and the service (test_service.py), 28 tests in total
- embedding-testing/: experiments and results for the embedding database (see RESULTS.md in that folder)
- requirements.txt: packages (langdetect is optional, sentence-transformers is only needed for the embedding database)

## Not built yet (empty placeholder files)

These wait for decisions in the group (which tier-1 detector, which guard model).

- src/tier1/tfidf_lr.py, src/tier1/train.py: the tier-1 detector and its training
- src/guards/: guard models
- tests/test_tier1.py: tests for tier 1

## Folders

    src/                the code
    tests/              unit tests
    embedding-testing/  experiments and results for the embedding database

## Run (from inside the core-skeleton folder)

    pip install -r requirements.txt
    python -m unittest discover -s tests -t . -v

## Known limits

- The hash database only finds exact repeats (after cleaning). One changed word gives no match.
- The embedding database finds near-copies, not attacks reworded completely. Results and experiments: embedding-testing/RESULTS.md.
- The tier-1 model only knows English, so other languages should go to the guard.
- Open decisions: which tier-1 detector, which guard model(s).