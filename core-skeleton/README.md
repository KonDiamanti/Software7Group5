# SW7: cheap prompt check in front of guard models

Project skeleton for a shared service that checks a prompt before it reaches a guard model:
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
  - hash_db.py: works, exact match on a hash of the cleaned prompt, stored in sqlite (uses the cleaning from tier1/cleaning.py)
  - embedding_db.py: placeholder only, hash or embedding is still undecided
- tests/test_attack_db.py: tests for the database (8 tests)
- requirements.txt: packages (so far only langdetect is used, and it is optional; the others are for the tier-1 model later)

## Not built yet (empty placeholder files)

These wait for decisions in the group (which tier-1 detector, which guard model).

- src/tier1/tfidf_lr.py, src/tier1/train.py: the tier-1 detector and its training
- src/router.py: sends a prompt to the guard only when tier 1 is unsure
- src/service.py: the whole flow (database, tier 1, guard)
- src/guards/: guard models
- tests/test_tier1.py, tests/test_service.py: tests for those parts

## Folders

    src/          the code
    tests/        unit tests

## Run (from inside the core-skeleton folder)

    pip install -r requirements.txt
    python -m unittest discover -s tests -t . -v

## Known limits

- The hash database only finds exact repeats (after cleaning). One changed word gives no match.
- The tier-1 model only knows English, so other languages should go to the guard.
- Open decisions: which tier-1 detector, which guard model(s), hash or embedding database.