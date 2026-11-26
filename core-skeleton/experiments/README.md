# Attack database experiment: SHA-256 vs SimHash vs embedding

Which way of storing known attacks should the shared database use?

## Run it

From `core-skeleton`:

```
pip install numpy pandas nltk wordllama
python -m nltk.downloader wordnet         # only needed for make_variants
python -m experiments.make_variants       # builds data/variants_seed0..4.csv (already included)
python -m experiments.compare_attack_db   # writes results/attack_db_comparison.md
```

To use all-MiniLM-L6-v2 instead of WordLlama, `pip install sentence-transformers` on a machine with Hugging Face access. The script picks it up by itself.

## How it works

1. The 586 test prompts from `classifier-testing` are split at random. Half the attacks go into the database. The other half are "unseen".
2. From each stored attack, we make variants: unchanged copy, formatting tricks, typos, synonym swaps, and the attack wrapped inside a harmless message.
3. Each method gets the loosest threshold that blocks none of the `benign_tune` harmless prompts. Those prompts are not used again.
4. We report how much of each variant set gets blocked, how many held-out harmless prompts get blocked (`benign_eval`, plus 20 handwritten `hard_benign` prompts about attacks), speed, storage, and how many words of a stored attack can be guessed back from its fingerprint.
5. Steps 1 to 4 repeat on 5 random splits. The report gives mean ± spread.

## Limits

- Small data: 91 prompts to tune each threshold, and 92 + 20 to test false positives. A 0% false-positive result still allows a real rate of a few %.
- No real paraphrase set. The `unseen` set (different attacks from the same datasets) is the closest thing we have.
- The default embedding (WordLlama) is a simple word-average model. A sentence model like MiniLM should do better on rewrites and worse on word recovery.
- The lookup scans every row. With many rows you'd need an index: bit-band tables for SimHash, or a vector index like FAISS or pgvector for embeddings.
