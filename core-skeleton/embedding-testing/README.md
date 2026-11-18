# Embedding database tests

The experiments behind the embedding attack database in `src/attack_db/embedding_db.py`. They compare it with the hash version and measure the network effect. The results are in [RESULTS.md](RESULTS.md), the raw outputs in [results/](results/).

## Scripts

- `repeat_rate.py`: how many attacks have a near-copy in the data, hash vs embedding at several cutoffs (about 1 min)
- `show_examples.py`: example pairs the embedding finds and the hash misses, and harmless prompts it wrongly matches (about 1 min)
- `paraphrase_test.py`: known attacks edited by a script (synonyms, typos, filler, leetspeak, ...), are they still recognised? (1-2 min)
- `real_paraphrases.py`: hand-written rewordings, harmless look-alikes, and the leetspeak fix (1-2 min)
- `compare_models.py`: three embedding models, memory vs classifier (several minutes)
- `learning_curve.py`: does the service catch more unseen attacks when more apps share? (a few minutes)
- `cascade_sim.py`: the whole cascade with N apps, counting guard calls, missed attacks and harmless prompts blocked (a few minutes)
- `paraphrase_set.py`: not run on its own, it holds the hand-written attacks, rewordings and look-alikes the scripts use

## Run (from inside the core-skeleton folder)

    pip install -r requirements.txt
    python embedding-testing\repeat_rate.py

The first run downloads the datasets and the embedding models from Hugging Face. The scripts save embeddings in `embedding-testing\cache` (not committed), so later runs are faster.