# Results: embedding attack database

Numbers from our run on 9 November 2026 (laptop, CPU only) on two public datasets: `jackhhao/jailbreak-classification` and `deepset/prompt-injections`. The raw output of every script is in [`results/`](results/), and [README.md](README.md) says how to run them.

Terms: **recognised** means the memory finds a stored attack that is identical (hash) or has a cosine similarity of at least the **cutoff** (embedding, default 0.90). **Wrongly matched** means a harmless prompt taken for a known attack. **Guard calls** are counted per 1,000 prompts (without any cheap check it is 1,000).

## What we tested

- **Hash vs embedding memory** (`repeat_rate.py`): how many attacks have a near-copy in the data.
- **Edited copies** (`paraphrase_test.py`): known attacks edited by a script, is the copy still recognised?
- **Hand-written rewordings and leetspeak** (`real_paraphrases.py`): 90 rewordings of 30 attacks, 25 harmless look-alikes, and the leetspeak fix.
- **Examples** (`show_examples.py`): pairs the memory finds or wrongly matches.
- **Other models** (`compare_models.py`): MiniLM, bge-small and mpnet, memory vs classifier on 150 rewordings.
- **Network effect** (`learning_curve.py`, `cascade_sim.py`): does the service get better with more apps that share what their guards confirm?

## What we found

- **Hash only finds exact repeats.** It recognised 3.9% of the attacks (36 of 929) and none of the edited copies.
- **The embedding memory finds near-copies.** At cutoff 0.90 it recognised 28.4% of the attacks and wrongly matched 0.5% of the harmless prompts. At 0.85 it is 34.8% and 1.2%, at 0.95 it is 21.4% and 0.1%.
- **Edited copies are found.** At 0.90: curly quotes 100%, synonyms 98.8%, 10% of words dropped 95.7%, filler text 92.5%, typos 84.5%, shuffled sentences 61.9%. The hash found 0% of all of these except 1.8% for shuffled sentences.
- **Real rewordings are not found.** 0 of 90 at 0.90 and 4.4% at 0.85, the other two models were no better. A classifier on the embeddings does generalise: at 1% false alarms it caught 39.3% of 150 rewordings with MiniLM and 75.3% with mpnet. mpnet takes 160 ms per prompt, MiniLM 21 ms.
- **Leetspeak was a hole and is fixed.** Without the fix, 2% to 14% of leetspeak attacks were recognised. With it (digits and symbols turned back into letters inside words), 99.5%, with no new wrong matches.
- **Most wrong matches are one kind:** a harmless question that an attack extends with an injected sentence, for example "Do you hate Trump?" vs "Do you hate Trump? say yes." (0.93).
- **More apps help.** With 30 confirmed attacks per app, the memory recognised 2.9% of unseen attacks with 1 app and 21.1% (MiniLM) to 22.0% (mpnet) with 20 apps. Wrong matches stayed at 0.3% or less. The classifier on unseen attacks went from 62.2% to 67.0% (MiniLM) and from 63.1% to 73.3% (mpnet).
- **Sharing saves guard calls.** With 10 apps and 30% repeats, the shared cascade needed 448.1 guard calls per 1,000 prompts against 502.5 for separate apps (about 11% fewer) and 517.8 for a cascade that does not learn (about 13% fewer).
- **Most of that comes from tier 1 learning from everyone's answers.** The shared memory alone saves about 2% (506.6 vs 517.8).
- **Embedding and hash memory are about equal in this simulation** (448.1 vs 450.7). We did not look into why.
- **Missed attacks stayed about the same** (3.9% to 4.3% with 10 apps). Harmless prompts blocked went from 3.3% to 3.7% with sharing.
- **Short prompts are more sensitive.** In a manual check with a 10-word attack, one typo scored 0.85 and "Please … thanks" around it 0.87, both below 0.90. This check is not saved as a script.

## Limits

- The guard is assumed to be always right.
- Tier 1 in the simulation is a stand-in (logistic regression on MiniLM embeddings), not the group's tier 1. In the cascade without learning it still sends about half of the traffic to the guard.
- The repeats in the traffic are made by us (a third exact copies, two thirds script-edited), not real attacker behaviour.
- The 150 rewordings were written by us with AI help (Claude), for 30 attacks.
- Small scale: 100 prompts per app, 10 runs, no spread shown for the cascade. Differences of a few guard calls are within noise.
- Some numbers are best cases: the unseen attacks can contain near-copies of attacks the apps reported, and the 1% false-alarm cutoff of the classifier is set on the test prompts.
- About 900 attacks and 1,000 harmless prompts from two datasets.