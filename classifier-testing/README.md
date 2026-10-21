# Classifier testing

Before we build our own cheap check for jailbreaks and prompt injections, we test how well existing classifiers do on the same labelled prompts. This folder has the notebook for that test and the results we use in our report. It is part of our SW7 project at AAU Copenhagen (group S7G5).

## What's in here

| File | What it is |
|---|---|
| `S7G5_Classifiers.ipynb` | The notebook. It loads the data, runs the six classifiers and makes the tables for the report. |
| `results/` | The results of our Colab run on 20 October 2026, which are the numbers in the report. A new run of the notebook overwrites them. |
| `requirements.txt` | The Python libraries the notebook needs. |

## Running it

The full guide, for Google Colab and for your own computer, is at the top of the notebook (section 0). On your own computer it comes down to this:

Windows:
```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\hf auth login
```

Mac or Linux:
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf auth login
```

Then open the notebook in VS Code, pick the `.venv` kernel and click Run All. Prompt Guard 2, Llama Guard 3 and the HackAPrompt dataset need a Hugging Face account with access to them. Without access the notebook still runs, but skips those three.

## Results

Detection rate (Det.) and false alarm rate (FA) in %, from the Colab run on a Tesla T4. F1 and time are over deepset and jackhhao together.

| Classifier | deepset Det. | deepset FA | jackhhao Det. | jackhhao FA | HackAPrompt Det. | F1 | ms per prompt |
|---|---|---|---|---|---|---|---|
| Rule-based check | 16.7 | 0.0 | 84.9 | 14.6 | 18.5 | 0.74 | 2.4 |
| ProtectAI DeBERTa v2 | 36.7 | 0.0 | 84.2* | 1.6* | 99.5 | 0.82* | 71.7 |
| Prompt Guard 2 (86M) | 13.3 | 0.0 | 93.5 | 0.8 | 88.5 | 0.82 | 36.7 |
| Qwen3Guard (0.6B) | 13.3 | 1.8 | 80.6 | 1.6 | 2.0 | 0.75 | 630.7 |
| Llama Guard 3 (1B) | 18.3 | 1.8 | 48.2 | 8.1 | 20.5 | 0.54 | 443.8 |
| TF-IDF (safe-guard) | 3.3 | 0.0 | 84.9† | 0.8† | 8.5 | 0.75† | 1.3 |

\* ProtectAI was trained on jackhhao. † TF-IDF's training data has near copies of 40 jackhhao jailbreaks. Without them it catches 80.8%.

In short: Prompt Guard 2 is the best jailbreak detector. All six classifiers miss most of the deepset injections, but the two injection models catch almost all HackAPrompt attacks, so the problem is the kind of injections in deepset and not prompt injections in general. The guard LLMs judge harm and not tricks, and the cheap checks only know the words they were built or trained on. Section 7 of the notebook explains each result.

## Before you commit

Clear the outputs first (in VS Code: the `...` menu at the top of the notebook > Clear All Outputs), so the file stays small and the changes are easy to read. Never commit a Hugging Face token.
