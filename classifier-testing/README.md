# Classifier testing

We test how well existing classifiers and our own cheap checks detect jailbreaks and prompt injections, all on the same labelled prompts. The results and what we found are in [RESULTS.md](RESULTS.md), and the numbers go into the section "Testing Existing Classifiers" of our report.

| File | What it is |
|---|---|
| `S7G5_Classifiers.ipynb` | Runs everything and makes the tables. Start here. |
| `prompt_data.py` | Loads the test prompts, and the safe-guard training data without near copies of test prompts. |
| `classifiers.py` | The twelve classifiers. |
| `benchmark.py` | Runs a classifier on every prompt and makes the result tables. |
| `our_prompts.csv` | Our own test prompts. |
| `results/` | The CSVs of our latest run. A new run overwrites them. |
| `RESULTS.md` | The results, what we found, and a log of all our runs. |

## Data

| Dataset | Used for | Attacks | Harmless |
|---|---|---|---|
| `deepset/prompt-injections` (test split) | testing: prompt injections, more than a third in German | 60 | 56 |
| `jackhhao/jailbreak-classification` (test split) | testing: jailbreaks, many of them long role play | 139 | 123 |
| `hackaprompt/hackaprompt-dataset` (200 random attacks that worked) | testing: prompt injections | 200 | 0 |
| `our_prompts.csv` | testing: our own prompts, only examples so far | 4 | 4 |
| `xTRam1/safe-guard-prompt-injection` (train split) | training the learned cheap checks | | |

## Setup

You need Python 3.10 or newer and VS Code with the Python and Jupyter extensions. In a terminal in this folder, run once:

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

`hf auth login` asks for a Hugging Face token (Settings > Access Tokens on huggingface.co, "Read" is enough). Nothing shows while you paste it. Prompt Guard 2, Llama Guard 3 and HackAPrompt also need access: request it on the pages of [Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) and [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-1B), and accept the terms of [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset). Without access these three are skipped.

With an NVIDIA GPU, install the GPU version of torch afterwards (the command for your CUDA version is on [pytorch.org](https://pytorch.org/get-started/locally/)).

## Running

Open the notebook, click Select Kernel > Python Environments > `.venv`, and click Run All. On a GPU a run takes about 15 to 20 minutes, on a CPU about an hour, mostly for the two guard LLMs.

The notebook reloads the `.py` files by itself, so after changing one you only have to run the cells again.

## Adding a classifier

A classifier is a function that gets one prompt and returns 1 (attack) or 0 (harmless), or a pair `(answer, score)` with a score from 0 to 1. Put it in `classifiers.py` and add a cell to section 3 of the notebook:

```python
bench.run("My classifier", my_check)
```

It then shows up in every table and in the Overleaf rows. A Hugging Face text-classification model needs only one line:

```python
bench.run("My model", hf_classifier("name/of-the-model", DEVICE))
```

If the model calls attacks something else than `INJECTION`, `JAILBREAK`, `MALICIOUS` or `LABEL_1`, add its label to `ATTACK_LABELS` in `classifiers.py`.

## Adding our own prompts

Add a row to `our_prompts.csv` with the prompt and 1 for an attack or 0 for a harmless prompt. If the prompt has a comma, a quote mark or a line break, put it in double quotes and write each quote mark inside it twice (`""`). They are scored as their own dataset, `ours`.

## Before commit

- Clear the notebook outputs (the `...` menu at the top of the notebook > Clear All Outputs), so the file stays small and the changes are easy to read.
- Never commit your Hugging Face token or the `.venv` folder.
