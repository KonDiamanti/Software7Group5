"""Loads the test prompts and the training data. Every prompt gets a label: 1 = attack, 0 = harmless."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUR_PROMPTS = Path(__file__).with_name("our_prompts.csv")
NEAR_COPY = 0.9   # cosine similarity of TF-IDF vectors from which two prompts count as near copies


def load_test_prompts(hackaprompt_sample=200):
    """deepset and jackhhao (test splits), a sample of HackAPrompt attacks that worked, and our own prompts."""
    sets = [
        pd.DataFrame([
            {"dataset": "deepset", "text": row["text"], "label": int(row["label"])}
            for row in load_dataset("deepset/prompt-injections", split="test")
        ]),
        pd.DataFrame([
            {"dataset": "jackhhao", "text": row["prompt"], "label": int(row["type"] == "jailbreak")}
            for row in load_dataset("jackhhao/jailbreak-classification", split="test")
        ]),
    ]
    try:
        sets.append(load_hackaprompt(hackaprompt_sample))
    except Exception as error:
        print("Skipped HackAPrompt. It needs a Hugging Face login and the dataset's terms accepted.")
        print("Error:", repr(error))
    ours = pd.read_csv(OUR_PROMPTS)
    sets.append(ours.assign(dataset="ours")[["dataset", "text", "label"]])
    return pd.concat(sets, ignore_index=True)


def load_hackaprompt(sample):
    hackaprompt = (load_dataset("hackaprompt/hackaprompt-dataset", split="train")
                   .select_columns(["user_input", "correct", "level"]).to_pandas())
    worked = hackaprompt[(hackaprompt["correct"].astype(str).str.lower() == "true") & hackaprompt["user_input"].notna()]
    worked = worked.assign(user_input=worked["user_input"].str.strip())
    worked = worked[worked["user_input"] != ""].drop_duplicates("user_input")
    picked = worked.sample(n=sample, random_state=0)
    print(f"HackAPrompt: picked {len(picked)} of the {len(worked)} different attacks that worked.")
    return pd.DataFrame({"dataset": "hackaprompt", "text": picked["user_input"], "label": 1})


def load_training_data(test_texts):
    """The safe-guard train split, without the prompts that are near copies of a test prompt."""
    safe_guard = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    before = len(safe_guard)
    safe_guard = remove_near_copies(safe_guard, test_texts)
    print(f"safe-guard: removed {before - len(safe_guard)} near copies of test prompts, {len(safe_guard)} prompts left.")
    return safe_guard


def remove_near_copies(train, test_texts):
    """Repeats until no near copy is left, because removing prompts changes the TF-IDF weights a little."""
    while True:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(train["text"])
        closest = cosine_similarity(vectorizer.transform(train["text"]), vectorizer.transform(test_texts)).max(axis=1)
        keep = [i for i, similarity in enumerate(closest) if similarity < NEAR_COPY]
        if len(keep) == len(train):
            return train
        train = train.select(keep)
