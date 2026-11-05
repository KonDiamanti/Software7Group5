"""The classifiers. Each one is a function check(text) that returns 1 (attack) or 0 (harmless),
or a pair (answer, score) with a score between 0 and 1 for how sure it is that the prompt is an attack."""
import collections
import gc
import re
import time

import torch
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def pick_device():
    """The guard LLMs were made for bfloat16. Older GPUs like the T4 don't support it, so they get 32-bit."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
        print("GPU:", torch.cuda.get_device_name(0))
        return "cuda", dtype
    if torch.backends.mps.is_available():
        print("GPU: Mac (mps)")
        return "mps", torch.float32
    print("No GPU, everything runs on the CPU.")
    return "cpu", torch.float32


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Rule-based check: one regular expression per trick, following Liu et al. (2024) and Rao et al. (2024).
# Take new rules from papers or training data, never from the test prompts.
RULES = {
    "ignore instructions": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|guidelines|prompts?|directions|everything)\b",
    "role play":           r"\b(pretend|act as|you are now|from now on,? you|role-?play|stay in character|imagine you are)\b",
    "jailbreak names":     r"\b((?-i:DAN|STAN|DUDE|AIM)|jailbreak|jailbroken|developer mode|do anything now)\b",
    "no limits":           r"\b(no|without|free from|any) (restrictions|limits|limitations|filters|rules|censorship|ethics|morals)\b|\b(unfiltered|uncensored)\b",
    "system prompt":       r"\b(system prompt|initial instructions|hidden instructions|(reveal|print|show) your (instructions|prompt))\b",
    "privilege":           r"\b(sudo|admin mode|god mode|root access)\b",
    "fake system tag":     r"(\[system\]|<\|?system\|?>|#{2,}\s*(system|instruction))",
    "encoded text":        r"[A-Za-z0-9+/]{40,}={0,2}",
    "hidden characters":   r"[​-‏⁠-⁤﻿]",
}
COMPILED_RULES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in RULES.items()}


def rule_check(text):
    return int(any(rule.search(text) for rule in COMPILED_RULES.values()))


# Small trained models from Hugging Face (text classification)
ATTACK_LABELS = {"INJECTION", "JAILBREAK", "MALICIOUS", "LABEL_1"}   # Prompt Guard 2 calls attacks LABEL_1


def hf_classifier(model_id, device):
    classifier = pipeline("text-classification", model=model_id, device=device)

    def check(text):
        labels = classifier(text, top_k=None, truncation=True, max_length=512)
        if labels and isinstance(labels[0], list):   # some transformers versions wrap the list in another list
            labels = labels[0]
        top = max(labels, key=lambda label: label["score"])
        attack_score = sum(label["score"] for label in labels if label["label"].upper() in ATTACK_LABELS)
        return int(top["label"].upper() in ATTACK_LABELS), attack_score

    return check


# Guard LLMs: they write an answer, which we turn into a label. The score is how likely the model was
# to write the attack word instead of the other answer words.
class GuardLLM:
    def __init__(self, model_id, device, dtype):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.odd_answers = []   # answers that were not in the expected format

    def generate(self, inputs, max_new_tokens, **options):
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     output_logits=True, return_dict_in_generate=True, **options)
        return output, output.sequences[0][inputs["input_ids"].shape[1]:]

    def word_score(self, output, tokens, words, attack_word, after=""):
        """Compares the probabilities of the words at the first answer token after the text `after`.
        Returns None if the answer has another format."""
        for step, token in enumerate(tokens):
            piece = self.tokenizer.decode(token)
            if piece.strip() and self.tokenizer.decode(tokens[:step]).rstrip().endswith(after):
                break
        else:
            return None
        space = " " if piece.startswith(" ") else ""
        word_ids = [self.tokenizer.encode(space + word, add_special_tokens=False)[0] for word in words]
        if len(set(word_ids)) < len(words) or token.item() not in word_ids:
            return None
        probabilities = torch.softmax(output.logits[step][0].float(), dim=-1)[word_ids]
        return (probabilities[words.index(attack_word)] / probabilities.sum()).item()


class Qwen3Guard(GuardLLM):
    """Answers "Safety: Safe", "Unsafe" or "Controversial". Only Unsafe counts as an attack."""

    def __init__(self, device, dtype):
        super().__init__("Qwen/Qwen3Guard-Gen-0.6B", device, dtype)

    def __call__(self, text):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        output, tokens = self.generate(inputs, max_new_tokens=128)
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if not re.search(r"Safety: (Safe|Unsafe|Controversial)", answer):
            self.odd_answers.append(answer)
        score = self.word_score(output, tokens, ["Safe", "Unsafe", "Controversial"], "Unsafe", after="Safety:")
        return int("Safety: Unsafe" in answer), score


class LlamaGuard3(GuardLLM):
    """Answers "safe" or "unsafe" plus a harm category. It has no category for jailbreaks or injections."""

    def __init__(self, device, dtype):
        super().__init__("meta-llama/Llama-Guard-3-1B", device, dtype)

    def __call__(self, text):
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", return_dict=True).to(self.device)
        output, tokens = self.generate(inputs, max_new_tokens=10, pad_token_id=0)
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip().lower()
        if not answer.startswith(("safe", "unsafe")):
            self.odd_answers.append(answer)
        score = self.word_score(output, tokens, ["safe", "unsafe"], "unsafe")
        return int(answer.startswith("unsafe")), score


# Learned cheap checks, trained by us on safe-guard. Each train_* function returns (check, training seconds).
WORDS = dict(ngram_range=(1, 2))                                                               # words and word pairs
LETTERS = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, max_features=200000)  # groups of 2-5 letters


def tfidf_models():
    # The SVMs are calibrated so that they give a score (5-fold, so five models run per prompt)
    return {
        "TF-IDF (safe-guard)":  make_pipeline(TfidfVectorizer(**WORDS), LogisticRegression(max_iter=1000)),
        "TF-IDF + SVM":         make_pipeline(TfidfVectorizer(**WORDS), CalibratedClassifierCV(LinearSVC(), cv=5)),
        "TF-IDF + Naive Bayes": make_pipeline(TfidfVectorizer(**WORDS), MultinomialNB()),
        "Char TF-IDF + LR":     make_pipeline(TfidfVectorizer(**LETTERS), LogisticRegression(max_iter=1000)),
        "Char TF-IDF + SVM":    make_pipeline(TfidfVectorizer(**LETTERS), CalibratedClassifierCV(LinearSVC(), cv=5)),
    }


def train_sklearn(model, train):
    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    seconds = time.perf_counter() - start

    def check(text):
        score = model.predict_proba([text])[0][1]
        return int(score >= 0.5), score

    return check, seconds


def train_embedding_lr(train, device):
    """all-MiniLM-L6-v2 turns each prompt into a vector of its meaning, logistic regression learns from those."""
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    start = time.perf_counter()
    vectors = embedder.encode(train["text"], batch_size=64, show_progress_bar=True)
    model = LogisticRegression(max_iter=1000).fit(vectors, train["label"])
    seconds = time.perf_counter() - start

    def check(text):
        score = model.predict_proba(embedder.encode([text]))[0][1]
        return int(score >= 0.5), score

    return check, seconds


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=100, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(torch.cat([h[-2], h[-1]], dim=1))).squeeze(-1)


def train_lstm(train, device, seed=0, epochs=5):
    """A small bidirectional LSTM trained from scratch. A tenth of the data is held out to check the training.
    The result depends a lot on the seed, so compare several seeds before drawing conclusions."""
    torch.manual_seed(seed)
    texts, labels = list(train["text"]), list(train["label"])

    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text.lower())[:256]

    counter = collections.Counter(word for text in texts for word in tokenize(text))
    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(20000))}   # 0 = padding, 1 = unknown word

    def batchify(batch_texts, batch_labels=None):
        seqs = [torch.tensor([vocab.get(word, 1) for word in tokenize(text)] or [1]) for text in batch_texts]
        lengths = torch.tensor([len(seq) for seq in seqs])
        x = pad_sequence(seqs, batch_first=True).to(device)
        y = None if batch_labels is None else torch.tensor(batch_labels, dtype=torch.float32, device=device)
        return x, lengths, y

    train_x, val_x, train_y, val_y = train_test_split(texts, labels, test_size=0.1, random_state=0, stratify=labels)
    lstm = LSTMClassifier(len(vocab) + 2).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    start = time.perf_counter()
    for epoch in range(epochs):
        lstm.train()
        order = torch.randperm(len(train_x)).tolist()
        for i in range(0, len(order), 64):
            batch = order[i:i + 64]
            x, lengths, y = batchify([train_x[j] for j in batch], [train_y[j] for j in batch])
            optimizer.zero_grad()
            loss_fn(lstm(x, lengths), y).backward()
            optimizer.step()
        lstm.eval()
        with torch.no_grad():
            x, lengths, y = batchify(val_x, val_y)
            accuracy = ((lstm(x, lengths) > 0).float() == y).float().mean().item()
        print(f"epoch {epoch + 1}: validation accuracy {accuracy:.3f}")
    seconds = time.perf_counter() - start

    def check(text):
        x, lengths, _ = batchify([text])
        with torch.no_grad():
            score = torch.sigmoid(lstm(x, lengths)).item()
        return int(score >= 0.5), score

    return check, seconds
