# embedding_db_wordllama.py
# used by experiments/compare_attack_db.py (the service itself uses embedding_db.py).
#
# third version of the attack database: an embedding (a list of numbers for the meaning) next to the exact hash.
#
# a small language model turns the cleaned prompt into a vector. prompts with a similar meaning
# point in a similar direction, so "how close" = cosine similarity (1 = same direction, 0 = unrelated).
#
# good: can catch rewrites that keep the meaning but change the words.
# weak: bigger rows (hundreds of numbers), needs a model for every lookup, and the vector says a lot
#       about the original words, so it is weaker as "stored hashed" privacy.
#
# model: all-MiniLM-L6-v2 if sentence-transformers is installed, else WordLlama
# (its weights ship inside the pip package, so it works without Hugging Face access).

import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from .simhash_db import exact_hash, normalise


class _MiniLM:
    name = "all-MiniLM-L6-v2 (sentence-transformers)"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class _WordLlama:
    name = "WordLlama l2_supercat 256-d (bundled weights)"

    def __init__(self):
        import wordllama
        from wordllama import WordLlama
        # the package ships its tokenizer but looks for it in the cache folder, so copy it there once
        pkg = Path(wordllama.__file__).parent / "tokenizers"
        cache = Path(WordLlama.get_file_path("tokenizer", None))
        cache.mkdir(parents=True, exist_ok=True)
        for f in pkg.glob("*.json"):
            if not (cache / f.name).exists():
                shutil.copy(f, cache / f.name)
        self.model = WordLlama.load()

    def encode(self, texts):
        v = np.asarray(self.model.embed(list(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def load_default_model(prefer="minilm"):
    if prefer == "minilm":
        try:
            return _MiniLM()
        except Exception:
            pass
    return _WordLlama()


class EmbeddingAttackDB:
    def __init__(self, path=":memory:", min_similarity=0.9, model=None):
        self.min_similarity = min_similarity   # cosine similarity from which a prompt counts as a known attack
        self.model = model or load_default_model()
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS attacks ("
            "hash TEXT PRIMARY KEY, embedding BLOB, source TEXT, added REAL)")
        self.con.commit()
        rows = self.con.execute("SELECT embedding, source FROM attacks").fetchall()
        dim = len(self.embed("x"))
        self._vecs = (np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
                      if rows else np.zeros((0, dim), dtype=np.float32))
        self._sources = [r[1] for r in rows]

    def embed(self, text):
        return self.model.encode([normalise(text)])[0]

    def add(self, text, source=""):
        v = self.embed(text)
        cur = self.con.execute(
            "INSERT OR IGNORE INTO attacks VALUES (?, ?, ?, ?)",
            (exact_hash(text), v.tobytes(), source, time.time()))
        self.con.commit()
        if cur.rowcount:
            self._vecs = np.vstack([self._vecs, v[None, :]])
            self._sources.append(source)

    def nearest(self, text):
        """(cosine similarity, source) of the closest stored attack, or (None, None) when empty"""
        if len(self._vecs) == 0:
            return None, None
        sims = self._vecs @ self.embed(text)
        i = int(sims.argmax())
        return float(sims[i]), self._sources[i]

    def lookup(self, text):
        row = self.con.execute(
            "SELECT source FROM attacks WHERE hash = ?", (exact_hash(text),)).fetchone()
        if row is not None:
            return row[0] or "known attack"
        sim, source = self.nearest(text)
        if sim is not None and sim >= self.min_similarity:
            return source or "known attack (close meaning)"
        return None

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM attacks").fetchone()[0]
