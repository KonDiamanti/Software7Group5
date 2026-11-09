# embedding_db.py
# second version of the attack database: finds attacks that are close in meaning,
# not only exact repeats. it has the same three methods as HashAttackDB
# (add, lookup, count), so the rest of the project does not change.
#
# how it works: every attack is turned into a vector (an embedding). a new prompt is
# a "known attack" when its closest stored attack is at least `cutoff` similar.
#
# what our tests showed (see experiments):
#   - finds small edits of a known attack (other words in a few places, extra text,
#     typos, leetspeak) that the hash version misses completely
#   - does NOT find an attack that is reworded completely in other words
#   - cutoff 0.90 wrongly matched about 0.5% of harmless prompts in our test data
#     (mostly a harmless question that an attack extends with an injected sentence)

import hashlib
import re
import sqlite3
import time

import numpy as np

from ..tier1.cleaning import clean_lower

# curly quotes -> straight quotes
QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
# 1gn0r3 -> ignore (only used inside words that also have letters)
LEET_BACK = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalise(text):
    # same cleaning as tier 1, then quotes and leetspeak, so these tricks do not hide an attack
    text = clean_lower(text).translate(QUOTES)
    words = [w.translate(LEET_BACK) if re.search(r"[a-z]", w) else w for w in text.split()]
    return " ".join(words)


def default_embedder():
    # imported here so the rest of the project works without sentence-transformers installed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True)


class EmbeddingAttackDB:
    def __init__(self, path=":memory:", cutoff=0.90, embedder=None, model_name="all-MiniLM-L6-v2"):
        # embedder: function that takes a list of texts and returns one vector per text.
        # tests pass a small fake one, so they need no download.
        self.cutoff = cutoff
        self.embedder = embedder
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS attacks ("
            "hash TEXT PRIMARY KEY, source TEXT, added REAL, vec BLOB)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.con.commit()

        # vectors from one model mean nothing to another model, so a file remembers its model
        row = self.con.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
        if row is None:
            self.con.execute("INSERT INTO meta VALUES ('model', ?)", (model_name,))
            self.con.commit()
        elif row[0] != model_name:
            self.con.close()  # otherwise Windows keeps the file locked
            raise ValueError(f"this database was made with {row[0]}, not {model_name}")

        # load what is already stored
        self._keys, self._sources, rows = set(), [], []
        for key, source, blob in self.con.execute("SELECT hash, source, vec FROM attacks ORDER BY rowid"):
            self._keys.add(key)
            self._sources.append(source)
            rows.append(np.frombuffer(blob, dtype=np.float32))
        self._mat = np.stack(rows) if rows else None

    def _embed(self, text):
        if self.embedder is None:
            self.embedder = default_embedder()
        vec = np.asarray(self.embedder([normalise(text)])[0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def add(self, text, source=""):
        # the same prompt twice is fine, the second one is ignored
        key = hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()
        if key in self._keys:
            return
        vec = self._embed(text)
        self.con.execute("INSERT INTO attacks VALUES (?, ?, ?, ?)",
                         (key, source, time.time(), vec.tobytes()))
        self.con.commit()
        self._keys.add(key)
        self._sources.append(source)
        # simple and fine for thousands of attacks. for millions use an index (for example faiss)
        self._mat = vec[None, :] if self._mat is None else np.vstack([self._mat, vec])

    def nearest(self, text):
        # (similarity, source) of the closest stored attack, or None if the database is empty
        if self._mat is None:
            return None
        sims = self._mat @ self._embed(text)
        i = int(sims.argmax())
        return float(sims[i]), self._sources[i]

    def lookup(self, text):
        best = self.nearest(text)
        if best is None or best[0] < self.cutoff:
            return None
        similarity, source = best
        return f"{source or 'known attack'} (similarity {similarity:.2f})"

    def count(self):
        return len(self._keys)