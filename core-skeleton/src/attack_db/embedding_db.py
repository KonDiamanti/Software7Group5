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
#
# quarantine (optional, off by default): a report from an app (source "feedback:<app>") is trusted
# only by the app that made it, until another app reports the same or a close prompt, or the guard
# model blocks it. Then it counts for every app. One wrong report then harms one app, not all of them.

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


FEEDBACK = "feedback:"   # source of an entry that an app reported through /feedback


def default_embedder():
    # imported here so the rest of the project works without sentence-transformers installed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True)


class EmbeddingAttackDB:
    supports_apps = True   # lookup(text, app=...) knows which app asks (needed for the quarantine)

    def __init__(self, path=":memory:", cutoff=0.90, embedder=None, model_name="all-MiniLM-L6-v2",
                 quarantine=False, min_reporters=2):
        # quarantine: a reported attack counts only for the apps that reported it, until `min_reporters`
        # different apps agree on it (same or close prompt), or the guard model blocks it.
        # embedder: function that takes a list of texts and returns one vector per text.
        # tests pass a small fake one, so they need no download.
        self.cutoff = cutoff
        self.embedder = embedder
        self.quarantine = quarantine
        self.min_reporters = min_reporters
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS attacks ("
            "hash TEXT PRIMARY KEY, source TEXT, added REAL, vec BLOB)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        # who reported an entry, and which reported entries are confirmed (only used with the quarantine)
        self.con.execute("CREATE TABLE IF NOT EXISTS reports (hash TEXT, app TEXT, PRIMARY KEY (hash, app))")
        self.con.execute("CREATE TABLE IF NOT EXISTS confirmed (hash TEXT PRIMARY KEY)")
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
        self._keys, self._sources, self._hashes, rows = set(), [], [], []
        for key, source, blob in self.con.execute("SELECT hash, source, vec FROM attacks ORDER BY rowid"):
            self._keys.add(key)
            self._sources.append(source)
            self._hashes.append(key)
            rows.append(np.frombuffer(blob, dtype=np.float32))
        self._mat = np.stack(rows) if rows else None
        self._index = {key: i for i, key in enumerate(self._hashes)}
        reporters = {}
        for key, app in self.con.execute("SELECT hash, app FROM reports"):
            reporters.setdefault(key, set()).add(app)
        confirmed = {row[0] for row in self.con.execute("SELECT hash FROM confirmed")}
        self._reporters = [reporters.get(key, set()) for key in self._hashes]
        # everything that is not a report counts for everybody; a report only after it was confirmed
        self._confirmed = [not (src or "").startswith(FEEDBACK) or key in confirmed
                           for key, src in zip(self._hashes, self._sources)]
        self._by_app = {}
        for i, apps in enumerate(self._reporters):
            for app in apps:
                self._by_app.setdefault(app, set()).add(i)

    def _embed(self, text):
        if self.embedder is None:
            self.embedder = default_embedder()
        vec = np.asarray(self.embedder([normalise(text)])[0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _insert(self, key, source, vec, confirmed):
        self.con.execute("INSERT INTO attacks VALUES (?, ?, ?, ?)",
                         (key, source, time.time(), vec.tobytes()))
        self._keys.add(key)
        self._sources.append(source)
        self._hashes.append(key)
        self._index[key] = len(self._hashes) - 1
        self._reporters.append(set())
        self._confirmed.append(confirmed)
        # simple and fine for thousands of attacks. for millions use an index (for example faiss)
        self._mat = vec[None, :] if self._mat is None else np.vstack([self._mat, vec])
        return len(self._hashes) - 1

    def _confirm(self, i):
        if not self._confirmed[i]:
            self._confirmed[i] = True
            self.con.execute("INSERT OR IGNORE INTO confirmed VALUES (?)", (self._hashes[i],))

    def add(self, text, source=""):
        # the same prompt twice is fine, the second one is ignored
        key = hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()
        i = self._index.get(key)
        report = self.quarantine and source.startswith(FEEDBACK)
        if not report:
            if i is None:
                self._insert(key, source, self._embed(text), True)
            else:
                self._confirm(i)   # for example the guard blocks a prompt that an app reported before
            self.con.commit()
            return
        app = source[len(FEEDBACK):]
        if i is None:
            i = self._insert(key, source, self._embed(text), False)
        self._reporters[i].add(app)
        self._by_app.setdefault(app, set()).add(i)
        self.con.execute("INSERT OR IGNORE INTO reports VALUES (?, ?)", (key, app))
        # the reports that wait and are close to this one: if enough different apps are among them, all are confirmed
        sims = self._mat @ self._mat[i]
        waiting = [int(j) for j in np.flatnonzero(sims >= self.cutoff) if not self._confirmed[j]]
        agreeing = set().union(*(self._reporters[j] for j in waiting)) if waiting else set()
        if len(agreeing) >= self.min_reporters:
            for j in waiting:
                self._confirm(j)
        self.con.commit()

    def nearest(self, text, app=None):
        # (similarity, source) of the closest stored attack that this app may use,
        # or None if there is none. Without the quarantine every attack counts for every app.
        if self._mat is None:
            return None
        sims = self._mat @ self._embed(text)
        if self.quarantine:
            visible = np.array(self._confirmed, dtype=bool)
            for i in self._by_app.get(app, ()):
                visible[i] = True   # the app trusts its own reports
            sims = np.where(visible, sims, -1.0)
        i = int(sims.argmax())
        if sims[i] < -0.5:
            return None
        return float(sims[i]), self._sources[i]

    def lookup(self, text, app=None):
        best = self.nearest(text, app)
        if best is None or best[0] < self.cutoff:
            return None
        similarity, source = best
        return f"{source or 'known attack'} (similarity {similarity:.2f})"

    def count(self):
        return len(self._keys)
