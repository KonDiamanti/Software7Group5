# hash_db.py
# first version of the attack database: exact match on a fingerprint (hash) of the prompt.
# stored in a small sqlite file, so there is nothing extra to install.
#
# good: very fast, and it never blocks something that was not seen before.
# weak: one changed word gives a different hash, so it only finds exact repeats
# (after cleaning). that is why an embedding version might be needed later.

import hashlib
import sqlite3
import time

from ..tier1.cleaning import clean_lower


def fingerprint(text):
    # same cleaning as tier 1, plus squeezing spaces, so tiny formatting changes still match
    norm = " ".join(clean_lower(text).split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class HashAttackDB:
    def __init__(self, path=":memory:"):
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS attacks ("
            "hash TEXT PRIMARY KEY, source TEXT, added REAL)")
        self.con.commit()

    def add(self, text, source=""):
        # same prompt twice is fine, the second one is ignored
        self.con.execute(
            "INSERT OR IGNORE INTO attacks VALUES (?, ?, ?)",
            (fingerprint(text), source, time.time()))
        self.con.commit()

    def lookup(self, text):
        row = self.con.execute(
            "SELECT source FROM attacks WHERE hash = ?", (fingerprint(text),)).fetchone()
        if row is None:
            return None
        return row[0] or "known attack"

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM attacks").fetchone()[0]