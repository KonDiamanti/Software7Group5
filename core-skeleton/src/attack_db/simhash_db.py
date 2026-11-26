# simhash_db.py
# second version of the attack database: a similarity fingerprint (SimHash) next to the exact hash.
#
# how SimHash works, in short:
#   1. cut the cleaned prompt into small overlapping pieces (here: 4 letters, or 3 words)
#   2. give every piece a normal 64-bit hash
#   3. for each of the 64 bit positions, count how many pieces have a 1 there vs a 0
#   4. the fingerprint gets a 1 where the 1s won, and a 0 where the 0s won
# a small edit only changes a few pieces, so only a few votes move and only a few bits flip.
# "how close" two prompts are = how many of the 64 bits differ (Hamming distance).
#
# good: tiny (8 bytes a row), fast to compare, catches typos and small edits.
# weak: it looks at the letters, not the meaning, so a full rewrite of an attack is a new prompt to it.

import hashlib
import sqlite3
import time

import numpy as np

from ..tier1.cleaning import clean_lower

BITS = 64
_SHIFTS = np.arange(BITS, dtype=np.uint64)


def normalise(text):
    return " ".join(clean_lower(text).split())


def exact_hash(text):
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def pieces(text, mode="char", n=4):
    norm = normalise(text)
    units = norm.split() if mode == "word" else norm
    if len(units) <= n:
        return [norm] if norm else []
    joiner = " " if mode == "word" else ""
    return [joiner.join(units[i:i + n]) for i in range(len(units) - n + 1)]


def _hash64(piece):
    return int.from_bytes(hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(text, mode="char", n=4):
    hashes = np.array([_hash64(p) for p in pieces(text, mode, n)], dtype=np.uint64)
    if hashes.size == 0:
        return 0
    bits = (hashes[:, None] >> _SHIFTS) & np.uint64(1)          # one row per piece, one column per bit
    votes = bits.sum(axis=0).astype(np.int64) * 2 - len(hashes)   # +1 for every 1, -1 for every 0
    return int(sum(1 << i for i in range(BITS) if votes[i] > 0))


def _popcount(x):
    # number of 1 bits in each uint64; np.bitwise_count exists from numpy 2.0
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def _to_sql(v):    # sqlite stores signed 64-bit numbers
    return v - (1 << 64) if v >= (1 << 63) else v


def _from_sql(v):
    return v + (1 << 64) if v < 0 else v


class SimHashAttackDB:
    def __init__(self, path=":memory:", max_distance=3, mode="char", n=4):
        self.max_distance = max_distance   # how many of the 64 bits may differ and still count as a match
        self.mode, self.n = mode, n
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS attacks ("
            "hash TEXT PRIMARY KEY, simhash INTEGER, source TEXT, added REAL)")
        self.con.commit()
        rows = self.con.execute("SELECT simhash, source FROM attacks").fetchall()
        # small copy in memory, so a lookup is one numpy pass instead of a query per row
        self._fps = np.array([_from_sql(r[0]) for r in rows], dtype=np.uint64)
        self._sources = [r[1] for r in rows]

    def fingerprint(self, text):
        return simhash(text, self.mode, self.n)

    def add(self, text, source=""):
        fp = self.fingerprint(text)
        cur = self.con.execute(
            "INSERT OR IGNORE INTO attacks VALUES (?, ?, ?, ?)",
            (exact_hash(text), _to_sql(fp), source, time.time()))
        self.con.commit()
        if cur.rowcount:
            self._fps = np.append(self._fps, np.uint64(fp))
            self._sources.append(source)

    def nearest(self, text):
        """(distance in bits, source) of the closest stored attack, or (None, None) when empty"""
        if self._fps.size == 0:
            return None, None
        dist = _popcount(self._fps ^ np.uint64(self.fingerprint(text)))
        i = int(dist.argmin())
        return int(dist[i]), self._sources[i]

    def lookup(self, text):
        # step 2 from the report: exact hash first, almost free
        row = self.con.execute(
            "SELECT source FROM attacks WHERE hash = ?", (exact_hash(text),)).fetchone()
        if row is not None:
            return row[0] or "known attack"
        # step 3: closest fingerprint
        dist, source = self.nearest(text)
        if dist is not None and dist <= self.max_distance:
            return source or "known attack (close variant)"
        return None

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM attacks").fetchone()[0]
