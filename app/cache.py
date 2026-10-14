"""Semantic cache: nearest-neighbour search over past query embeddings.

Vector search is a numpy matrix product (exact, fine up to ~100K entries). To use Qdrant/Redis,
implement the same `lookup` / `store` interface and keep the validity rules (tenant, version, TTL).
"""
import json
import threading
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class Lookup:
    score: float          # best similarity among *valid* entries (even if below threshold)
    payload: dict | None  # set only when score >= threshold
    matched_query: str | None = None


def is_cacheable(result: dict, personalized: bool) -> bool:
    """Never cache errors, empty answers, 'I don't know', or answers that depend on who is asking."""
    if personalized or result.get("error"):
        return False
    ans = (result.get("answer") or "").strip()
    if not ans or "i don't know" in ans.lower():
        return False
    return True


class SemanticCache:
    def __init__(self, db, threshold: float, ttl_seconds: int):
        self.db, self.threshold, self.ttl = db, threshold, ttl_seconds
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        rows = self.db.query("SELECT * FROM cache_entries")
        self.meta = [dict(ts=r["ts"], tenant=r["tenant"], version=r["version"], query=r["query"],
                          payload=json.loads(r["payload"])) for r in rows]
        self.mat = (np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
                    if rows else None)

    def clear(self):
        self.db.execute("DELETE FROM cache_entries")
        self.meta, self.mat = [], None

    def __len__(self):
        return len(self.meta)

    def lookup(self, vec: np.ndarray, tenant: str, version: str) -> Lookup:
        with self.lock:
            if self.mat is None:
                return Lookup(0.0, None)
            now = time.time()
            valid = np.array([m["tenant"] == tenant and m["version"] == version and now - m["ts"] <= self.ttl
                              for m in self.meta])
            if not valid.any():
                return Lookup(0.0, None)
            sims = np.where(valid, self.mat @ vec, -1.0)
            i = int(np.argmax(sims))
            score = float(sims[i])
            if score >= self.threshold:
                return Lookup(score, self.meta[i]["payload"], self.meta[i]["query"])
            return Lookup(score, None, self.meta[i]["query"])

    def store(self, query: str, vec: np.ndarray, payload: dict, tenant: str, version: str):
        now = time.time()
        vec = vec.astype(np.float32)
        self.db.execute(
            "INSERT INTO cache_entries(ts,tenant,version,query,vec,payload) VALUES(?,?,?,?,?,?)",
            (now, tenant, version, query, vec.tobytes(), json.dumps(payload)))
        with self.lock:
            self.meta.append(dict(ts=now, tenant=tenant, version=version, query=query, payload=payload))
            self.mat = vec[None, :] if self.mat is None else np.vstack([self.mat, vec])

    def purge_stale(self, version: str):
        """Housekeeping: drop entries from older corpus versions or past TTL."""
        cutoff = time.time() - self.ttl
        self.db.execute("DELETE FROM cache_entries WHERE version != ? OR ts < ?", (version, cutoff))
        self._load()
