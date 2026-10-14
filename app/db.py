"""SQLite: request log, cache entries, and the corpus version."""
import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, mode TEXT, query TEXT, tenant TEXT,
  cache_hit INTEGER, route TEXT, model TEXT, escalated INTEGER,
  input_tokens INTEGER, output_tokens INTEGER, cost REAL, latency_ms REAL,
  quality REAL, similarity REAL);
CREATE TABLE IF NOT EXISTS cache_entries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tenant TEXT, version TEXT,
  query TEXT, vec BLOB, payload TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


class DB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def get_meta(self, key, default=None):
        rows = self.query("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_meta(self, key, value):
        self.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, value))
