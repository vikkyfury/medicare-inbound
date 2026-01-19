# app/cache.py
from __future__ import annotations
import os, json, time, sqlite3, hashlib, threading
from typing import Any, Optional

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "cache.sqlite3")
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

_lock = threading.Lock()

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    return conn

def _sha1(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_key(namespace: str, payload: dict) -> str:
    return f"{namespace}:{_sha1(payload)}"

def get_cache(key: str, ttl_sec: Optional[float] = None) -> Optional[dict]:
    now = time.time()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute("SELECT v, ts FROM kv WHERE k=?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            v_json, ts = row
            if ttl_sec is not None and (now - ts) > ttl_sec:
                # expired -> delete and miss
                conn.execute("DELETE FROM kv WHERE k=?", (key,))
                conn.commit()
                return None
            return json.loads(v_json)
        finally:
            conn.close()

def set_cache(key: str, value: dict) -> None:
    now = time.time()
    data = json.dumps(value, ensure_ascii=False)
    with _lock:
        conn = _connect()
        try:
            conn.execute("INSERT OR REPLACE INTO kv(k, v, ts) VALUES(?,?,?)", (key, data, now))
            conn.commit()
        finally:
            conn.close()
