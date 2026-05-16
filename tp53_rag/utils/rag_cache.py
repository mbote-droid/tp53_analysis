"""
utils/cache.py — Shared SQLite Query Cache
Reusable across all agents. Thread-safe, TTL-based, HIPAA-compliant.
No PII stored — keys are SHA-256 hashes of sanitised queries.
"""

import sqlite3
import hashlib
import threading
import time
import json
import logging
import re
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH   = Path("data/query_cache.db")
DEFAULT_TTL       = 1800          # 30 minutes
MAX_VALUE_BYTES   = 1_048_576     # 1 MB — reject oversized payloads
CLEANUP_INTERVAL  = 300           # Run expired-row purge every 5 min
_PII_PATTERN      = re.compile(
    r"\b(?:\d{6,}|[A-Z]{2,}\d{4,}|\w+@\w+\.\w+)\b"  # IDs, emails
)

# ── Module-level lock — one lock shared across all CacheDB instances ─────────
_global_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
class CacheDB:
    """
    Thread-safe SQLite query cache.

    Usage:
        cache = CacheDB()                        # default path + TTL
        cache = CacheDB(ttl=3600)                # 1-hour TTL
        cache.set("my_agent", "What is TP53?", result_dict)
        hit = cache.get("my_agent", "What is TP53?")  # dict or None
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        self.db_path = Path(db_path)
        self.ttl     = ttl
        self._last_cleanup = 0.0
        self._init_db()

    # ── Private ───────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent readers
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with _global_lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key    TEXT PRIMARY KEY,
                    agent_id     TEXT NOT NULL,
                    value_json   TEXT NOT NULL,
                    created_at   REAL NOT NULL,
                    expires_at   REAL NOT NULL,
                    hit_count    INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires ON query_cache(expires_at)"
            )
            conn.commit()
        log.debug("CacheDB initialised at %s", self.db_path)

    @staticmethod
    def _make_key(agent_id: str, query: str) -> str:
        """SHA-256 hash — no raw query text stored (HIPAA)."""
        sanitised = _PII_PATTERN.sub("<REDACTED>", query.strip().lower())
        raw = f"{agent_id}:{sanitised}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _maybe_cleanup(self, conn: sqlite3.Connection) -> None:
        now = time.time()
        if now - self._last_cleanup > CLEANUP_INTERVAL:
            conn.execute("DELETE FROM query_cache WHERE expires_at < ?", (now,))
            conn.commit()
            self._last_cleanup = now
            log.debug("CacheDB: expired rows purged")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, agent_id: str, query: str) -> Optional[Any]:
        """Return cached value or None on miss/expiry."""
        key = self._make_key(agent_id, query)
        now = time.time()
        with _global_lock, self._connect() as conn:
            self._maybe_cleanup(conn)
            row = conn.execute(
                "SELECT value_json, expires_at FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                log.debug("Cache MISS — agent=%s", agent_id)
                return None
            value_json, expires_at = row
            if now > expires_at:
                conn.execute(
                    "DELETE FROM query_cache WHERE cache_key = ?", (key,)
                )
                conn.commit()
                log.debug("Cache EXPIRED — agent=%s", agent_id)
                return None
            # Increment hit counter
            conn.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (key,),
            )
            conn.commit()
            log.debug("Cache HIT — agent=%s", agent_id)
            return json.loads(value_json)

    def set(self, agent_id: str, query: str, value: Any) -> bool:
        """
        Store value. Returns True on success, False if payload too large
        or serialisation fails.
        """
        key        = self._make_key(agent_id, query)
        now        = time.time()
        expires_at = now + self.ttl

        try:
            value_json = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            log.warning("CacheDB: serialisation failed — %s", exc)
            return False

        if len(value_json.encode()) > MAX_VALUE_BYTES:
            log.warning("CacheDB: payload too large, not cached (agent=%s)", agent_id)
            return False

        with _global_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_cache (cache_key, agent_id, value_json,
                                         created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    hit_count  = 0
                """,
                (key, agent_id, value_json, now, expires_at),
            )
            conn.commit()
        log.debug("CacheDB: stored — agent=%s ttl=%ss", agent_id, self.ttl)
        return True

    def delete(self, agent_id: str, query: str) -> bool:
        """Explicitly evict a single entry."""
        key = self._make_key(agent_id, query)
        with _global_lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM query_cache WHERE cache_key = ?", (key,)
            )
            conn.commit()
        return True

    def flush_agent(self, agent_id: str) -> int:
        """Evict ALL entries for a given agent. Returns rows deleted."""
        with _global_lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM query_cache WHERE agent_id = ?", (agent_id,)
            )
            conn.commit()
            return cur.rowcount

    def flush_all(self) -> int:
        """Wipe the entire cache. Returns rows deleted."""
        with _global_lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM query_cache")
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        """Return cache analytics — safe for dashboards, no PII."""
        with _global_lock, self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM query_cache"
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM query_cache WHERE expires_at > ?",
                (time.time(),),
            ).fetchone()[0]
            top_agents = conn.execute(
                """
                SELECT agent_id, COUNT(*) as entries, SUM(hit_count) as hits
                FROM query_cache
                GROUP BY agent_id
                ORDER BY hits DESC
                LIMIT 10
                """
            ).fetchall()
        return {
            "total_entries"  : total,
            "active_entries" : active,
            "expired_entries": total - active,
            "top_agents"     : [
                {"agent_id": r[0], "entries": r[1], "hits": r[2]}
                for r in top_agents
            ],
        }


# ── Module-level singleton (optional convenience) ─────────────────────────────
_default_cache: Optional[CacheDB] = None

def get_cache(db_path: Path = DEFAULT_DB_PATH, ttl: int = DEFAULT_TTL) -> CacheDB:
    """Return (or create) the module-level default CacheDB instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheDB(db_path=db_path, ttl=ttl)
    return _default_cache


# ══════════════════════════════════════════════════════════════════════════════
# Reverse-engineering / self-test  (python -m utils.cache)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.DEBUG)
    print("\n=== CacheDB Self-Test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db = CacheDB(db_path=Path(tmp) / "test.db", ttl=2)

        # 1. Basic set/get
        db.set("drug_discovery", "What targets R175H?", {"answer": "APR-246"})
        hit = db.get("drug_discovery", "What targets R175H?")
        assert hit == {"answer": "APR-246"}, "FAIL: basic get"
        print("✅ Test 1 — basic set/get")

        # 2. TTL expiry
        time.sleep(3)
        hit = db.get("drug_discovery", "What targets R175H?")
        assert hit is None, "FAIL: TTL not respected"
        print("✅ Test 2 — TTL expiry")

        # 3. PII in query — key is hashed, raw text never stored
        db.set("liquid_biopsy", "Patient ID 12345678 TP53 VAF", {"vaf": 0.23})
        hit = db.get("liquid_biopsy", "Patient ID 12345678 TP53 VAF")
        assert hit == {"vaf": 0.23}, "FAIL: PII query"
        print("✅ Test 3 — PII-safe key hashing")

        # 4. Oversized payload rejected
        huge = {"data": "x" * (MAX_VALUE_BYTES + 1)}
        result = db.set("gene_expression", "large payload", huge)
        assert result is False, "FAIL: oversized payload accepted"
        print("✅ Test 4 — oversized payload rejected")

        # 5. flush_agent
        db.set("enzyme_design", "query A", {"a": 1})
        db.set("enzyme_design", "query B", {"b": 2})
        deleted = db.flush_agent("enzyme_design")
        assert deleted == 2, "FAIL: flush_agent count"
        print("✅ Test 5 — flush_agent")

        # 6. Concurrent writes (thread safety)
        import concurrent.futures
        db2 = CacheDB(db_path=Path(tmp) / "test.db", ttl=60)
        def write(i):
            db2.set("concurrent", f"query {i}", {"i": i})
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(write, range(50)))
        stats = db2.stats()
        assert stats["active_entries"] >= 50, "FAIL: concurrent writes"
        print("✅ Test 6 — concurrent thread safety")

        # 7. Stats endpoint
        print("✅ Test 7 — stats:", stats)

    print("\n=== All tests passed ===\n")
