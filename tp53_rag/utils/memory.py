"""
TP53 RAG Platform — long-term conversation memory.

Persists past Q&A turns to a local SQLite database so a conversation can resume
across app restarts ("doesn't start from zero") instead of living only in
volatile session state. Privacy-first: every stored turn is PII-scrubbed first,
so the memory never leaks identifiers.

Pure + dependency-light: no LLM, no network. All methods degrade gracefully
(never raise) so a memory failure can never crash the app.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Default lightweight PII scrub (used if no scrubber is injected) ─────────
_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b'), "[EMAIL]"),
    (re.compile(r'(?<!\d)(?:\+?254|0)[17]\d{8}\b'), "[PHONE]"),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
    (re.compile(r'\bPT-\d{4}-\d{3,6}\b'), "[PATIENT_ID]"),
    (re.compile(r'\b[A-Z]{2}\d{6,9}\b'), "[PATIENT_ID]"),
    (re.compile(r'\b(patient|pt)\s*#?\s*\d{3,8}\b', re.I), "[PATIENT_ID]"),
]


def _default_scrub(text: str) -> str:
    text = text or ""
    for pat, repl in _PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


class ConversationMemory:
    """SQLite-backed, PII-scrubbed conversation memory.

    Args:
        db_path: SQLite file. Defaults to data/conversation_memory.db.
        scrubber: callable(str)->str applied to every stored field. Defaults to
                  a built-in light PII scrub. Inject the platform's PIIScrubber
                  for stronger coverage.
        max_turns_per_session: hard cap; oldest turns are pruned beyond this.
    """

    def __init__(self,
                 db_path: Optional[str] = None,
                 scrubber: Optional[Callable[[str], str]] = None,
                 max_turns_per_session: int = 200):
        self._db = Path(db_path) if db_path else Path("data/conversation_memory.db")
        self._scrub = scrubber or _default_scrub
        self._cap = max(int(max_turns_per_session), 1)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self._db.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self._db))

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        ts REAL NOT NULL,
                        agent_type TEXT,
                        question TEXT,
                        answer TEXT
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session ON conversation_memory(session_id, ts)"
                )
        except Exception:
            pass  # memory is best-effort; never crash the app

    def remember(self, session_id: str, question: str, answer: str,
                 agent_type: str = "") -> bool:
        """Store one turn (PII-scrubbed). Returns True on success."""
        if not session_id or not (question or answer):
            return False
        q = self._scrub(str(question))
        a = self._scrub(str(answer))
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO conversation_memory (session_id, ts, agent_type, question, answer) "
                    "VALUES (?,?,?,?,?)",
                    (session_id, time.time(), str(agent_type), q, a),
                )
                # prune anything beyond the per-session cap (keep newest)
                conn.execute(
                    """DELETE FROM conversation_memory
                       WHERE session_id = ? AND id NOT IN (
                           SELECT id FROM conversation_memory
                           WHERE session_id = ? ORDER BY ts DESC LIMIT ?
                       )""",
                    (session_id, session_id, self._cap),
                )
            return True
        except Exception:
            return False

    def recent(self, session_id: str, limit: int = 6) -> List[Dict]:
        """Return the most recent turns (oldest→newest) for a session."""
        if not session_id:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT ts, agent_type, question, answer FROM conversation_memory "
                    "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                    (session_id, max(int(limit), 1)),
                ).fetchall()
            rows.reverse()  # chronological
            return [
                {"ts": ts, "agent_type": at, "question": q, "answer": a}
                for ts, at, q, a in rows
            ]
        except Exception:
            return []

    def history_strings(self, session_id: str, limit: int = 6) -> List[str]:
        """Recent turns as flat strings, ready for rag_chain conversation_history."""
        out: List[str] = []
        for t in self.recent(session_id, limit):
            if t["question"]:
                out.append(f"User: {t['question']}")
            if t["answer"]:
                out.append(f"Assistant: {t['answer']}")
        return out

    def rolling_summary(self, session_id: str, max_chars: int = 1200) -> str:
        """A compact text summary of the recent session for prompt injection."""
        turns = self.recent(session_id, limit=12)
        if not turns:
            return ""
        parts = [f"- Q: {t['question']} | A: {t['answer'][:160]}" for t in turns if t["question"]]
        text = "Earlier in this conversation:\n" + "\n".join(parts)
        return text[:max_chars]

    def clear(self, session_id: Optional[str] = None) -> None:
        """Delete one session's memory, or ALL memory if session_id is None."""
        try:
            with self._lock, self._connect() as conn:
                if session_id:
                    conn.execute("DELETE FROM conversation_memory WHERE session_id = ?",
                                 (session_id,))
                else:
                    conn.execute("DELETE FROM conversation_memory")
        except Exception:
            pass

    def stats(self) -> Dict:
        try:
            with self._connect() as conn:
                turns = conn.execute("SELECT COUNT(*) FROM conversation_memory").fetchone()[0]
                sessions = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM conversation_memory"
                ).fetchone()[0]
            return {"turns": turns, "sessions": sessions, "db": str(self._db)}
        except Exception:
            return {"turns": 0, "sessions": 0, "db": str(self._db)}


# Module-level singleton + convenience accessor
_memory_singleton: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    global _memory_singleton
    if _memory_singleton is None:
        _memory_singleton = ConversationMemory()
    return _memory_singleton
