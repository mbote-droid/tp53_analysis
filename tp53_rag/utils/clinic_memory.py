"""
============================================================
Precision Onco Africa - Clinic Memory (Epistemic Override / Doctor Loop)
utils/clinic_memory.py
============================================================
When a clinician corrects an answer, that correction is saved locally and
injected as HIGH-PRIORITY context into future prompts for related questions —
so the system "learns" from this specific clinic's expert without any model
weight update, retraining, or online fine-tuning. It is few-shot personalisation
("clinic memory"), stored as plain JSON you can read and delete.

Honest by construction: no weights change; the corrections are transparent text
you can inspect/clear; and they are injected as *context*, not baked into the
model. Research use only.

Pure Python + a JSON file (written to a writable temp path so it survives on
read-only cloud filesystems). Never raises — degrades to in-memory only if the
disk is unavailable.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

DEFAULT_PATH = Path(tempfile.gettempdir()) / "poa_clinic_memory.json"

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


class ClinicMemory:
    """JSON-backed store of clinician corrections + relevance retrieval."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self._items: List[Dict] = self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> List[Dict]:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
        except Exception as e:  # pragma: no cover
            log.warning(f"Clinic memory load failed: {e}")
        return []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._items, indent=2),
                                 encoding="utf-8")
        except Exception as e:  # pragma: no cover
            log.warning(f"Clinic memory save failed (in-memory only): {e}")

    # ── mutations ────────────────────────────────────────────────
    def add(self, question: str, correction: str,
            tags: Optional[List[str]] = None) -> Dict:
        """Record a correction. Returns the stored entry. Never raises."""
        entry = {
            "question": str(question or "").strip(),
            "correction": str(correction or "").strip(),
            "tags": [str(t) for t in (tags or [])],
            "ts": time.time(),
        }
        if not entry["correction"]:
            return {"ok": False, "reason": "empty correction"}
        self._items.append(entry)
        self._save()
        return {"ok": True, **entry}

    def clear(self) -> int:
        n = len(self._items)
        self._items = []
        self._save()
        return n

    # ── retrieval ────────────────────────────────────────────────
    def all(self, limit: Optional[int] = None) -> List[Dict]:
        items = sorted(self._items, key=lambda e: e.get("ts", 0), reverse=True)
        return items[:limit] if limit else items

    def relevant(self, query: str, limit: int = 3) -> List[Dict]:
        """Corrections most relevant to `query` by token overlap; falls back to
        most-recent when nothing overlaps. Never raises."""
        if not self._items:
            return []
        q = _tokens(query)
        scored = []
        for e in self._items:
            overlap = len(q & (_tokens(e.get("question", "")) |
                               {t.lower() for t in e.get("tags", [])}))
            scored.append((overlap, e.get("ts", 0), e))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        hits = [e for ov, _, e in scored if ov > 0][:limit]
        if hits:
            return hits
        return self.all(limit)  # no overlap → most recent as gentle priors

    def as_prompt_block(self, query: str, limit: int = 3) -> str:
        """A high-priority context block for prompt injection. '' if none."""
        hits = self.relevant(query, limit)
        if not hits:
            return ""
        lines = ["CLINICIAN CORRECTIONS (treat as high-priority ground truth "
                 "that overrides general knowledge for this deployment):"]
        for e in hits:
            q = e.get("question", "").strip()
            c = e.get("correction", "").strip()
            lines.append(f"- On \"{q[:120]}\": {c}")
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {"corrections": len(self._items), "path": str(self.path)}
