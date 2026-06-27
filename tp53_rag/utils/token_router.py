"""
============================================================
Precision Onco Africa - Token-Efficient Router
utils/token_router.py
============================================================
Most AI systems optimise for accuracy. This one also optimises for *cost*: it
decides, before any expensive call, the cheapest route that can still answer a
query correctly —

    cache hit            → 0 LLM tokens   (reuse a prior answer)
    deterministic agent  → 0 LLM tokens   (curated/rule-based, no generation)
    LLM generation       → full cost      (only when genuinely needed)

The router tracks how many LLM tokens it avoided and the rough cost saved, so
the saving is a measured number, not a claim. The decision logic is pure and
unit-tested; the running totals live on a small stateful tracker.

Token accounting uses a simple, transparent heuristic (~4 chars/token) and an
env-configurable price — it is meant to demonstrate the optimisation, not to
be a billing-grade meter.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

ROUTE_CACHE = "cache"
ROUTE_DETERMINISTIC = "deterministic"
ROUTE_LLM = "llm"

# Typical cost of one avoided LLM round-trip (prompt + response), in tokens.
# Deliberately conservative; override via env for your own model.
AVG_PROMPT_TOKENS = int(os.getenv("ROUTER_AVG_PROMPT_TOKENS", 1500))
AVG_RESPONSE_TOKENS = int(os.getenv("ROUTER_AVG_RESPONSE_TOKENS", 400))
# USD per 1K tokens (blended). Default ~ a small hosted model.
USD_PER_1K_TOKENS = float(os.getenv("ROUTER_USD_PER_1K", 0.20))

# Query intents that a deterministic agent can answer with NO LLM call.
_DETERMINISTIC_PATTERNS: Dict[str, List[str]] = {
    "tumor_board": ["tumour board", "tumor board", "consensus", "debate",
                    "panel", "multidisciplinary"],
    "explainability": ["why", "explain", "evidence", "rationale", "justify"],
    "classification": ["classify", "is it pathogenic", "pathogenic?", "hotspot",
                       "vus", "significance", "contact mutant", "conformational"],
    "atlas": ["african", "africa", "regional", "prevalence", "command center",
              "continental", "burden"],
    "structure": ["alphafold", "plddt", "structure", "domain", "needle plot"],
}

# A bare protein change (e.g. "R175H", "p.R248Q") is classifiable deterministically.
_VARIANT_RE = re.compile(r"\b[p\.]?[A-Z]\d{1,3}[A-Z\*]\b")


def estimate_tokens(text: str) -> int:
    """Transparent ~4-chars-per-token estimate. Never negative."""
    return max(0, len(str(text or "")) // 4)


def decide_route(query: str, cache_hit: bool = False,
                 force_llm: bool = False) -> Dict:
    """Pure routing decision for a query. Returns the route, a human reason,
    and the estimated LLM tokens avoided by not generating."""
    q = str(query or "").strip().lower()
    avoided = AVG_PROMPT_TOKENS + AVG_RESPONSE_TOKENS

    if force_llm:
        return {"route": ROUTE_LLM, "reason": "caller forced LLM generation",
                "tokens_saved": 0}
    if cache_hit:
        return {"route": ROUTE_CACHE,
                "reason": "semantic cache hit — reused a prior answer",
                "tokens_saved": avoided}
    if not q:
        return {"route": ROUTE_LLM, "reason": "empty query — defer to LLM",
                "tokens_saved": 0}

    # Deterministic-answerable?
    for intent, kws in _DETERMINISTIC_PATTERNS.items():
        if any(kw in q for kw in kws):
            return {"route": ROUTE_DETERMINISTIC,
                    "reason": f"answerable by the {intent} agent without generation",
                    "tokens_saved": avoided}
    if _VARIANT_RE.search(str(query or "")):
        return {"route": ROUTE_DETERMINISTIC,
                "reason": "bare variant — classified deterministically",
                "tokens_saved": avoided}

    return {"route": ROUTE_LLM,
            "reason": "open-ended query needs grounded generation",
            "tokens_saved": 0}


@dataclass
class RouterStats:
    queries: int = 0
    cache: int = 0
    deterministic: int = 0
    llm: int = 0
    tokens_saved: int = 0
    history: List[Dict] = field(default_factory=list)


class TokenRouter:
    """Stateful router that records decisions and accumulates savings."""

    def __init__(self) -> None:
        self._stats = RouterStats()
        self._lock = threading.Lock()

    def route(self, query: str, cache_hit: bool = False,
              force_llm: bool = False) -> Dict:
        decision = decide_route(query, cache_hit=cache_hit, force_llm=force_llm)
        with self._lock:
            self._stats.queries += 1
            self._stats.tokens_saved += decision["tokens_saved"]
            if decision["route"] == ROUTE_CACHE:
                self._stats.cache += 1
            elif decision["route"] == ROUTE_DETERMINISTIC:
                self._stats.deterministic += 1
            else:
                self._stats.llm += 1
            self._stats.history.append(
                {"q": str(query)[:80], **decision})
            self._stats.history = self._stats.history[-100:]  # cap
        return decision

    def report(self) -> Dict:
        with self._lock:
            s = self._stats
            avoided_calls = s.cache + s.deterministic
            usd = round(s.tokens_saved / 1000.0 * USD_PER_1K_TOKENS, 4)
            pct = round(avoided_calls / s.queries * 100, 1) if s.queries else 0.0
            return {
                **asdict(s),
                "llm_calls_avoided": avoided_calls,
                "pct_avoided": pct,
                "usd_saved_est": usd,
                "price_per_1k": USD_PER_1K_TOKENS,
            }

    def reset(self) -> None:
        with self._lock:
            self._stats = RouterStats()


_router = TokenRouter()


def get_router() -> TokenRouter:
    return _router
