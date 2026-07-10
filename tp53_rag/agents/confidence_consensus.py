"""
============================================================
Precision Onco Africa - Agent-Confidence Consensus
agents/confidence_consensus.py
============================================================
Instead of six specialists debating in prose, each one returns a *probability
distribution* over a fixed set of management options, and the board's consensus
is the aggregate of those distributions — a mathematical vote you can graph,
rather than a paragraph you have to read.

Honest framing:
  * These are **model-reported confidences** (the model's own stated
    probabilities), NOT token-level logprobs — the account's models are all
    reasoning/verbose and don't emit a clean single vote token, so we ask for a
    JSON distribution instead. This is labelled as such everywhere; it is
    decision-support (RUO), not a calibrated clinical probability.
  * Each specialist samples at its own orthogonal temperature (reusing
    agents/orthogonal_personas), so the six votes are genuinely independent,
    not an echo chamber.
  * A malformed model reply degrades to a flagged uniform vote — never raises.

The LLM is injected (a backend, or a generate_fn factory), so this is fully
unit-testable without a live model.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import log

CONSENSUS_DISCLAIMER = ("Model-reported confidence distribution (each agent's "
                        "own stated probabilities), aggregated — not token "
                        "logprobs and not a calibrated clinical probability. "
                        "Research use only.")

# Fixed option set the agents vote over. (letter, label)
DEFAULT_OPTIONS: List[Tuple[str, str]] = [
    ("A", "Active surveillance / observation"),
    ("B", "Standard chemotherapy"),
    ("C", "MDM2-inhibitor / p53-reactivation trial"),
    ("D", "Targeted / biomarker-driven therapy"),
]

# Board specialists — persona keys so orthogonal temperatures apply.
SPECIALISTS: List[str] = ["Pathologist", "Geneticist", "Oncologist",
                          "Surgeon", "Pharmacologist", "Equity Officer"]


def _options_block(options: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{k} = {label}" for k, label in options)


def _system_prompt(member: str, temperament: str) -> str:
    return (f"You are the {member} on a precision-oncology tumour board "
            f"(reasoning posture: {temperament}). For the given TP53 case, "
            "estimate the probability that EACH listed option is the best "
            "FIRST management step. Respond with ONLY a JSON object mapping "
            "each option letter to a probability in [0,1] that sums to ~1.0 — "
            'e.g. {"A":0.1,"B":0.2,"C":0.6,"D":0.1}. No prose, no explanation.')


def _user_prompt(mutation: str, case: Dict, options: List[Tuple[str, str]]) -> str:
    ctx = []
    if case.get("cancer"):
        ctx.append(f"cancer: {case['cancer']}")
    if case.get("stage"):
        ctx.append(f"stage: {case['stage']}")
    if case.get("vaf") is not None:
        ctx.append(f"VAF: {case['vaf']}")
    ctxs = (" (" + ", ".join(ctx) + ")") if ctx else ""
    return (f"TP53 {mutation}{ctxs}.\nOPTIONS:\n{_options_block(options)}\n\n"
            "Return the JSON probability distribution now.")


def parse_distribution(text: str,
                       options: List[Tuple[str, str]]) -> Tuple[Dict[str, float], bool]:
    """Extract a {letter: prob} distribution from a model reply.
    Returns (normalised_distribution, ok). On failure -> (uniform, False).
    Never raises."""
    letters = [k for k, _ in options]
    uniform = {k: round(1.0 / len(letters), 4) for k in letters}
    if not text:
        return uniform, False
    # Grab the first {...} block.
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return uniform, False
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return uniform, False
    dist: Dict[str, float] = {}
    for k in letters:
        v = raw.get(k)
        if v is None:  # tolerate lowercase / label keys
            v = raw.get(k.lower())
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        dist[k] = max(0.0, v)
    total = sum(dist.values())
    if total <= 0:
        return uniform, False
    return {k: round(v / total, 4) for k, v in dist.items()}, True


def agent_distribution(member: str, mutation: str, case: Dict,
                       options: List[Tuple[str, str]],
                       generate_fn: Callable[[str, str], str]) -> Dict:
    """One specialist's probability vote. Never raises."""
    from agents.orthogonal_personas import persona_for
    temperament = persona_for(member).get("temperament", "balanced")
    try:
        text = generate_fn(_system_prompt(member, temperament),
                           _user_prompt(mutation, case, options))
    except Exception as e:
        log.warning(f"Consensus vote failed for {member}: {e}")
        text = ""
    dist, ok = parse_distribution(text, options)
    return {"member": member, "temperament": temperament,
            "distribution": dist, "parsed": ok}


def convene_confidence_consensus(
        mutation: str, case: Optional[Dict] = None,
        options: Optional[List[Tuple[str, str]]] = None,
        backend: object = None,
        generate_fn: Optional[Callable[[str, str], str]] = None) -> Dict:
    """Run the probability vote across all specialists and aggregate.

    Provide ONE of: `generate_fn` (used for every agent) — handy for tests; or
    `backend` (each agent is bound to its orthogonal temperature via
    orthogonal_generate). If neither, the default backend is built. Never
    raises."""
    case = dict(case or {})
    options = options or DEFAULT_OPTIONS
    letters = [k for k, _ in options]

    # Reasoning models (e.g. minimax-m3) spend most of a small budget on
    # internal reasoning before emitting the JSON, so give the votes room.
    _VOTE_TOKENS = 800

    def _gen_for(member: str) -> Callable[[str, str], str]:
        if generate_fn is not None:
            return generate_fn
        from agents.orthogonal_personas import orthogonal_generate
        bk = backend
        if bk is None:
            from agents.rag_chain import _build_backend
            bk = _build_backend()
        return orthogonal_generate(bk, member, max_tokens=_VOTE_TOKENS)

    # Run the six votes CONCURRENTLY (they're independent I/O-bound calls) —
    # ~6x faster than sequential, and an honest demonstration of parallel
    # multi-agent inference. Order is preserved.
    from concurrent.futures import ThreadPoolExecutor
    agents: List[Dict] = [None] * len(SPECIALISTS)

    def _one(i_member):
        i, member = i_member
        return i, agent_distribution(member, mutation, case, options,
                                     _gen_for(member))
    try:
        with ThreadPoolExecutor(max_workers=len(SPECIALISTS)) as ex:
            for i, res in ex.map(_one, list(enumerate(SPECIALISTS))):
                agents[i] = res
    except Exception as e:  # pragma: no cover - fall back to sequential
        log.warning(f"Consensus parallel run failed ({e}); going sequential.")
        agents = [agent_distribution(m, mutation, case, options, _gen_for(m))
                  for m in SPECIALISTS]

    # Aggregate: mean probability per option across agents.
    consensus = {k: round(sum(a["distribution"][k] for a in agents) / len(agents), 4)
                 for k in letters}
    top = max(consensus, key=consensus.get)
    top_label = dict(options)[top]
    # Agreement = mean probability the agents assigned to the winning option.
    agreement = round(sum(a["distribution"][top] for a in agents) / len(agents), 4)
    parsed_ok = sum(1 for a in agents if a["parsed"])

    return {
        "success": True,
        "mutation": mutation,
        "options": [{"letter": k, "label": lbl} for k, lbl in options],
        "agents": agents,
        "consensus": consensus,
        "top_option": top,
        "top_label": top_label,
        "agreement": agreement,
        "parsed_ok": parsed_ok,
        "n_agents": len(agents),
        "disclaimer": CONSENSUS_DISCLAIMER,
    }
