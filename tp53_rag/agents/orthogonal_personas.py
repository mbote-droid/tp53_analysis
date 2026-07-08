"""
============================================================
Precision Onco Africa - Orthogonal Personas (anti echo-chamber)
agents/orthogonal_personas.py
============================================================
If every agent shares one base model at one temperature, a "debate" is
theatre: they tend to agree because they share the same biases. To make
multi-agent disagreement REAL, each role is given an orthogonal reasoning
posture — a distinct temperature and frequency penalty — so genuine
divergence has to be earned, not assumed.

This module is the honest home of that idea:
  * ORTHOGONAL_PERSONAS documents each role's temperament + sampling params.
  * orthogonal_generate() binds a backend to a role's params, so any
    LLM-driven multi-agent step (e.g. the skeptic↔proposer debate) actually
    samples each role differently.
  * The deterministic tumour-board specialists carry their temperament as an
    honest LABEL of their designed posture (they reach different stances by
    rule, not by a fake temperature knob).

Pure config + a thin generator wrapper. Never raises.
"""
from __future__ import annotations

from typing import Callable, Dict

# role → sampling posture. Low temp = strict/conservative; higher = exploratory.
ORTHOGONAL_PERSONAS: Dict[str, Dict] = {
    "Pathologist":    {"temperature": 0.1, "frequency_penalty": 0.0,
                       "temperament": "conservative"},
    "Geneticist":     {"temperature": 0.2, "frequency_penalty": 0.0,
                       "temperament": "evidence-strict"},
    "Oncologist":     {"temperature": 0.4, "frequency_penalty": 0.2,
                       "temperament": "balanced"},
    "Surgeon":        {"temperature": 0.3, "frequency_penalty": 0.1,
                       "temperament": "pragmatic"},
    "Pharmacologist": {"temperature": 0.6, "frequency_penalty": 0.3,
                       "temperament": "exploratory"},
    "Equity Officer": {"temperament": "access-focused", "temperature": 0.4,
                       "frequency_penalty": 0.2},
    "Skeptic":        {"temperature": 0.2, "frequency_penalty": 0.4,
                       "temperament": "adversarial"},
    "Proposer":       {"temperature": 0.5, "frequency_penalty": 0.1,
                       "temperament": "constructive"},
}

_DEFAULT = {"temperature": 0.3, "frequency_penalty": 0.0, "temperament": "balanced"}


def persona_for(member: str) -> Dict:
    """Case-insensitive lookup of a role's orthogonal posture. Never raises."""
    if not member:
        return dict(_DEFAULT)
    key = str(member).strip()
    if key in ORTHOGONAL_PERSONAS:
        return dict(ORTHOGONAL_PERSONAS[key])
    low = key.lower()
    for k, v in ORTHOGONAL_PERSONAS.items():
        if k.lower() == low:
            return dict(v)
    return dict(_DEFAULT)


def orthogonal_generate(backend, member: str,
                        max_tokens: int = 512) -> Callable[[str, str], str]:
    """Return a generate_fn(system, user)->str bound to `member`'s orthogonal
    temperature / frequency penalty, so different roles sample differently.
    Tolerates backends that ignore some kwargs (they just use what they honour).
    """
    posture = persona_for(member)
    temp = posture.get("temperature", 0.3)
    freq = posture.get("frequency_penalty", 0.0)

    def _gen(system_prompt: str, user_prompt: str) -> str:
        try:
            return backend.generate(system_prompt, user_prompt,
                                    max_tokens=max_tokens, temperature=temp,
                                    frequency_penalty=freq)
        except TypeError:
            # Backend doesn't accept frequency_penalty (or temperature) — retry
            # with only what the common signature guarantees.
            try:
                return backend.generate(system_prompt, user_prompt,
                                        max_tokens=max_tokens, temperature=temp)
            except TypeError:
                return backend.generate(system_prompt, user_prompt,
                                        max_tokens=max_tokens)

    return _gen
