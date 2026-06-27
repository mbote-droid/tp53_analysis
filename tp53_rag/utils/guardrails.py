"""
============================================================
Precision Onco Africa - Dual Guardrails
utils/guardrails.py
============================================================
Every answer passes through two independent gates before it reaches a user:

    answer
      → 1. SYNTACTIC guardrail   (well-formed, non-empty, no error/echo markers)
      → 2. SCIENTIFIC guardrail  (claims cross-checked against ClinVar)
      → confidence
      → gate verdict (pass / flag / block)

The two gates are deliberately different in kind: one checks *form*, the other
checks *fact*. A fluent answer that contradicts ClinVar fails the scientific
gate even though it passes the syntactic one — which is exactly the failure a
single validator misses.

Reuses the existing ClinVar conflict checker for the scientific gate. Pure and
offline (curated ClinVar baseline); never raises; never fabricates a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from utils.logger import log

DISCLAIMER = ("Automated dual-gate check (form + fact) — research use only, "
              "not a guarantee of clinical correctness.")

GATE_PASS = "pass"
GATE_FLAG = "flag"
GATE_BLOCK = "block"

_ERROR_MARKERS = ("query error", "agent unavailable", "report generation failed",
                  "traceback", "exception:", "none\nnone")


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    severity: str          # ok | warn | fail


def _syntactic_gate(answer: str) -> GateResult:
    """Form check: non-empty, sane length, no error/echo markers."""
    text = str(answer or "").strip()
    if not text:
        return GateResult("syntactic", False, "Empty response.", "fail")
    low = text.lower()
    for marker in _ERROR_MARKERS:
        if marker in low:
            return GateResult("syntactic", False,
                              f"Contains an error marker ('{marker}').", "fail")
    if len(text) < 12:
        return GateResult("syntactic", False,
                          "Response too short to be meaningful.", "warn")
    return GateResult("syntactic", True, "Well-formed, non-empty response.", "ok")


def _scientific_gate(answer: str, mutation: Optional[str]) -> GateResult:
    """Fact check: cross-examine TP53 classification claims against ClinVar."""
    try:
        from agents.clinvar_conflict_checker import check_conflicts
        # Prefer text-based checking so the claim *asserted in the answer* is
        # compared with ClinVar (not just the bare mutation). Fall back to the
        # mutation alone only when there is no answer text.
        text = str(answer or "").strip()
        res = (check_conflicts(text=text) if text
               else check_conflicts(mutation=mutation))
    except Exception as e:  # pragma: no cover
        log.warning(f"Scientific gate unavailable: {e}")
        return GateResult("scientific", True,
                          "ClinVar check unavailable — not blocking.", "warn")
    verdict = res.get("verdict", "no_claims")
    if verdict == "concordant":
        return GateResult("scientific", True,
                          "All checked claims agree with ClinVar.", "ok")
    if verdict == "no_claims":
        return GateResult("scientific", True,
                          "No verifiable TP53 claim to check.", "ok")
    if verdict == "conflict_high":
        return GateResult("scientific", False,
                          f"{res.get('conflicts_found', 0)} high-severity "
                          "ClinVar conflict(s).", "fail")
    return GateResult("scientific", False,
                      f"{res.get('conflicts_found', 0)} ClinVar conflict(s).",
                      "warn")


def run_guardrails(answer: str, mutation: Optional[str] = None) -> Dict:
    """Run both gates and return a combined verdict. Never empty, never raises."""
    g1 = _syntactic_gate(answer)
    g2 = _scientific_gate(answer, mutation)
    gates: List[GateResult] = [g1, g2]

    # Gate logic: any hard fail → block; any warn → flag; else pass.
    if any(g.severity == "fail" for g in gates):
        gate = GATE_BLOCK
    elif any(g.severity == "warn" for g in gates):
        gate = GATE_FLAG
    else:
        gate = GATE_PASS

    # Confidence: start at 1.0, subtract per issue.
    confidence = 1.0
    for g in gates:
        if g.severity == "fail":
            confidence -= 0.5
        elif g.severity == "warn":
            confidence -= 0.2
    confidence = round(max(0.0, confidence), 2)

    flags = [f"{g.name}: {g.detail}" for g in gates if not g.passed]
    return {
        "gate": gate,
        "passed": gate == GATE_PASS,
        "confidence": confidence,
        "gates": [asdict(g) for g in gates],
        "flags": flags,
        "disclaimer": DISCLAIMER,
        "message": (f"Guardrails: {gate.upper()} "
                    f"(confidence {confidence:.0%})"),
    }
