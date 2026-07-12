"""
============================================================
Precision Onco Africa - Adversarial Evidence Layer
agents/adversarial_evidence.py
============================================================
Most RAG systems only retrieve evidence that SUPPORTS an answer. This layer
does the opposite on purpose: it actively hunts for evidence that would
CONTRADICT a proposed recommendation, then stages a bounded skeptic vs
proposer exchange so the contradicting evidence is confronted before anything
reaches the clinician.

Three honest pieces (none touch model internals):

  1. counterfactual_trials() — beyond "which trials match", it also asks
     "which trials for this variant were STOPPED (terminated/withdrawn/
     suspended)?" and folds that into a transparent Viability Score.

  2. gather_contradicting_evidence() — pulls ClinVar conflicts + stopped-trial
     signals into one negative-evidence bundle.

  3. bounded_debate() — a HARD-CAPPED 2-turn exchange (skeptic challenges →
     proposer rebuts). It NEVER loops to convergence — that would freeze an
     8 GB laptop and is explicitly disallowed. Exactly `max_turns` LLM calls.

Everything is decision-support (RUO). The Viability Score is an ILLUSTRATIVE
composite heuristic — NOT a validated efficacy or survival probability.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from utils.logger import log

VIABILITY_DISCLAIMER = ("Viability is an illustrative composite heuristic for "
                        "triage/attention only — NOT a validated efficacy or "
                        "survival probability. Research use only.")

# Statuses that mean a trial was stopped early — a real counterfactual signal.
_STOPPED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}

# Negative-signal severity weights for the Viability Score (sum ≤ 1).
_NEG_WEIGHTS = {"stopped_trials": 0.34, "clinvar_conflicts": 0.33,
                "resistance": 0.33}
_NEG_CAP = 3.0  # a class saturates its penalty at this many hits


def viability_score(positive_matches: int, negatives: Dict[str, int],
                    weights: Optional[Dict[str, float]] = None) -> float:
    """V = S_match · (1 − min(1, Σ_k w_k·h_k)); h_k = min(1, count_k/CAP).

    S_match scales with the number of positive (matching) trials; the penalty
    grows with normalised negative-signal counts. Clamped to [0, 1]. Pure.
    Heuristic — see VIABILITY_DISCLAIMER.
    """
    weights = weights or _NEG_WEIGHTS
    pos = max(0, int(positive_matches))
    s_match = 0.3 if pos == 0 else min(1.0, 0.55 + 0.09 * (pos - 1))
    penalty = 0.0
    for k, w in weights.items():
        h = min(1.0, max(0, negatives.get(k, 0)) / _NEG_CAP)
        penalty += w * h
    v = s_match * (1.0 - min(1.0, penalty))
    return round(max(0.0, min(1.0, v)), 2)


def counterfactual_trials(mutation: str, cancer: str = "",
                          matcher: Optional[object] = None) -> Dict:
    """Dual retrieval: recruiting matches (positive) + stopped trials
    (negative) for a variant, folded into a Viability Score. `matcher` is any
    object exposing .search(...) like ClinicalTrialsMatcher; injected for
    tests. Never raises."""
    try:
        if matcher is None:
            from agents.clinical_trials import ClinicalTrialsMatcher
            matcher = ClinicalTrialsMatcher()
    except Exception as e:  # pragma: no cover
        return {"success": False, "reason": str(e)}

    try:
        pos = matcher.search(mutation=mutation, cancer_type=cancer,
                             status="RECRUITING")
        positive = [t for t in pos.get("trials", [])]
        # Negative sweep: no status filter, then keep the stopped ones.
        allres = matcher.search(mutation=mutation, cancer_type=cancer,
                               status="")
        stopped = [t for t in allres.get("trials", [])
                   if str(t.get("status", "")).upper() in _STOPPED_STATUSES]
    except Exception as e:
        log.warning(f"counterfactual_trials failed: {e}")
        return {"success": False, "reason": str(e)}

    negatives = {"stopped_trials": len(stopped)}
    score = viability_score(len(positive), negatives)
    return {"success": True, "mutation": mutation, "cancer": cancer,
            "positive_count": len(positive), "stopped_count": len(stopped),
            "stopped_trials": stopped[:5], "viability": score,
            "disclaimer": VIABILITY_DISCLAIMER}


def gather_contradicting_evidence(mutation: str, cancer: str = "",
                                  matcher: Optional[object] = None) -> Dict:
    """Bundle negative/contradicting signals: ClinVar conflicts + stopped
    trials. Returns a structured dict (never raises)."""
    contradictions: List[str] = []
    clinvar_conflicts = 0
    try:
        from agents.clinvar_conflict_checker import check_conflicts
        cv = check_conflicts(mutation=mutation)
        verdict = cv.get("verdict", "no_claims")
        if verdict.startswith("conflict"):
            clinvar_conflicts = cv.get("conflicts_found", 1)
            contradictions.append(
                f"ClinVar shows {clinvar_conflicts} classification conflict(s) "
                f"for {mutation}.")
    except Exception as e:  # pragma: no cover
        log.warning(f"ClinVar contradiction check failed: {e}")

    ct = counterfactual_trials(mutation, cancer, matcher=matcher)
    stopped = ct.get("stopped_count", 0) if ct.get("success") else 0
    if stopped:
        contradictions.append(
            f"{stopped} clinical trial(s) for this variant/condition were "
            f"stopped early (terminated/withdrawn/suspended).")

    return {"mutation": mutation, "cancer": cancer,
            "contradictions": contradictions,
            "clinvar_conflicts": clinvar_conflicts,
            "stopped_trials": stopped,
            "viability": ct.get("viability") if ct.get("success") else None,
            "has_contradictions": bool(contradictions)}


_SKEPTIC_SYSTEM = (
    "You are the SKEPTIC on an AI tumour board. Your sole job is to challenge "
    "the proposed recommendation using the contradicting evidence provided. "
    "Be specific and evidence-bound; raise the strongest genuine objections. "
    "Do not invent evidence. If the contradicting evidence is weak, say the "
    "proposal largely stands. 3–5 sentences.")

_PROPOSER_SYSTEM = (
    "You are the PROPOSER on an AI tumour board defending a recommendation. "
    "Respond to the skeptic's objections honestly: concede what is valid, "
    "rebut what is not, and state the residual uncertainty plainly. Do not "
    "overclaim. 3–5 sentences.")


def _generate_with_retry(gen_fn: Callable[[str, str], str], system: str,
                         user: str, attempts: int = 3,
                         base_delay: float = 0.6) -> str:
    """Call `gen_fn(system, user)` with bounded retry + exponential backoff.

    Hosted model endpoints occasionally return a transient error (e.g. a
    provider HTTP 500) that succeeds on a second try. We retry ONLY on raised
    exceptions — an empty string is a valid (if unhelpful) response and is
    returned as-is so the caller degrades gracefully. If every attempt raises,
    the last exception propagates so `bounded_debate` reports the debate as
    unavailable instead of crashing.
    """
    attempts = max(1, int(attempts))
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return gen_fn(system, user)
        except Exception as e:  # transient provider/network error
            last = e
            if i < attempts - 1:
                log.warning(f"bounded_debate: generate attempt {i + 1}/"
                            f"{attempts} failed ({e}); retrying")
                time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


def bounded_debate(proposal: str, contradicting_evidence: List[str],
                   generate_fn: Callable[[str, str], str],
                   max_turns: int = 2,
                   role_generate: Optional[Callable[[str], Callable]] = None
                   ) -> Dict:
    """A HARD-CAPPED skeptic↔proposer exchange. Exactly min(max_turns, 2)
    LLM calls — never loops to convergence. `generate_fn(system, user)->str`
    is injected. If `role_generate(role)->generate_fn` is supplied, each role
    uses its own (orthogonal-temperature) generator so the debate is not an
    echo chamber. Never raises."""
    turns: List[Dict] = []
    cap = max(1, min(int(max_turns), 2))  # hard ceiling of 2, per design
    evidence_txt = ("\n".join(f"- {c}" for c in contradicting_evidence)
                    or "- No specific contradicting evidence was retrieved.")

    def _gen_for(role: str):
        if role_generate is not None:
            try:
                return role_generate(role)
            except Exception:
                return generate_fn
        return generate_fn

    try:
        skeptic_user = (f"PROPOSED RECOMMENDATION:\n{proposal}\n\n"
                        f"CONTRADICTING EVIDENCE:\n{evidence_txt}\n\n"
                        "Give your strongest evidence-bound challenge.")
        skeptic = (_generate_with_retry(_gen_for("Skeptic"),
                                        _SKEPTIC_SYSTEM, skeptic_user)
                   or "").strip()
        turns.append({"role": "skeptic", "text": skeptic})

        if cap >= 2:
            proposer_user = (f"YOUR RECOMMENDATION:\n{proposal}\n\n"
                             f"SKEPTIC'S CHALLENGE:\n{skeptic}\n\n"
                             "Respond: concede, rebut, and state residual "
                             "uncertainty.")
            rebuttal = (_generate_with_retry(_gen_for("Proposer"),
                                             _PROPOSER_SYSTEM, proposer_user)
                        or "").strip()
            turns.append({"role": "proposer", "text": rebuttal})
    except Exception as e:
        log.error(f"bounded_debate failed: {e}")
        return {"success": False, "reason": str(e), "turns": turns,
                "turn_count": len(turns)}

    return {"success": True, "turns": turns, "turn_count": len(turns),
            "hard_capped_at": cap,
            "note": "Bounded 2-turn cross-examination (never run to convergence)."}


def adversarial_review(mutation: str, proposal: str, cancer: str = "",
                       generate_fn: Optional[Callable[[str, str], str]] = None,
                       matcher: Optional[object] = None) -> Dict:
    """Full pass: gather contradicting evidence, then run the bounded debate.
    If no generate_fn is supplied, builds the default backend. Never raises."""
    evidence = gather_contradicting_evidence(mutation, cancer, matcher=matcher)
    role_generate = None
    if generate_fn is None:
        try:
            from agents.rag_chain import _build_backend
            from agents.orthogonal_personas import orthogonal_generate
            backend = _build_backend()

            def generate_fn(system, user):  # noqa: E306
                return backend.generate(system, user, max_tokens=512)

            # Orthogonal sampling per role → the skeptic reasons strictly, the
            # proposer more exploratorily, so the debate isn't an echo chamber.
            def role_generate(role):  # noqa: E306
                return orthogonal_generate(backend, role, max_tokens=512)
        except Exception as e:
            return {"success": False, "reason": str(e), "evidence": evidence}
    debate = bounded_debate(proposal, evidence["contradictions"], generate_fn,
                            role_generate=role_generate)
    return {"success": debate.get("success", False), "evidence": evidence,
            "debate": debate, "viability": evidence.get("viability")}
