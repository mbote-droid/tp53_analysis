"""
============================================================
Precision Onco Africa - Epistemic Uncertainty (honest multi-sample)
agents/uncertainty.py
============================================================
The legitimate version of "MC-dropout / token-entropy" uncertainty for a
hosted model we don't control: sample the SAME prompt N times at temperature
> 0 and measure how much the answers actually agree. Low agreement = the model
is not confident here; high agreement = it is.

Epistemic Uncertainty Index (documented in METHODS.md):

    U = 1 − [ 2 / (N(N−1)) ] · Σ_{i<j} cos(e_i, e_j)

where e_i is the embedding of sample i. Bands: green < 0.15, amber 0.15–0.35,
red ≥ 0.35. This measures MODEL-OUTPUT AGREEMENT — an epistemic-uncertainty
proxy — NOT a clinical probability.

Both the generator and the embedder are injectable, so this is fully unit-
testable without a live model. Never raises.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from utils.logger import log

GREEN_MAX = 0.15
AMBER_MAX = 0.35

UNCERTAINTY_DISCLAIMER = ("Uncertainty reflects agreement across repeated model "
                          "samples (an epistemic proxy) — not a clinical "
                          "probability. Research use only.")


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def band(uncertainty: float) -> str:
    if uncertainty < GREEN_MAX:
        return "green"
    if uncertainty < AMBER_MAX:
        return "amber"
    return "red"


def epistemic_uncertainty(samples: List[str],
                          embed_fn: Callable[[str], List[float]]) -> Dict:
    """Compute the Epistemic Uncertainty Index over N sampled answers.
    Fewer than 2 samples → uncertainty is undefined (returns None). Pure
    aside from embed_fn. Never raises."""
    texts = [s for s in (samples or []) if s and s.strip()]
    n = len(texts)
    if n < 2:
        return {"n_samples": n, "uncertainty": None, "agreement": None,
                "band": "unknown",
                "note": "Need ≥2 samples to measure agreement."}
    try:
        embs = [embed_fn(t) for t in texts]
    except Exception as e:  # pragma: no cover
        log.warning(f"Uncertainty embedding failed: {e}")
        return {"n_samples": n, "uncertainty": None, "agreement": None,
                "band": "unknown", "note": f"Embedding unavailable: {e}"}

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine(embs[i], embs[j]))
    agreement = sum(sims) / len(sims) if sims else 0.0
    # cosine can be slightly <0; clamp agreement to [0,1] for a sane index.
    agreement = max(0.0, min(1.0, agreement))
    uncertainty = round(1.0 - agreement, 3)
    return {"n_samples": n, "uncertainty": uncertainty,
            "agreement": round(agreement, 3), "band": band(uncertainty),
            "disclaimer": UNCERTAINTY_DISCLAIMER}


def _most_representative(samples: List[str],
                         embed_fn: Callable[[str], List[float]]) -> str:
    """Return the sample with the highest mean similarity to the others (the
    'centroid' answer) — the most defensible one to surface."""
    texts = [s for s in samples if s and s.strip()]
    if len(texts) <= 1:
        return texts[0] if texts else ""
    try:
        embs = [embed_fn(t) for t in texts]
    except Exception:
        return texts[0]
    best_i, best_score = 0, -1.0
    for i in range(len(texts)):
        score = sum(_cosine(embs[i], embs[j])
                    for j in range(len(texts)) if j != i)
        if score > best_score:
            best_i, best_score = i, score
    return texts[best_i]


def sample_and_measure(system_prompt: str, user_prompt: str,
                       generate_fn: Callable[[str, str], str],
                       embed_fn: Optional[Callable[[str], List[float]]] = None,
                       n: int = 3) -> Dict:
    """Sample the prompt n times, measure epistemic uncertainty, and return a
    representative answer with the uncertainty band. generate_fn should already
    be bound to a temperature > 0 for meaningful variation. Never raises."""
    n = max(2, int(n))
    if embed_fn is None:
        try:
            from knowledge_base.vector_store import _get_embedding_model
            model = _get_embedding_model()
            embed_fn = model.embed_query
        except Exception as e:
            return {"success": False, "reason": f"No embedder: {e}"}

    samples: List[str] = []
    for _ in range(n):
        try:
            samples.append((generate_fn(system_prompt, user_prompt) or "").strip())
        except Exception as e:
            log.warning(f"Uncertainty sample failed: {e}")
    samples = [s for s in samples if s]
    if not samples:
        return {"success": False, "reason": "All samples failed."}

    metric = epistemic_uncertainty(samples, embed_fn)
    answer = _most_representative(samples, embed_fn)
    return {"success": True, "answer": answer, "samples": samples, **metric}
