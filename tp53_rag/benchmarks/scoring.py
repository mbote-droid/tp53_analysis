"""
Benchmark scoring helpers — pure functions, no I/O, no LLM.

Kept separate from the runner so they can be unit-tested in isolation.
Every function tolerates None / missing / unexpected values and never raises
on bad input (honours the platform's zero-empty-output + graceful-degradation
rules).
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Canonical 5-point clinical-significance vocabulary.
CANONICAL_SIGNIFICANCE = {
    "pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign",
}

# Collapse the 5-point scale into 3 clinically-actionable buckets so that, e.g.,
# "likely_pathogenic" still counts as concordant with "pathogenic".
_BUCKET = {
    "pathogenic": "pathogenic_leaning",
    "likely_pathogenic": "pathogenic_leaning",
    "benign": "benign_leaning",
    "likely_benign": "benign_leaning",
    "vus": "uncertain",
    "unknown": "uncertain",
    "": "uncertain",
}


def normalize_significance(value: Optional[str]) -> str:
    """Lower-case, whitespace/dash-normalised significance string.

    Maps common synonyms onto the canonical vocabulary. Unknown values pass
    through normalised (never raises); callers can treat them as 'uncertain'.
    """
    if value is None:
        return ""
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    synonyms = {
        "uncertain_significance": "vus",
        "variant_of_uncertain_significance": "vus",
        "uncertain": "vus",
        "neutral": "benign",
        "functional": "benign",
    }
    return synonyms.get(s, s)


def significance_bucket(value: Optional[str]) -> str:
    """Collapse a significance value into pathogenic_leaning / benign_leaning /
    uncertain."""
    return _BUCKET.get(normalize_significance(value), "uncertain")


def score_variant(predicted: Dict, expected: Dict) -> Dict:
    """Score one variant prediction against its expected record.

    Returns a non-empty result dict; missing fields degrade to misses rather
    than errors.
    """
    pred_sig = normalize_significance((predicted or {}).get("clinical_significance"))
    exp_sig = normalize_significance((expected or {}).get("expected_significance"))
    pred_iarc = (predicted or {}).get("iarc_classification")
    exp_iarc = (expected or {}).get("expected_iarc")

    exact = bool(exp_sig) and pred_sig == exp_sig
    concordant = significance_bucket(pred_sig) == significance_bucket(exp_sig)

    # IARC is only scored where a ground-truth value exists.
    if exp_iarc:
        iarc_match: Optional[bool] = (str(pred_iarc).upper() == str(exp_iarc).upper()
                                      if pred_iarc else False)
    else:
        iarc_match = None  # not applicable

    return {
        "mutation": (expected or {}).get("mutation", "?"),
        "expected_significance": exp_sig or "(none)",
        "predicted_significance": pred_sig or "(none)",
        "exact_match": exact,
        "concordant": concordant,
        "expected_iarc": exp_iarc,
        "predicted_iarc": pred_iarc,
        "iarc_match": iarc_match,
        "expected_bucket": significance_bucket(exp_sig),
        "predicted_bucket": significance_bucket(pred_sig),
    }


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def aggregate(results: List[Dict]) -> Dict:
    """Aggregate per-variant results into overall metrics.

    Reports exact accuracy, concordant (bucketed) accuracy, IARC concordance,
    and precision/recall/F1 for detecting pathogenic-leaning variants. Always
    returns a populated dict, even for an empty input.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0, "exact_accuracy": 0.0, "concordant_accuracy": 0.0,
            "iarc_concordance": 0.0, "iarc_scored": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "note": "No results to aggregate.",
        }

    exact = sum(1 for r in results if r.get("exact_match"))
    concordant = sum(1 for r in results if r.get("concordant"))

    iarc_scored = [r for r in results if r.get("iarc_match") is not None]
    iarc_hits = sum(1 for r in iarc_scored if r.get("iarc_match"))

    tp = fp = fn = tn = 0
    for r in results:
        exp_pos = r.get("expected_bucket") == "pathogenic_leaning"
        pred_pos = r.get("predicted_bucket") == "pathogenic_leaning"
        if exp_pos and pred_pos:
            tp += 1
        elif pred_pos and not exp_pos:
            fp += 1
        elif exp_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "n": n,
        "exact_accuracy": round(_safe_div(exact, n), 4),
        "concordant_accuracy": round(_safe_div(concordant, n), 4),
        "iarc_concordance": round(_safe_div(iarc_hits, len(iarc_scored)), 4),
        "iarc_scored": len(iarc_scored),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
