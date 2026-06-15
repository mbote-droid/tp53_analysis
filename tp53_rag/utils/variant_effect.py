"""
============================================================
TP53 RAG Platform - ESM-2 variant-effect prediction (runtime)
utils/variant_effect.py
============================================================
Serves real ESM-2 protein-language-model variant-effect scores **offline, with
no torch at runtime**.

How it works
------------
The heavy compute (running the ESM-2 transformer over every TP53 substitution)
is done ONCE, ahead of time, by `tools/precompute_esm2.py` on a torch-enabled
machine (a GPU laptop or a free Colab). That produces a small JSON matrix:

    data/esm2_tp53_effect.json
      { model, uniprot, sequence, method, scores: { "175": {"H": -8.1, ...} } }

This module just loads that matrix and answers lookups — so the 8 GB / offline
deployment never needs torch or the model weights.

Honesty: if the matrix has not been generated yet, predictions are reported as
**unavailable** with a clear instruction. No scores are ever fabricated.

Scoring convention (wild-type marginal log-likelihood ratio):
    score = log P(mutant | context) - log P(wild-type | context)
More negative  ->  the model finds the mutant less likely  ->  more deleterious.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

DEFAULT_MATRIX_PATH = Path("data/esm2_tp53_effect.json")

_VARIANT_RE = re.compile(r'\b([A-Z])\s*(\d{1,3})\s*([A-Z])\b', re.I)

# Qualitative buckets for the ESM-2 LLR. Thresholds are heuristic (the ESM
# variant-effect literature does not fix a universal cut-off) and so are
# CONFIGURABLE via environment variables — tune the labelling without touching
# code (e.g. after calibrating against labelled data). More negative = more
# deleterious; thresholds should stay in ascending order.
#   ESM2_THRESH_DELETERIOUS  (default -7.5)  score <= this -> "likely deleterious"
#   ESM2_THRESH_POSSIBLY     (default -4.0)  score <= this -> "possibly deleterious"
#   ESM2_THRESH_UNCERTAIN    (default  0.0)  score <= this -> "uncertain"; else "likely tolerated"
_DEFAULT_THRESHOLDS = {
    "ESM2_THRESH_DELETERIOUS": -7.5,
    "ESM2_THRESH_POSSIBLY": -4.0,
    "ESM2_THRESH_UNCERTAIN": 0.0,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _buckets() -> List[tuple]:
    """Thresholds read fresh from env each call so they're runtime-configurable."""
    return [
        (_env_float("ESM2_THRESH_DELETERIOUS", _DEFAULT_THRESHOLDS["ESM2_THRESH_DELETERIOUS"]),
         "likely deleterious"),
        (_env_float("ESM2_THRESH_POSSIBLY", _DEFAULT_THRESHOLDS["ESM2_THRESH_POSSIBLY"]),
         "possibly deleterious"),
        (_env_float("ESM2_THRESH_UNCERTAIN", _DEFAULT_THRESHOLDS["ESM2_THRESH_UNCERTAIN"]),
         "uncertain"),
        (float("inf"), "likely tolerated"),
    ]


def _bucket(score: float) -> str:
    for thresh, label in _buckets():
        if score <= thresh:
            return label
    return "uncertain"


@dataclass
class VariantEffect:
    """ESM-2 variant-effect result. Always populated (never raises)."""
    query: str
    wild_type: str = ""
    position: Optional[int] = None
    mutant: str = ""
    esm2_score: Optional[float] = None
    interpretation: str = "unavailable"
    available: bool = False
    model: str = ""
    source: str = "not_computed"
    notes: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def parse_variant(text: str):
    """Return (wt, position:int, mut) from e.g. 'R175H' / 'p.R175H', or None."""
    if not text:
        return None
    m = _VARIANT_RE.search(text.replace("p.", ""))
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


class VariantEffectPredictor:
    """Loads a precomputed ESM-2 matrix and serves offline lookups."""

    def __init__(self, matrix_path: Optional[str] = None):
        self._path = Path(matrix_path) if matrix_path else DEFAULT_MATRIX_PATH
        self._matrix: Optional[Dict] = None
        self._load()

    def _load(self) -> None:
        try:
            if self._path.is_file():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("scores"), dict):
                    self._matrix = data
                    log.info(f"ESM-2 variant-effect matrix loaded "
                             f"({len(data['scores'])} positions, model={data.get('model')})")
        except Exception as e:
            log.warning(f"ESM-2 matrix load failed: {e}")
            self._matrix = None

    @property
    def available(self) -> bool:
        return self._matrix is not None

    def predict(self, variant: str) -> VariantEffect:
        variant = (variant or "").strip()
        res = VariantEffect(query=variant)
        parsed = parse_variant(variant)
        if not parsed:
            res.notes = "Could not parse a missense variant (e.g. R175H)."
            return res
        wt, pos, mut = parsed
        res.wild_type, res.position, res.mutant = wt, pos, mut

        if not self.available:
            res.notes = ("ESM-2 scores not precomputed. Run "
                         "`python tools/precompute_esm2.py` on a torch-enabled "
                         "machine to generate data/esm2_tp53_effect.json.")
            return res

        res.model = self._matrix.get("model", "")
        # Optional wild-type sanity check against the stored sequence
        seq = self._matrix.get("sequence") or ""
        if seq and 1 <= pos <= len(seq) and seq[pos - 1] != wt:
            res.notes = (f"Wild-type mismatch: position {pos} is {seq[pos-1]} in "
                         f"the reference, not {wt}. Check the variant.")
            return res

        score = (self._matrix.get("scores", {}).get(str(pos), {}) or {}).get(mut)
        if score is None:
            res.notes = f"No ESM-2 score stored for {wt}{pos}{mut}."
            return res
        try:
            res.esm2_score = round(float(score), 3)
        except (TypeError, ValueError):
            res.notes = "Stored score was not numeric."
            return res
        res.interpretation = _bucket(res.esm2_score)
        res.available = True
        res.source = "esm2_precomputed"
        return res

    def stats(self) -> Dict:
        if not self.available:
            return {"available": False, "path": str(self._path)}
        return {
            "available": True,
            "model": self._matrix.get("model", ""),
            "positions": len(self._matrix.get("scores", {})),
            "method": self._matrix.get("method", ""),
            "path": str(self._path),
        }


_predictor: Optional[VariantEffectPredictor] = None


def get_predictor() -> VariantEffectPredictor:
    global _predictor
    if _predictor is None:
        _predictor = VariantEffectPredictor()
    return _predictor


def predict_effect(variant: str) -> Dict:
    """Convenience: predict and return a plain dict."""
    return get_predictor().predict(variant).to_dict()
