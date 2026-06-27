"""
============================================================
Precision Onco Africa - Synthetic Lethality & Network Modeler
agents/synthetic_lethality.py
============================================================
Identifies synthetic-lethal (SL) targets for TP53-mutant tumours — genes
whose inhibition selectively kills p53-deficient cells (the tumour) while
sparing p53-wild-type normal tissue.

Offline-first & curated: a sourced SL knowledge base (DepMap dependency
signals + published p53 SL screens) ships in-code. Rule-based scoring ranks
targets by evidence strength and druggability. No LLM / network required.

DISCLAIMER: curated from public DepMap/literature for research context —
not patient-specific predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "synthetic_lethality"
DISCLAIMER = ("Curated DepMap/literature synthetic-lethality context — "
              "research use only, not patient-specific predictions.")

_EVIDENCE_SCORE = {"high": 3, "medium": 2, "emerging": 1}
_DRUGGABILITY_SCORE = {"clinical": 3, "preclinical": 2, "tool": 1, "none": 0}

# Curated TP53 synthetic-lethal partners (sourced).
SL_TARGETS: List[Dict] = [
    {"gene": "WEE1", "mechanism": "G2/M checkpoint kinase — abrogation forces "
     "mitotic catastrophe in p53-deficient cells", "drug": "Adavosertib (AZD1775)",
     "druggability": "clinical", "evidence": "high",
     "source": "DepMap; Hirai et al.; multiple p53 SL screens"},
    {"gene": "ATR", "mechanism": "Replication-stress checkpoint kinase; p53-null "
     "cells depend on ATR for survival", "drug": "Ceralasertib (AZD6738)",
     "druggability": "clinical", "evidence": "high",
     "source": "DepMap; Reaper et al."},
    {"gene": "CHEK1", "mechanism": "CHK1 checkpoint kinase; loss of p53 increases "
     "CHK1 dependency", "drug": "Prexasertib", "druggability": "clinical",
     "evidence": "high", "source": "DepMap; published SL screens"},
    {"gene": "PLK1", "mechanism": "Polo-like kinase 1; mitotic dependency in "
     "p53-mutant tumours", "drug": "Onvansertib", "druggability": "clinical",
     "evidence": "medium", "source": "DepMap"},
    {"gene": "AURKB", "mechanism": "Aurora kinase B; mitotic SL signal with p53 loss",
     "drug": "Barasertib", "druggability": "preclinical", "evidence": "medium",
     "source": "DepMap"},
    {"gene": "POLQ", "mechanism": "DNA polymerase theta (MMEJ repair); SL in "
     "repair-deficient/p53-altered contexts", "drug": "Novobiocin (POLθ inhibitor)",
     "druggability": "preclinical", "evidence": "medium",
     "source": "Zhou et al.; DepMap"},
    {"gene": "KIF18A", "mechanism": "Mitotic kinesin; selective dependency in "
     "chromosomally-unstable/p53-altered tumours", "drug": "KIF18A inhibitors (early)",
     "druggability": "preclinical", "evidence": "emerging",
     "source": "recent DepMap dependency maps"},
    {"gene": "ENDOD1", "mechanism": "Endonuclease; reported synthetic-lethal "
     "partner of mutant p53", "drug": "—", "druggability": "none",
     "evidence": "emerging", "source": "published p53 SL literature"},
]


@dataclass
class SLResult:
    mutation: str
    targets: List[Dict]
    network_edges: List[Dict]
    top_target: str
    disclaimer: str = DISCLAIMER


class SyntheticLethalityModeler:
    """Rank synthetic-lethal targets for a TP53-mutant context."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/synthetic_lethality.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"SL audit dir unavailable: {e}")

    @staticmethod
    def _score(t: Dict) -> int:
        return (_EVIDENCE_SCORE.get(t.get("evidence"), 0) * 2
                + _DRUGGABILITY_SCORE.get(t.get("druggability"), 0))

    def model(self, mutation: str = "TP53-mutant",
              min_evidence: str = "emerging") -> Dict:
        """Return ranked SL targets + a TP53-centric network. Never empty."""
        mut = str(mutation or "TP53-mutant").strip()
        floor = _EVIDENCE_SCORE.get(min_evidence, 1)
        targets = [dict(t, sl_score=self._score(t)) for t in SL_TARGETS
                   if _EVIDENCE_SCORE.get(t.get("evidence"), 0) >= floor]
        if not targets:  # never empty — fall back to the full set
            targets = [dict(t, sl_score=self._score(t)) for t in SL_TARGETS]
        targets.sort(key=lambda t: t["sl_score"], reverse=True)

        edges = [{"source": "TP53", "target": t["gene"],
                  "weight": t["sl_score"], "evidence": t["evidence"]}
                 for t in targets]

        result = SLResult(
            mutation=mut, targets=targets, network_edges=edges,
            top_target=targets[0]["gene"],
        )
        self._audit(f"sl:{mut} -> {len(targets)} targets, top={result.top_target}")
        return {
            **asdict(result),
            "count": len(targets),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": f"{len(targets)} synthetic-lethal target(s) for {mut} "
                       f"(top: {result.top_target})",
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"SL audit failed: {e}")


_modeler = SyntheticLethalityModeler()


def model_synthetic_lethality(mutation: str = "TP53-mutant") -> Dict:
    return _modeler.model(mutation)
