"""
============================================================
Precision Onco Africa - Evidence Scenario Explorer ("Digital Twin")
agents/digital_twin.py
============================================================
Explores how a case *might* unfold under different management strategies, using
patterns drawn from published TP53 cohort literature — NOT a prediction for an
individual patient.

This is the honest form of a "digital twin": it surfaces evidence-based
scenarios (each labelled illustrative, with its basis and caveats) so a
clinician can reason about options. It never outputs a fabricated survival
figure for the specific patient; outcomes are expressed qualitatively or as
population-level literature ranges that are clearly marked as such.

Deterministic, offline, curated. Pure and never-empty.

DISCLAIMER: illustrative scenarios from published cohort patterns — research
use only, NOT individual prognosis, prediction, or medical advice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log
from agents.tumor_board import parse_variant

AGENT_ID = "digital_twin"
DISCLAIMER = ("Illustrative evidence scenarios from published TP53 cohort "
              "patterns — research use only. NOT a prediction, prognosis, or "
              "treatment recommendation for any individual patient.")


@dataclass
class Scenario:
    name: str
    intervention: str
    illustrative_outcome: str       # qualitative / population-level, labelled
    evidence_basis: str
    confidence: str                 # high | moderate | low | investigational
    caveat: str


@dataclass
class TwinExploration:
    mutation: str
    cancer: str
    stage: str
    classification: str
    scenarios: List[Dict] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class DigitalTwinExplorer:
    """Generate illustrative, evidence-grounded management scenarios."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/digital_twin.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Digital-twin audit dir unavailable: {e}")

    def explore(self, mutation: str, case: Optional[Dict] = None) -> Dict:
        """Return illustrative scenarios for a case. Never empty; no fabricated
        individual outcomes."""
        case = dict(case or {})
        cancer = str(case.get("cancer") or "the tumour").strip() or "the tumour"
        stage = str(case.get("stage") or "unspecified").strip() or "unspecified"
        vp = parse_variant(mutation)
        pathogenic = vp.klass in ("contact", "conformational", "truncating",
                                  "other_hotspot")

        scenarios: List[Scenario] = []

        # Scenario 1 — standard DNA-damaging chemotherapy.
        if pathogenic:
            scenarios.append(Scenario(
                name="Standard DNA-damaging chemotherapy",
                intervention="Platinum / anthracycline-based regimen",
                illustrative_outcome=("Published cohorts associate TP53 disruption "
                    "with *relatively* poorer response and shorter disease-free "
                    "intervals than TP53-wild-type tumours (population-level, not "
                    "individual)."),
                evidence_basis="TP53 prognostic literature; IARC TP53 database",
                confidence="moderate",
                caveat="Cohort-level trend; many other factors drive an "
                       "individual's response."))
        else:
            scenarios.append(Scenario(
                name="Standard of care by stage",
                intervention="Stage-directed guideline therapy",
                illustrative_outcome=("With an unclassified variant, expected "
                    "outcomes track stage and histology rather than the variant."),
                evidence_basis="NCCN (gene-agnostic) staging principles",
                confidence="moderate",
                caveat="Variant significance not established."))

        # Scenario 2 — p53-reactivation pathway (only if reactivatable).
        if vp.reactivatable:
            scenarios.append(Scenario(
                name="p53-reactivation pathway",
                intervention="Eprenetapopt (APR-246) ± backbone, trial setting",
                illustrative_outcome=("Early-phase trials report biological "
                    "activity for reactivatable mutants; durable benefit remains "
                    "under investigation."),
                evidence_basis="APR-246 / eprenetapopt clinical trials",
                confidence="investigational",
                caveat="Trial-stage; access is limited, especially regionally."))

        # Scenario 3 — synthetic lethality (for loss-of-function contexts).
        if pathogenic:
            scenarios.append(Scenario(
                name="Synthetic-lethal targeting",
                intervention="WEE1 / ATR / CHK1 inhibition (trial)",
                illustrative_outcome=("Pre-clinical and early-clinical data support "
                    "selective vulnerability of p53-deficient cells."),
                evidence_basis="DepMap dependencies; SL trial literature",
                confidence="investigational",
                caveat="Largely trial-stage; not standard of care."))

        # Scenario 4 — surgery + adjuvant by stage.
        early = stage.upper() in ("I", "II", "0")
        scenarios.append(Scenario(
            name="Surgery + adjuvant (stage-directed)",
            intervention=("Curative-intent resection then adjuvant" if early
                          else "Systemic-first; surgery per protocol"),
            illustrative_outcome=("Earlier stage is associated with better "
                "outcomes across cohorts; the molecular profile informs adjuvant "
                "choice." if early else "Advanced stage favours systemic control "
                "first."),
            evidence_basis="AJCC/UICC staging outcome data",
            confidence="moderate",
            caveat="Stage-level association, not an individual forecast."))

        exploration = TwinExploration(
            mutation=vp.raw or str(mutation), cancer=cancer, stage=stage,
            classification=vp.klass, scenarios=[asdict(s) for s in scenarios])
        self._audit(f"twin:{vp.raw} {cancer}/{stage} -> {len(scenarios)} scenarios")
        return {
            **asdict(exploration),
            "scenario_count": len(scenarios),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": (f"{len(scenarios)} illustrative scenario(s) for "
                        f"{vp.raw} in {cancer} ({stage})"),
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Digital-twin audit failed: {e}")


_twin = DigitalTwinExplorer()


def explore_twin(mutation: str, case: Optional[Dict] = None) -> Dict:
    """Module-level convenience wrapper."""
    return _twin.explore(mutation, case)
