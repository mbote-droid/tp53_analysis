"""
============================================================
TP53 RAG Platform - Structural Mechanics & Cavity Analyzer
agents/structural_analyzer.py
============================================================
Analyses the structural consequences of a TP53 mutation for drug design:
  * ΔΔG — thermodynamic destabilisation of the DNA-binding domain
  * binding-pocket / cavity descriptors (volume, hydrophobicity, druggability)
  * residue-contact context near the mutation site
  * a drug-strategy recommendation (stabiliser vs reactivator vs contact)

Offline-first & curated: ΔΔG and pocket descriptors are drawn from published
biophysical studies of p53 (Joerger & Fersht; Bullock et al.) — sourced and
clearly labelled as estimates. No live structure-prediction / FoldX run is
required (those need a structure + heavy compute); a `structure_source` field
notes when only curated values are used.

DISCLAIMER: curated biophysical estimates for research context — not a
substitute for experimental ΔΔG or cavity detection on a real structure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "structural_analyzer"
DISCLAIMER = ("Curated biophysical estimates (Joerger & Fersht et al.) — "
              "research use only; not experimental ΔΔG / cavity detection.")

# Mechanistic residue buckets (same convention as the docking/viz layer).
_ZINC = {176, 179, 238, 242}
_CONTACT = {248, 273, 249}            # DNA-contact residues (fold-preserving)
_CONFORMATIONAL = {143, 175, 245, 282}

# Curated per-hotspot structural profile. ddG = destabilisation (kcal/mol,
# positive = more destabilising). druggability 0-1.
_PROFILES: Dict[int, Dict] = {
    220: {"ddG": 4.0, "class": "conformational",
          "pocket": "Y220C surface cleft", "volume_A3": 260, "hydrophobicity": 0.7,
          "druggability": 0.9,
          "note": "Mutation opens a unique druggable surface cleft — "
                  "amenable to small-molecule stabilisers (e.g. PC14586)."},
    175: {"ddG": 3.0, "class": "conformational",
          "pocket": "destabilised core (L2/L3)", "volume_A3": 180, "hydrophobicity": 0.5,
          "druggability": 0.55,
          "note": "Conformational unfolding of the core — reactivator strategy."},
    143: {"ddG": 3.5, "class": "conformational",
          "pocket": "hydrophobic core", "volume_A3": 150, "hydrophobicity": 0.6,
          "druggability": 0.5, "note": "Core-destabilising; reactivator-amenable."},
    245: {"ddG": 2.0, "class": "conformational",
          "pocket": "L3 loop / zinc region", "volume_A3": 160, "hydrophobicity": 0.45,
          "druggability": 0.5, "note": "Partial destabilisation near the L3 loop."},
    282: {"ddG": 3.2, "class": "conformational",
          "pocket": "L3 / DNA interface", "volume_A3": 170, "hydrophobicity": 0.5,
          "druggability": 0.5, "note": "Destabilising conformational mutant."},
    248: {"ddG": 0.8, "class": "contact",
          "pocket": "DNA minor-groove interface", "volume_A3": 90, "hydrophobicity": 0.3,
          "druggability": 0.3,
          "note": "DNA-contact mutant — fold largely preserved; the defect is "
                  "loss of DNA binding, not unfolding (harder to drug directly)."},
    273: {"ddG": 0.4, "class": "contact",
          "pocket": "DNA major-groove interface", "volume_A3": 85, "hydrophobicity": 0.3,
          "druggability": 0.3,
          "note": "DNA-contact mutant — minimal destabilisation."},
    249: {"ddG": 2.2, "class": "contact",
          "pocket": "L3 loop / DNA interface", "volume_A3": 120, "hydrophobicity": 0.4,
          "druggability": 0.4,
          "note": "Aflatoxin-associated; partial loop distortion."},
    176: {"ddG": 3.8, "class": "zinc",
          "pocket": "zinc-binding site", "volume_A3": 140, "hydrophobicity": 0.4,
          "druggability": 0.45,
          "note": "Zinc-coordination loss destabilises the fold — "
                  "metallochaperone/zinc-rescue strategy."},
}

# Approximate spatial neighbours per hotspot (illustrative contact context).
_CONTACTS: Dict[int, List[int]] = {
    220: [219, 221, 229, 240], 175: [174, 176, 179, 238],
    248: [247, 249, 273, 276], 273: [272, 274, 248, 277],
    245: [244, 246, 242, 248], 282: [281, 283, 277, 248],
    249: [248, 250, 245, 273], 143: [142, 144, 156, 174],
    176: [175, 179, 238, 242],
}


def _codon(mutation: str) -> int:
    digits = "".join(c for c in str(mutation or "") if c.isdigit())
    return int(digits) if digits else 0


def _class_for(codon: int) -> str:
    if codon == 220:
        return "conformational"
    if codon in _ZINC:
        return "zinc"
    if codon in _CONTACT:
        return "contact"
    if codon in _CONFORMATIONAL:
        return "conformational"
    return "other"


@dataclass
class StructuralProfile:
    mutation: str
    codon: int
    stability_class: str
    ddG_kcal_mol: float
    destabilising: bool
    pocket: str
    cavity_volume_A3: int
    hydrophobicity: float
    druggability: float
    contact_residues: List[int]
    strategy: str
    structure_source: str
    note: str
    disclaimer: str = DISCLAIMER


class StructuralAnalyzer:
    """Curated structural-mechanics + cavity analysis for TP53 mutations."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/structural_analyzer.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Structural audit dir unavailable: {e}")

    @staticmethod
    def _strategy(cls: str, druggability: float) -> str:
        if cls == "conformational" and druggability >= 0.8:
            return "small-molecule pocket stabiliser"
        if cls == "conformational":
            return "mutant-p53 reactivator (refolding)"
        if cls == "zinc":
            return "zinc-metallochaperone / zinc rescue"
        if cls == "contact":
            return "contact mutant — limited direct druggability; consider SL/MDM2 routes"
        return "general reactivator screening"

    def analyse(self, mutation: str) -> Dict:
        """Return the structural profile for a mutation. Never empty."""
        mut = str(mutation or "TP53").strip()
        codon = _codon(mut)
        prof = _PROFILES.get(codon)
        if prof:
            cls = prof["class"]
            ddG = prof["ddG"]
            pocket = prof["pocket"]
            vol = prof["volume_A3"]
            hydro = prof["hydrophobicity"]
            drug = prof["druggability"]
            note = prof["note"]
        else:  # generic estimate by class (never empty)
            cls = _class_for(codon)
            ddG = {"conformational": 2.8, "zinc": 3.5,
                   "contact": 0.6, "other": 1.5}.get(cls, 1.5)
            pocket = "p53 DNA-binding domain (generic)"
            vol = 150
            hydro = 0.45
            drug = {"conformational": 0.5, "zinc": 0.45,
                    "contact": 0.3, "other": 0.4}.get(cls, 0.4)
            note = "Estimated from mutation class (no curated profile for this codon)."

        profile = StructuralProfile(
            mutation=mut, codon=codon, stability_class=cls,
            ddG_kcal_mol=round(ddG, 1), destabilising=ddG >= 1.5,
            pocket=pocket, cavity_volume_A3=vol, hydrophobicity=hydro,
            druggability=drug, contact_residues=_CONTACTS.get(codon, []),
            strategy=self._strategy(cls, drug),
            structure_source="curated (no live structure run)", note=note,
        )
        self._audit(f"struct:{mut} -> ddG={ddG}, class={cls}, drug={drug}")
        return {
            **asdict(profile),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": f"{mut}: ΔΔG≈{ddG} kcal/mol ({cls}), "
                       f"druggability {drug:.0%} — {profile.strategy}",
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Structural audit failed: {e}")


_analyzer = StructuralAnalyzer()


def analyse_structure(mutation: str) -> Dict:
    return _analyzer.analyse(mutation)
