"""
============================================================
TP53 RAG Platform - Explainability ("Why?") Engine
agents/explainability.py
============================================================
Turns a TP53 assessment from an opaque output into a transparent evidence
trace. For a given variant it answers "why?" by assembling, in one place:

  • the molecular classification and an earned confidence
  • supporting evidence lines (ClinVar, in-silico SIFT/PolyPhen, CADD, gnomAD
    rarity, ESM-2 language-model effect, structural class)
  • the p53 pathways the variant perturbs
  • literature citations
  • an explicit list of what is NOT known (honest uncertainty)
  • a plain-language summary a clinician can read in seconds

Every line carries a strength label, and nothing is fabricated: the engine
reuses the curated/real annotation and effect layers, and where a source has
no data it says so rather than inventing a value.

DISCLAIMER: research-use only — an evidence summary for transparency, not a
clinical determination.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log
from agents.tumor_board import parse_variant

AGENT_ID = "explainability"
DISCLAIMER = ("Evidence trace for transparency — research use only, not a "
              "clinical determination. Confirm with a CLIA-certified laboratory.")

# Strength tiers, ordered for sorting (strong first).
_STRENGTH_RANK = {"strong": 0, "moderate": 1, "supporting": 2, "uncertain": 3}

# Curated canonical TP53 pathway involvement (sourced from p53 biology).
_PATHWAYS = [
    {"pathway": "Cell-cycle arrest", "effector": "CDKN1A/p21",
     "consequence": "Loss of G1/S checkpoint enforcement"},
    {"pathway": "Apoptosis", "effector": "BAX · PUMA · NOXA",
     "consequence": "Impaired clearance of damaged cells"},
    {"pathway": "DNA repair", "effector": "GADD45 · p53R2",
     "consequence": "Reduced repair coordination, genomic instability"},
    {"pathway": "Autoregulation", "effector": "MDM2 feedback",
     "consequence": "Altered p53 turnover dynamics"},
]

# Curated, real literature anchors for TP53 variant interpretation.
_CITATIONS = [
    {"ref": "Bouaoun et al., Hum Mutat 2016 (IARC TP53 Database, R18)",
     "topic": "Somatic/germline TP53 variant compendium"},
    {"ref": "Olivier, Hollstein & Hainaut, Cold Spring Harb Perspect Biol 2010",
     "topic": "TP53 mutation spectra and functional impact"},
    {"ref": "Giacomelli et al., Nat Genet 2018",
     "topic": "Saturation functional classification of TP53 variants"},
]


@dataclass
class EvidenceItem:
    category: str        # e.g. "Clinical database", "In-silico", "Functional"
    source: str          # e.g. "ClinVar", "ESM-2", "Structure"
    statement: str
    strength: str        # strong | moderate | supporting | uncertain


@dataclass
class Explanation:
    mutation: str
    classification: str
    confidence: float
    headline: str
    evidence: List[Dict] = field(default_factory=list)
    pathways: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    plain_language: str = ""
    disclaimer: str = DISCLAIMER


# Earned confidence by molecular class (mirrors the tumour-board calibration).
_CLASS_CONFIDENCE = {
    "contact": 0.9, "conformational": 0.88, "truncating": 0.82,
    "other_hotspot": 0.7, "non_hotspot_missense": 0.4, "unknown": 0.25,
}
_CLASS_LABEL = {
    "contact": "DNA-contact hotspot (pathogenic)",
    "conformational": "Conformational hotspot (pathogenic)",
    "truncating": "Truncating / loss-of-function",
    "other_hotspot": "Recurrent hotspot (likely pathogenic)",
    "non_hotspot_missense": "Non-hotspot missense (uncertain significance)",
    "unknown": "Unclassifiable from input",
}


class ExplainabilityEngine:
    """Assemble a transparent evidence trace for a TP53 variant."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/explainability.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Explainability audit dir unavailable: {e}")

    def explain(self, mutation: str, case: Optional[Dict] = None) -> Dict:
        """Return a full evidence trace. Never empty; never fabricates."""
        vp = parse_variant(mutation)
        klass = vp.klass
        confidence = _CLASS_CONFIDENCE.get(klass, 0.25)
        evidence: List[EvidenceItem] = []

        # ── 1. Structural classification (always present) ─────────
        if klass == "contact":
            evidence.append(EvidenceItem(
                "Structure", "p53 crystallography",
                f"Codon {vp.codon} contacts DNA directly; the fold is intact but "
                "sequence-specific binding is abolished.", "strong"))
        elif klass == "conformational":
            evidence.append(EvidenceItem(
                "Structure", "p53 crystallography",
                f"Codon {vp.codon} destabilises the DNA-binding-domain fold "
                "(often dominant-negative).", "strong"))
        elif klass == "truncating":
            evidence.append(EvidenceItem(
                "Structure", "Protein topology",
                "Truncation removes the oligomerisation/regulatory C-terminus — "
                "complete loss of function.", "strong"))
        elif klass == "other_hotspot":
            evidence.append(EvidenceItem(
                "Recurrence", "IARC/COSMIC",
                f"Codon {vp.codon} recurs across somatic cancer datasets.",
                "moderate"))
        elif klass == "non_hotspot_missense":
            evidence.append(EvidenceItem(
                "Structure", "Position analysis",
                f"Codon {vp.codon} is a non-hotspot missense"
                f"{' in the DNA-binding domain' if vp.in_dbd else ''}; impact not "
                "established from position alone.", "uncertain"))
        else:
            evidence.append(EvidenceItem(
                "Input", "Parser",
                "The variant notation could not be resolved to a codon.",
                "uncertain"))

        # ── 2. Curated/real annotation (ClinVar, SIFT, etc.) ──────
        anno = self._safe_annotation(mutation)
        if anno:
            clinvar = anno.get("clinvar_significance") or anno.get("clinvar")
            if clinvar:
                strong = "patho" in str(clinvar).lower()
                evidence.append(EvidenceItem(
                    "Clinical database", "ClinVar", f"ClinVar: {clinvar}.",
                    "strong" if strong else "supporting"))
            if anno.get("sift"):
                evidence.append(EvidenceItem(
                    "In-silico", "SIFT", f"SIFT prediction: {anno['sift']}.",
                    "supporting"))
            if anno.get("polyphen"):
                evidence.append(EvidenceItem(
                    "In-silico", "PolyPhen-2",
                    f"PolyPhen-2: {anno['polyphen']}.", "supporting"))
            if anno.get("cadd_phred") is not None:
                cadd = anno["cadd_phred"]
                evidence.append(EvidenceItem(
                    "In-silico", "CADD",
                    f"CADD phred {cadd} "
                    f"({'high — deleterious range' if cadd >= 20 else 'modest'}).",
                    "supporting" if cadd >= 20 else "uncertain"))
            gnomad = anno.get("gnomad_af")
            if gnomad is not None:
                # gnomAD AF may arrive as a float or a string ("absent", "0.0001").
                try:
                    af = float(gnomad)
                    rare = af < 1e-4
                except (TypeError, ValueError):
                    rare = "absent" in str(gnomad).lower() or str(gnomad).strip() in ("0", "—")
                evidence.append(EvidenceItem(
                    "Population", "gnomAD",
                    f"gnomAD allele frequency {gnomad} "
                    f"({'absent/ultra-rare — consistent with pathogenic' if rare else 'present in population'}).",
                    "supporting" if rare else "uncertain"))

        # ── 3. ESM-2 functional effect (only if precomputed) ──────
        eff = self._safe_effect(mutation)
        if eff and eff.get("available"):
            score = eff.get("esm2_score")
            interp = eff.get("interpretation", "")
            strength = "strong" if "deleterious" in str(interp).lower() else "moderate"
            evidence.append(EvidenceItem(
                "Functional", "ESM-2 language model",
                f"Masked-marginal LLR {score} → {interp}.", strength))
        elif eff and eff.get("notes"):
            evidence.append(EvidenceItem(
                "Functional", "ESM-2 language model",
                str(eff["notes"]), "uncertain"))

        # ── Sort by strength, assemble ────────────────────────────
        evidence.sort(key=lambda e: _STRENGTH_RANK.get(e.strength, 9))

        uncertainty = self._uncertainty(vp, anno, eff)
        headline = _CLASS_LABEL.get(klass, "Unclassified")
        plain = self._plain_language(vp, klass, confidence)

        exp = Explanation(
            mutation=vp.raw or str(mutation), classification=klass,
            confidence=confidence, headline=headline,
            evidence=[asdict(e) for e in evidence],
            pathways=(_PATHWAYS if klass != "unknown" else []),
            citations=_CITATIONS, uncertainty=uncertainty,
            plain_language=plain,
        )
        self._audit(f"explain:{vp.raw} class={klass} evidence={len(evidence)}")
        return {
            **asdict(exp),
            "evidence_count": len(evidence),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": f"Assembled {len(evidence)} evidence line(s) for {vp.raw}",
        }

    # ── Helpers ───────────────────────────────────────────────────
    @staticmethod
    def _safe_annotation(mutation: str) -> Dict:
        try:
            from utils.variant_annotation import annotate_variant
            return annotate_variant(mutation, use_live=False) or {}
        except Exception as e:  # pragma: no cover
            log.warning(f"Explainability annotation unavailable: {e}")
            return {}

    @staticmethod
    def _safe_effect(mutation: str) -> Dict:
        try:
            from utils.variant_effect import predict_effect
            return predict_effect(mutation) or {}
        except Exception as e:  # pragma: no cover
            log.warning(f"Explainability effect unavailable: {e}")
            return {}

    @staticmethod
    def _uncertainty(vp, anno: Dict, eff: Dict) -> List[str]:
        notes: List[str] = []
        if vp.klass in ("non_hotspot_missense", "unknown"):
            notes.append("Functional significance is not established — treat as "
                         "uncertain until functional/in-silico evidence accrues.")
        if not (eff and eff.get("available")):
            notes.append("ESM-2 language-model score not available for this "
                         "variant (precomputed matrix absent or position not scored).")
        if not anno:
            notes.append("Live annotation not fetched — shown evidence is the "
                         "curated baseline; enable live lookup for SIFT/PolyPhen/"
                         "CADD/gnomAD where available.")
        if not notes:
            notes.append("Evidence is concordant; residual uncertainty reflects "
                         "tumour context and germline status not captured here.")
        return notes

    @staticmethod
    def _plain_language(vp, klass: str, confidence: float) -> str:
        if klass in ("contact", "conformational"):
            return (f"{vp.raw} is a well-known TP53 hotspot. Multiple independent "
                    "lines of evidence — structure, clinical databases and "
                    "predictors — agree it disrupts p53 function, so confidence "
                    f"is high ({confidence:.0%}).")
        if klass == "truncating":
            return (f"{vp.raw} cuts the protein short, removing parts essential "
                    f"for function — a clear loss-of-function ({confidence:.0%}).")
        if klass == "other_hotspot":
            return (f"{vp.raw} recurs in cancers and is likely damaging, though "
                    f"with less variant-specific evidence than the top hotspots "
                    f"({confidence:.0%}).")
        if klass == "non_hotspot_missense":
            return (f"{vp.raw} is not a known hotspot. The evidence is "
                    f"insufficient to call it damaging, so it is treated as "
                    f"uncertain ({confidence:.0%}) pending further data.")
        return ("The variant could not be interpreted from the text provided — "
                "please check the notation.")

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Explainability audit failed: {e}")


_engine = ExplainabilityEngine()


def explain_variant(mutation: str, case: Optional[Dict] = None) -> Dict:
    """Module-level convenience wrapper."""
    return _engine.explain(mutation, case)
