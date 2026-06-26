"""
============================================================
TP53 RAG Platform - Live AI Tumour Board
agents/tumor_board.py
============================================================
Convenes a virtual multidisciplinary tumour board for a TP53-mutant case.
A panel of specialist members (Pathologist, Geneticist, Oncologist, Surgeon,
Pharmacologist, Equity Officer) each forms an opinion from the variant's
established molecular properties, the members cross-examine each other, then
vote toward a consensus management recommendation with a calibrated
confidence.

Offline-first & curated: every opinion is derived from a sourced, rule-based
reading of well-characterised TP53 biology (mutation class, domain, stage).
No LLM or network is required; an optional narration layer can enrich the
prose when a model is available, but the debate and vote are deterministic
and unit-tested.

Confidence is *earned*, not asserted: well-characterised hotspots yield high
member confidence; variants of uncertain significance yield low confidence
and a consensus that explicitly recommends reclassification before acting.

DISCLAIMER: research-use only. A simulated panel for education and decision
support — not a real tumour board and not a clinical directive.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "tumor_board"
DISCLAIMER = ("Simulated multidisciplinary tumour board for education and "
              "decision support — research use only, not a clinical directive "
              "and not a substitute for a real tumour board.")

# ── Variant biology (consistent with utils/viz.py classification) ──
CONTACT_CODONS = {248, 273, 249}          # retain fold, lose DNA contact
CONFORMATIONAL_CODONS = {143, 175, 245, 282}   # destabilise the fold
OTHER_HOTSPOT_CODONS = {220, 213, 196, 158}
DBD_RANGE = (94, 292)                      # DNA-binding domain

# Recommendation themes the panel can vote for.
THEME_REACTIVATION = "p53-reactivation pathway (trial / molecular board referral)"
THEME_STANDARD = "stage-directed standard of care with TP53-aware prognosis"
THEME_RECLASSIFY = "reclassify the variant before acting (insufficient evidence)"


@dataclass
class MemberOpinion:
    member: str
    specialty: str
    icon: str
    stance: str
    rationale: str
    evidence: List[str]
    recommendation: str          # one of the THEME_* constants
    confidence: float            # 0.0–1.0, earned from evidence strength
    concerns: List[str] = field(default_factory=list)


@dataclass
class BoardConsensus:
    recommendation: str
    confidence: float
    agreement_ratio: float       # share of members backing the consensus
    dissents: List[str]
    key_evidence: List[str]
    rationale: str


@dataclass
class VariantProfile:
    raw: str
    codon: Optional[int]
    ref_aa: Optional[str]
    alt_aa: Optional[str]
    klass: str                   # contact | conformational | other_hotspot |
                                 # non_hotspot_missense | truncating | unknown
    in_dbd: bool
    reactivatable: bool


def parse_variant(mutation: str) -> VariantProfile:
    """Parse a protein change (e.g. 'R175H', 'p.R248Q', 'R213*') into a
    structured profile. Defensive — unparseable input yields an 'unknown'
    profile rather than raising."""
    raw = str(mutation or "").strip()
    s = raw.replace("p.", "").replace("(", "").replace(")", "")

    truncating = bool(re.search(r"(\*|fs|Ter|X)$", s, flags=re.IGNORECASE))
    m = re.match(r"^([A-Za-z])?\s*(\d+)\s*([A-Za-z\*]+)?", s)
    codon = ref_aa = alt_aa = None
    if m:
        ref_aa = (m.group(1) or "").upper() or None
        try:
            codon = int(m.group(2))
        except (TypeError, ValueError):
            codon = None
        alt_aa = (m.group(3) or "").upper() or None

    in_dbd = codon is not None and DBD_RANGE[0] <= codon <= DBD_RANGE[1]

    if codon is None:
        klass = "unknown"
    elif truncating:
        klass = "truncating"
    elif codon in CONTACT_CODONS:
        klass = "contact"
    elif codon in CONFORMATIONAL_CODONS:
        klass = "conformational"
    elif codon in OTHER_HOTSPOT_CODONS:
        klass = "other_hotspot"
    else:
        klass = "non_hotspot_missense"

    # Reactivation (APR-246/eprenetapopt) is best evidenced for DNA-contact and
    # certain structural mutants that retain a near-native scaffold.
    reactivatable = klass in ("contact", "conformational", "other_hotspot")

    return VariantProfile(
        raw=raw, codon=codon, ref_aa=ref_aa, alt_aa=alt_aa,
        klass=klass, in_dbd=in_dbd, reactivatable=reactivatable,
    )


class TumorBoard:
    """Convene a simulated multidisciplinary tumour board for a TP53 case."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/tumor_board.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Tumour-board audit dir unavailable: {e}")

    # ── Per-member reasoning ──────────────────────────────────────
    def _pathologist(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        if vp.klass == "contact":
            stance = "Aggressive molecular phenotype likely"
            rationale = (f"Codon {vp.codon} is a DNA-contact mutant: the fold is "
                         "preserved but sequence-specific DNA binding is lost, "
                         "typically a high-grade, loss-of-function pattern.")
            conf = 0.86
        elif vp.klass == "conformational":
            stance = "Aggressive molecular phenotype likely"
            rationale = (f"Codon {vp.codon} is a conformational mutant: the core "
                         "fold is destabilised (often dominant-negative), "
                         "correlating with poorly differentiated histology.")
            conf = 0.84
        elif vp.klass == "truncating":
            stance = "Loss-of-function phenotype"
            rationale = ("A truncating change removes the oligomerisation/"
                         "regulatory C-terminus — complete loss of suppressor "
                         "function expected.")
            conf = 0.8
        elif vp.klass == "non_hotspot_missense":
            stance = "Indeterminate histologic impact"
            rationale = (f"Codon {vp.codon} is a non-hotspot missense in "
                         f"{'the DBD' if vp.in_dbd else 'a non-core region'}; "
                         "functional impact is not established from position alone.")
            conf = 0.45
        else:
            stance = "Insufficient data for a histologic call"
            rationale = ("Variant could not be localised to a known structural "
                         "class; defer to IHC (p53 pattern) and grading.")
            conf = 0.3
        return MemberOpinion(
            member="Pathologist", specialty="Histopathology", icon="🔬",
            stance=stance, rationale=rationale,
            evidence=["IARC TP53 database", "p53 structure-function literature"],
            recommendation=(THEME_RECLASSIFY if vp.klass in ("unknown",
                            "non_hotspot_missense") else THEME_STANDARD),
            confidence=conf,
            concerns=([f"Confirm p53 IHC pattern correlates with {vp.klass} class"]
                      if vp.klass not in ("unknown",) else
                      ["Variant localisation uncertain"]),
        )

    def _geneticist(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        if vp.klass in ("contact", "conformational"):
            stance = "Pathogenic"
            rationale = (f"{vp.raw} is a canonical TP53 hotspot with concordant "
                         "ClinVar/IARC pathogenic classification.")
            conf = 0.9
            rec = THEME_REACTIVATION if vp.reactivatable else THEME_STANDARD
        elif vp.klass == "other_hotspot":
            stance = "Likely pathogenic"
            rationale = (f"Codon {vp.codon} recurs in somatic cancer datasets; "
                         "pathogenic but with less individual-variant evidence "
                         "than R175/R248/R273.")
            conf = 0.7
            rec = THEME_REACTIVATION if vp.reactivatable else THEME_STANDARD
        elif vp.klass == "truncating":
            stance = "Pathogenic (loss of function)"
            rationale = ("Nonsense/frameshift in TP53 is loss-of-function; "
                         "consider germline testing for Li-Fraumeni if early-onset.")
            conf = 0.82
            rec = THEME_STANDARD
        elif vp.klass == "non_hotspot_missense":
            stance = "Uncertain significance (VUS)"
            rationale = ("No established pathogenic classification for this "
                         "position; in-silico and functional evidence needed.")
            conf = 0.35
            rec = THEME_RECLASSIFY
        else:
            stance = "Unclassifiable from input"
            rationale = "Variant notation could not be resolved to a codon."
            conf = 0.25
            rec = THEME_RECLASSIFY
        concerns = []
        if vp.klass in ("contact", "conformational", "truncating"):
            concerns.append("Offer germline TP53 testing / genetic counselling "
                            "if young age or family history (Li-Fraumeni).")
        return MemberOpinion(
            member="Clinical Geneticist", specialty="Molecular genetics",
            icon="🧬", stance=stance, rationale=rationale,
            evidence=["ClinVar", "IARC TP53", "ACMG criteria"],
            recommendation=rec, confidence=conf, concerns=concerns,
        )

    def _oncologist(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        cancer = str(case.get("cancer") or "the tumour").strip() or "the tumour"
        if vp.klass in ("contact", "conformational", "truncating", "other_hotspot"):
            stance = "Adverse prognosis; expect chemoresistance"
            rationale = (f"TP53 disruption in {cancer} is associated with poorer "
                         "outcomes and relative resistance to DNA-damaging "
                         "chemotherapy; favour TP53-aware regimens and trials.")
            conf = 0.74
            rec = (THEME_REACTIVATION if vp.reactivatable else THEME_STANDARD)
            concerns = ["Avoid over-reliance on single-agent DNA-damaging chemo"]
        elif vp.klass == "non_hotspot_missense":
            stance = "Prognostic impact uncertain"
            rationale = ("Without a functional classification, the prognostic "
                         "weight of this variant is unclear; stage drives therapy.")
            conf = 0.45
            rec = THEME_STANDARD
            concerns = ["Do not escalate therapy on an unclassified variant alone"]
        else:
            stance = "Cannot prognosticate from variant"
            rationale = "Insufficient variant information to weigh prognosis."
            conf = 0.3
            rec = THEME_RECLASSIFY
            concerns = ["Stage-directed therapy pending variant clarification"]
        return MemberOpinion(
            member="Medical Oncologist", specialty="Systemic therapy", icon="⚕️",
            stance=stance, rationale=rationale,
            evidence=["NCCN guidance (gene-agnostic)", "TP53 prognostic literature"],
            recommendation=rec, confidence=conf, concerns=concerns,
        )

    def _surgeon(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        stage = str(case.get("stage") or "").upper().strip()
        early = stage in ("I", "II", "STAGE I", "STAGE II", "0") or stage.startswith(("I ", "II "))
        metastatic = stage in ("IV", "STAGE IV") or "IV" in stage
        if metastatic:
            stance = "Resection not primary; systemic-first"
            rationale = ("Metastatic stage favours systemic therapy; surgery "
                         "reserved for palliation or oligometastatic protocols.")
            conf = 0.7
            rec = THEME_STANDARD
        elif early:
            stance = "Resection with curative intent feasible"
            rationale = ("Early stage supports upfront resection with adequate "
                         "margins; molecular profile informs adjuvant therapy.")
            conf = 0.72
            rec = THEME_STANDARD
        else:
            stance = "Resectability depends on staging"
            rationale = ("Stage not provided or locally advanced — resectability "
                         "and neoadjuvant sequencing require full staging.")
            conf = 0.5
            rec = THEME_STANDARD
        return MemberOpinion(
            member="Surgical Oncologist", specialty="Operative oncology",
            icon="🔪", stance=stance, rationale=rationale,
            evidence=["AJCC/UICC staging principles", "operative oncology practice"],
            recommendation=rec, confidence=conf,
            concerns=(["Confirm full TNM staging before an operative decision"]
                      if not (early or metastatic) else []),
        )

    def _pharmacologist(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        if vp.reactivatable:
            stance = "Candidate for p53-reactivation strategy"
            rationale = (f"{vp.raw} retains a near-native scaffold amenable to "
                         "structural reactivation; eprenetapopt (APR-246) and "
                         "related agents are in trials for such mutants.")
            conf = 0.68
            rec = THEME_REACTIVATION
            concerns = ["Reactivation agents remain investigational — trial context"]
        elif vp.klass == "truncating":
            stance = "Reactivation unlikely; consider synthetic lethality"
            rationale = ("A truncated protein is a poor reactivation substrate; "
                         "WEE1/ATR/CHK1 synthetic-lethal approaches are more rational.")
            conf = 0.6
            rec = THEME_STANDARD
            concerns = ["Synthetic-lethal agents largely trial-stage"]
        else:
            stance = "No targeted p53 rationale established"
            rationale = ("Without a defined functional class, no specific p53-"
                         "targeted pharmacology is justified.")
            conf = 0.4
            rec = THEME_RECLASSIFY
            concerns = ["Avoid off-evidence targeted therapy"]
        return MemberOpinion(
            member="Clinical Pharmacologist", specialty="Targeted therapy",
            icon="💊", stance=stance, rationale=rationale,
            evidence=["ChEMBL", "ClinicalTrials.gov", "p53-reactivation trials"],
            recommendation=rec, confidence=conf, concerns=concerns,
        )

    def _equity(self, vp: VariantProfile, case: Dict) -> MemberOpinion:
        rationale = ("Many targeted/trial options are unavailable in low-resource "
                     "settings. Prioritise interventions deliverable locally and "
                     "name affordable, guideline-based alternatives.")
        if vp.reactivatable:
            stance = "Flag access gap for trial-only options"
            concerns = ["p53-reactivation trials rarely accessible regionally — "
                        "ensure a standard-of-care fallback is also planned",
                        "Cost/curated availability check (e.g. KEML formulary)"]
            conf = 0.62
        else:
            stance = "Favour locally deliverable standard of care"
            concerns = ["Confirm regimen availability and affordability locally"]
            conf = 0.6
        return MemberOpinion(
            member="Equity Officer", specialty="Access & affordability",
            icon="🌍", stance=stance, rationale=rationale,
            evidence=["regional formulary/availability data", "WHO EML"],
            recommendation=THEME_STANDARD, confidence=conf, concerns=concerns,
        )

    # ── Debate + consensus ────────────────────────────────────────
    @staticmethod
    def _debate(opinions: List[MemberOpinion]) -> List[Dict]:
        """Generate cross-examination exchanges: who agrees with whom, and who
        raises a challenge. Deterministic and grounded in the members' votes."""
        exchanges: List[Dict] = []
        tally: Dict[str, List[str]] = {}
        for o in opinions:
            tally.setdefault(o.recommendation, []).append(o.member)

        # Agreements: the largest bloc affirms a shared line.
        if tally:
            top_theme = max(tally, key=lambda k: len(tally[k]))
            backers = tally[top_theme]
            if len(backers) > 1:
                exchanges.append({
                    "type": "agreement",
                    "members": backers,
                    "text": f"{', '.join(backers)} align on: {top_theme}.",
                })
            # Challenges: dissenters question the majority.
            for theme, members in tally.items():
                if theme != top_theme:
                    for m in members:
                        exchanges.append({
                            "type": "challenge",
                            "members": [m],
                            "text": (f"{m} pushes back, favouring instead: "
                                     f"{theme}."),
                        })
        if not exchanges:
            exchanges.append({
                "type": "note", "members": [o.member for o in opinions],
                "text": "The panel reached an early unanimous position.",
            })
        return exchanges

    @staticmethod
    def _consensus(opinions: List[MemberOpinion]) -> BoardConsensus:
        """Confidence-weighted vote. Each member's vote weight is their own
        confidence, so a low-confidence member sways the result less."""
        weights: Dict[str, float] = {}
        for o in opinions:
            weights[o.recommendation] = weights.get(o.recommendation, 0.0) + o.confidence
        winner = max(weights, key=weights.get)
        backers = [o for o in opinions if o.recommendation == winner]
        mean_backer_conf = sum(o.confidence for o in backers) / max(len(backers), 1)
        agreement_ratio = len(backers) / max(len(opinions), 1)
        # Consensus confidence = how sure the backers are, scaled by how unified
        # the panel is (a split panel is reported as less certain, honestly).
        # Ranges from 0.5× (lone voice) to 1.0× (unanimous) of mean backer conf.
        consensus_conf = round(mean_backer_conf * (0.5 + 0.5 * agreement_ratio), 2)

        dissents = [f"{o.member}: {o.recommendation}" for o in opinions
                    if o.recommendation != winner]
        key_evidence = sorted({e for o in backers for e in o.evidence})

        if winner == THEME_RECLASSIFY:
            rationale = ("The panel cannot responsibly commit to a therapy: the "
                         "variant's functional significance is not established. "
                         "Recommend functional/in-silico reclassification first.")
        elif winner == THEME_REACTIVATION:
            rationale = ("The panel supports referral toward a p53-reactivation "
                         "pathway (molecular tumour board / trial), alongside a "
                         "stage-directed standard-of-care backbone.")
        else:
            rationale = ("The panel converges on stage-directed standard of care "
                         "with TP53-aware prognosis and surveillance.")

        return BoardConsensus(
            recommendation=winner, confidence=consensus_conf,
            agreement_ratio=round(agreement_ratio, 2), dissents=dissents,
            key_evidence=key_evidence, rationale=rationale,
        )

    def convene(self, mutation: str, case: Optional[Dict] = None) -> Dict:
        """Run the full board: opinions → debate → consensus. Never empty."""
        case = dict(case or {})
        case.setdefault("mutation", mutation)
        vp = parse_variant(mutation)

        opinions = [
            self._pathologist(vp, case),
            self._geneticist(vp, case),
            self._oncologist(vp, case),
            self._surgeon(vp, case),
            self._pharmacologist(vp, case),
            self._equity(vp, case),
        ]
        debate = self._debate(opinions)
        consensus = self._consensus(opinions)

        self._audit(f"board:{vp.raw} class={vp.klass} -> "
                    f"{consensus.recommendation} @ {consensus.confidence}")
        return {
            "mutation": vp.raw,
            "variant_profile": asdict(vp),
            "members": [asdict(o) for o in opinions],
            "debate": debate,
            "consensus": asdict(consensus),
            "disclaimer": DISCLAIMER,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": (f"Tumour board reached consensus for {vp.raw}: "
                        f"{consensus.recommendation} "
                        f"(confidence {consensus.confidence:.0%})"),
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Tumour-board audit failed: {e}")


_board = TumorBoard()


def convene_tumor_board(mutation: str, case: Optional[Dict] = None) -> Dict:
    """Module-level convenience wrapper."""
    return _board.convene(mutation, case)
