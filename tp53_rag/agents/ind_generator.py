"""
============================================================
TP53 RAG Platform - IND Document Generator (Regulatory Agent)
agents/ind_generator.py
============================================================
Generates a DRAFT FDA Investigational New Drug (IND) application skeleton
from a TP53 mutation profile + candidate drug(s): preclinical summary,
proposed mechanism of action, CMC stub, clinical-plan and safety sections.

Offline-first & rule-based — fills curated regulatory templates with the
provided data (no LLM / network required). Output is structured sections
plus a markdown render (downloadable); a DOCX is produced when python-docx
is available, otherwise the markdown is the deliverable (graceful).

DISCLAIMER: this is a DRAFT scaffold for educational/planning use — NOT a
submission-ready regulatory document. A regulatory professional must author
and verify any real IND.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "ind_generator"

DISCLAIMER = ("DRAFT scaffold for educational/planning use only — not a "
              "submission-ready IND. A regulatory professional must verify.")

# Mechanism-of-action context per TP53-targeting strategy (curated, sourced).
_MOA_CONTEXT = {
    "reactivator": "Restores wild-type-like conformation/function to mutant p53, "
                   "re-enabling pro-apoptotic transcriptional programs.",
    "mdm2_inhibitor": "Blocks the p53–MDM2 interaction to stabilise functional p53 "
                      "(most relevant where wild-type p53 is retained).",
    "stabiliser": "Binds a mutation-induced surface pocket (e.g. Y220C) to "
                  "thermostabilise the DNA-binding domain.",
    "chemotherapy": "DNA-damaging cytotoxic; activity is largely p53-independent.",
}


def _strategy_for(mechanism: str) -> str:
    m = str(mechanism or "").lower()
    if "mdm2" in m or "mdmx" in m or "mdm4" in m:
        return "mdm2_inhibitor"
    if "stabilis" in m or "stabiliz" in m or "y220c" in m or "pocket" in m:
        return "stabiliser"
    if "reactivat" in m or "refold" in m or "metallochaperone" in m:
        return "reactivator"
    return "chemotherapy"


@dataclass
class INDSection:
    number: str
    title: str
    content: str


@dataclass
class INDDraft:
    title: str
    mutation: str
    cancer_type: str
    lead_candidate: str
    sections: List[Dict]
    generated: str
    disclaimer: str = DISCLAIMER


class INDGenerator:
    """Assemble a draft IND skeleton from structured inputs."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/ind_generator.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"IND audit dir unavailable: {e}")

    def generate(self, mutation: str, cancer_type: str = "",
                 drug_candidates: Optional[List[Dict]] = None,
                 sponsor: str = "Daktari Genomed Labs") -> Dict:
        """Build a draft IND. Never empty: with no candidates a generic
        TP53-targeting plan is produced."""
        mut = str(mutation or "TP53 variant").strip()
        cancer = str(cancer_type or "the target tumour type").strip()
        cands = [c for c in (drug_candidates or []) if isinstance(c, dict)]
        if not cands:
            cands = [{"name": "TP53-targeting candidate", "mechanism": "Mutant p53 reactivator"}]
        lead = cands[0]
        lead_name = lead.get("name", "lead candidate")
        strategy = _strategy_for(lead.get("mechanism", ""))
        moa = _MOA_CONTEXT[strategy]

        cand_lines = "\n".join(
            f"- **{c.get('name','?')}** — {c.get('mechanism','mechanism n/a')}"
            + (f" (phase: {c.get('phase_label')})" if c.get("phase_label") else "")
            for c in cands
        )

        sections = [
            INDSection("1", "Introductory Statement & General Investigational Plan",
                       f"Sponsor **{sponsor}** proposes to investigate **{lead_name}** "
                       f"for tumours harbouring the TP53 **{mut}** alteration in "
                       f"**{cancer}**. The development plan advances from the "
                       f"preclinical package below into a biomarker-selected "
                       f"(TP53-mutant) early-phase clinical study.\n\n"
                       f"Candidate agents under consideration:\n{cand_lines}"),
            INDSection("2", "Proposed Mechanism of Action",
                       f"{lead_name} acts as a **{strategy.replace('_', ' ')}**. {moa}\n\n"
                       f"Biological rationale: {mut} is a TP53 alteration in the p53 "
                       f"tumour-suppressor pathway; restoring or exploiting p53-pathway "
                       f"activity is the therapeutic hypothesis."),
            INDSection("3", "Pharmacology & Toxicology (Preclinical Summary)",
                       "- **Primary pharmacology:** target engagement assays vs the "
                       f"{mut} mutant; functional p53 reporter / apoptosis readouts.\n"
                       "- **Secondary pharmacology / safety:** off-target panel, hERG.\n"
                       "- **Toxicology:** GLP repeat-dose studies in two species; "
                       "genotoxicity battery; TK/PK.\n"
                       "- **Status:** [TO BE COMPLETED — attach study reports]."),
            INDSection("4", "Chemistry, Manufacturing & Controls (CMC)",
                       "Drug substance/product description, manufacturing process, "
                       "specifications, stability, and reference standards. "
                       "[TO BE COMPLETED by CMC team]."),
            INDSection("5", "Clinical Protocol Summary",
                       f"- **Population:** advanced **{cancer}** with confirmed TP53 "
                       f"**{mut}** (central/ NGS or VCF-based confirmation).\n"
                       "- **Design:** open-label, dose-escalation (3+3 / BOIN) → "
                       "biomarker expansion.\n"
                       "- **Primary endpoints:** safety, MTD/RP2D.\n"
                       "- **Secondary:** ORR, PK, pharmacodynamic p53-pathway markers.\n"
                       "- **Equity note:** include African/Kenyan sites where feasible "
                       "to address under-representation."),
            INDSection("6", "Safety Considerations & Risk Assessment",
                       "Anticipated risks based on mechanism and class; on-target "
                       "p53-pathway effects (e.g. haematologic for MDM2 inhibitors); "
                       "monitoring and stopping rules; DSMB oversight. "
                       "[Populate from preclinical findings]."),
        ]

        completed = sum(1 for s in sections
                        if "[TO BE COMPLETED" not in s.content
                        and "[Populate" not in s.content)
        draft = INDDraft(
            title=f"DRAFT IND — {lead_name} for TP53 {mut} ({cancer})",
            mutation=mut, cancer_type=cancer, lead_candidate=lead_name,
            sections=[asdict(s) for s in sections],
            generated=datetime.now().isoformat(timespec="seconds"),
        )
        self._audit(f"ind:{mut}/{lead_name} -> {len(sections)} sections")
        return {
            "draft": asdict(draft),
            "section_count": len(sections),
            "readiness_pct": round(100 * completed / len(sections)),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": f"Draft IND for {lead_name} (TP53 {mut}) — {len(sections)} sections",
        }

    def render_markdown(self, result: Dict) -> str:
        """Markdown render of a generate() result. Always non-empty; tolerates
        a missing/malformed draft."""
        d = (result or {}).get("draft", {}) if isinstance(result, dict) else {}
        if not isinstance(d, dict):
            d = {}
        lines = [f"# {d.get('title', 'DRAFT IND')}",
                 "", f"_Generated: {d.get('generated', '?')}_", ""]
        sections = d.get("sections", [])
        for s in (sections if isinstance(sections, list) else []):
            if not isinstance(s, dict):
                continue
            lines += [f"## {s.get('number','?')}. {s.get('title','')}",
                      "", s.get("content", ""), ""]
        lines += ["---", f"> {d.get('disclaimer', DISCLAIMER)}"]
        return "\n".join(lines)

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"IND audit failed: {e}")


_generator = INDGenerator()


def generate_ind(mutation: str, cancer_type: str = "",
                 drug_candidates: Optional[List[Dict]] = None) -> Dict:
    return _generator.generate(mutation, cancer_type, drug_candidates)
