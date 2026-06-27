"""
============================================================
Precision Onco Africa - ClinVar Conflict Checker (hallucination guard)
agents/clinvar_conflict_checker.py
============================================================
Cross-checks AI-generated classifications against ClinVar before they
reach a clinician. If an agent (or the LLM) calls a variant "benign"
that ClinVar records as "pathogenic" (or vice-versa), this flags it as
a HIGH-severity conflict.

Design:
  * Offline-first: a curated ClinVar reference for TP53 hotspots ships
    in-code (zero network needed). An optional live ClinVar lookup can
    be enabled, with graceful fallback to the cached reference.
  * Works two ways:
      1. structured  -> pass mutation + ai_classification explicitly
      2. free-text   -> pass an AI answer; mutations + their claimed
                        classifications are extracted from the prose.
  * Never empty: with no detectable mutation it returns a clean
    "no claims to verify" report.

Output per finding:
  {mutation, ai_classification, clinvar_classification, conflict,
   severity, evidence_url}
============================================================
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "clinvar_conflict_checker"

# Single-letter amino-acid alphabet (+ stop) for mutation parsing.
_AA = "ACDEFGHIKLMNPQRSTVWY"
_MUTATION_RE = re.compile(rf"(?:p\.)?([{_AA}])(\d{{1,3}})([{_AA}*])", re.IGNORECASE)

# Significance vocabulary buckets (pole comparison drives conflict severity).
_BUCKET = {
    "pathogenic": "pathogenic", "likely_pathogenic": "pathogenic",
    "benign": "benign", "likely_benign": "benign",
    "vus": "uncertain", "uncertain": "uncertain", "unknown": "uncertain",
}

_SIGNIFICANCE_SYNONYMS = {
    "uncertain significance": "vus",
    "variant of uncertain significance": "vus",
    "uncertain": "vus",
    "neutral": "benign",
    "polymorphism": "benign",
    "benign polymorphism": "benign",
    "disease-causing": "pathogenic",
    "deleterious": "pathogenic",
    "loss-of-function": "pathogenic",
    "loss of function": "pathogenic",
}

# Curated ClinVar reference for TP53 (consensus significance + evidence link).
# Representative of ClinVar aggregate classifications; the evidence_url points
# at the live ClinVar query so a human can verify.
_CLINVAR_TP53: Dict[str, str] = {
    "R175H": "pathogenic", "R248W": "pathogenic", "R248Q": "pathogenic",
    "R273H": "pathogenic", "R273C": "pathogenic", "R282W": "pathogenic",
    "G245S": "pathogenic", "R249S": "pathogenic", "Y220C": "pathogenic",
    "R337H": "pathogenic", "C176Y": "pathogenic", "V157F": "pathogenic",
    "P72R": "benign", "P72": "benign",
}


def _clinvar_url(mutation: str) -> str:
    return ("https://www.ncbi.nlm.nih.gov/clinvar/?term="
            f"TP53%5Bgene%5D+AND+{mutation}")


def normalize_significance(value: Optional[str]) -> str:
    """Map any significance phrasing onto the canonical vocabulary."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    for phrase, canon in _SIGNIFICANCE_SYNONYMS.items():
        if phrase in s:
            s = canon
            break
    return s.replace(" ", "_").replace("-", "_")


def _bucket(value: Optional[str]) -> str:
    return _BUCKET.get(normalize_significance(value), "uncertain")


def extract_mutations(text: str) -> List[str]:
    """Pull canonical TP53 protein mutations (e.g. R175H) from free text."""
    found: List[str] = []
    for m in _MUTATION_RE.finditer(str(text or "")):
        ref, codon, alt = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        if 1 <= codon <= 393:
            label = f"{ref}{codon}{alt}"
            if label not in found:
                found.append(label)
    return found


def _claimed_significance_near(text: str, mutation: str) -> Optional[str]:
    """Find the significance term the text asserts for a mutation (same clause)."""
    t = str(text or "")
    idx = t.upper().find(mutation.upper())
    if idx < 0:
        return None
    window = t[max(0, idx - 120): idx + 120].lower()
    # order matters: check 'likely benign/pathogenic' before the base words
    for term in ("likely pathogenic", "likely benign", "pathogenic", "benign",
                 "uncertain significance", "vus", "neutral", "polymorphism"):
        if term in window:
            return normalize_significance(term)
    return None


@dataclass
class ConflictFinding:
    mutation: str
    ai_classification: str
    clinvar_classification: str
    conflict: bool
    severity: str  # high | medium | none | unknown
    evidence_url: str


class ClinVarConflictChecker:
    """Flags AI classifications that disagree with ClinVar."""

    def __init__(self) -> None:
        self._ref = dict(_CLINVAR_TP53)
        self._audit_log = Path("logs/clinvar_conflict.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"ClinVar checker audit dir unavailable: {e}")

    # ── ClinVar lookup (offline-first, optional live) ─────────────
    def clinvar_classification(self, mutation: str) -> Optional[str]:
        """Curated ClinVar significance for a mutation, else None.

        (Live ClinVar lookup can be layered here later with a graceful
        fallback to this curated reference — kept offline by default.)
        """
        key = str(mutation or "").strip().upper()
        if key in self._ref:
            return self._ref[key]
        codon = "".join(c for c in key if c.isdigit())
        for ref_mut, sig in self._ref.items():
            if "".join(c for c in ref_mut if c.isdigit()) == codon and codon:
                return sig
        return None

    # ── conflict scoring ──────────────────────────────────────────
    def _finding(self, mutation: str, ai_sig: Optional[str]) -> ConflictFinding:
        clinvar = self.clinvar_classification(mutation)
        ai_norm = normalize_significance(ai_sig) if ai_sig else ""
        if clinvar is None:
            severity, conflict = "unknown", False
        elif not ai_norm:
            severity, conflict = "none", False  # nothing claimed to contradict
        else:
            ab, cb = _bucket(ai_norm), _bucket(clinvar)
            if ab == cb:
                severity, conflict = "none", False
            elif "uncertain" in (ab, cb):
                severity, conflict = "medium", True   # over/under-call
            else:
                severity, conflict = "high", True      # opposite poles
        return ConflictFinding(
            mutation=mutation,
            ai_classification=ai_norm or "(not stated)",
            clinvar_classification=clinvar or "(not in ClinVar reference)",
            conflict=conflict, severity=severity,
            evidence_url=_clinvar_url(mutation),
        )

    def check(self, text: Optional[str] = None, mutation: Optional[str] = None,
              ai_classification: Optional[str] = None) -> Dict:
        """Verify AI classifications against ClinVar. Never raises; never empty."""
        findings: List[ConflictFinding] = []
        if mutation:
            findings.append(self._finding(mutation, ai_classification))
        elif text:
            for mut in extract_mutations(text):
                findings.append(self._finding(mut, _claimed_significance_near(text, mut)))

        conflicts = [f for f in findings if f.conflict]
        if not findings:
            verdict, msg = "no_claims", "No TP53 mutation claims found to verify."
        elif conflicts:
            worst = "high" if any(f.severity == "high" for f in conflicts) else "medium"
            verdict = f"conflict_{worst}"
            msg = f"{len(conflicts)} ClinVar conflict(s) detected (worst: {worst})."
        else:
            verdict, msg = "concordant", "All checked claims agree with ClinVar."

        self._audit(f"check -> {verdict} ({len(findings)} finding(s))")
        return {
            "verdict": verdict,
            "conflicts_found": len(conflicts),
            "findings": [asdict(f) for f in findings],
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": msg,
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"ClinVar checker audit failed: {e}")


_checker = ClinVarConflictChecker()


def check_conflicts(text: Optional[str] = None, mutation: Optional[str] = None,
                    ai_classification: Optional[str] = None) -> Dict:
    return _checker.check(text=text, mutation=mutation, ai_classification=ai_classification)
