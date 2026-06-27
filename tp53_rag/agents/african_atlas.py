"""
============================================================
Precision Onco Africa - African TP53 Atlas (Agent #17)
agents/african_atlas.py
============================================================
Regional cancer-genomics atlas of TP53 in African populations.

Where african_drift.py corrects Western-database bias using allele
frequencies, this Atlas answers a different question: *which TP53
mutations dominate which African regions, in which cancers, driven by
which environmental exposures* — the epidemiological picture that is
under-represented in global databases.

DATA PROVENANCE: the figures below are CURATED / REPRESENTATIVE values
distilled from published literature (IARC TP53 Database, GLOBOCAN, and
peer-reviewed African cohort studies). They are for research and
educational context only — not patient-specific measurements. Every
profile carries its source list.
============================================================
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "african_tp53_atlas"

DISCLAIMER = (
    "Curated/representative epidemiology from published literature — "
    "research & educational context only, not patient-specific data."
)


# ── Curated regional epidemiology ─────────────────────────────────
# Each profile is keyed by a short id. Frequencies are REPRESENTATIVE
# (see DISCLAIMER) and each entry lists its sources.
AFRICAN_TP53_PROFILES: Dict[str, Dict] = {
    "aflatoxin_hcc": {
        "title": "Aflatoxin-associated hepatocellular carcinoma",
        "regions": ["West Africa", "East Africa", "Central Africa"],
        "countries": ["Nigeria", "Senegal", "Gambia", "Mozambique",
                      "Kenya", "Ghana", "Guinea"],
        "dominant_cancer": "Hepatocellular carcinoma (liver)",
        "key_mutations": ["R249S"],
        "environmental_driver": "Aflatoxin B1 (contaminated maize/groundnuts) "
                                "with frequent hepatitis B co-infection",
        "representative_prevalence": "R249S in up to ~40% of HCC in high-aflatoxin areas",
        "burden_score": 90,  # 0-100 relative regional TP53-cancer burden
        "kenya_context": "Aflatoxin outbreaks recurrent in Eastern/Central Kenya; "
                         "HCC presents late — prioritise HBV vaccination & grain storage.",
        "sources": ["IARC TP53 Database", "Gouas et al. 2009",
                    "Kew MC 2013 (aflatoxin & HCC)"],
    },
    "escc_corridor": {
        "title": "East African oesophageal squamous-cell carcinoma corridor",
        "regions": ["East Africa"],
        "countries": ["Kenya", "Tanzania", "Malawi", "Ethiopia", "Uganda"],
        "dominant_cancer": "Oesophageal squamous-cell carcinoma (ESCC)",
        "key_mutations": ["R175H", "R248W", "R273H", "G245S"],
        "environmental_driver": "Very hot beverages, tobacco/alcohol, "
                                "nutritional deficiencies, possible PAH exposure",
        "representative_prevalence": "TP53 mutated in ~50-70% of ESCC tumours",
        "burden_score": 85,
        "kenya_context": "ESCC is among the top cancers in the Kenyan Rift Valley; "
                         "late presentation — endoscopy access is the bottleneck.",
        "sources": ["GLOBOCAN", "Menya et al. (Kenya ESCC)",
                    "IARC ESCC African series"],
    },
    "west_africa_breast": {
        "title": "West African breast cancer (TP53-enriched, triple-negative)",
        "regions": ["West Africa"],
        "countries": ["Nigeria", "Ghana", "Senegal", "Mali"],
        "dominant_cancer": "Breast carcinoma (often triple-negative/basal)",
        "key_mutations": ["R175H", "R248Q", "R273H", "Y220C"],
        "environmental_driver": "Genetic ancestry + reproductive/parity factors; "
                                "aggressive early-onset subtypes",
        "representative_prevalence": "TP53 mutated in ~30-50% (higher in TNBC)",
        "burden_score": 75,
        "kenya_context": "Breast cancer often diagnosed young and late in East Africa; "
                         "TNBC over-represented vs Western cohorts.",
        "sources": ["Huo et al. (Nigeria/Ghana breast)", "GLOBOCAN",
                    "African Breast Cancer studies"],
    },
    "cervical_hpv": {
        "title": "HPV-driven cervical cancer (TP53 functionally inactivated)",
        "regions": ["East Africa", "Southern Africa", "West Africa"],
        "countries": ["Kenya", "Tanzania", "South Africa", "Malawi",
                      "Zambia", "Uganda"],
        "dominant_cancer": "Cervical squamous-cell carcinoma",
        "key_mutations": [],  # usually TP53 wild-type; E6-mediated degradation
        "environmental_driver": "High-risk HPV (E6 protein degrades p53) "
                                "with HIV co-infection amplifying risk",
        "representative_prevalence": "TP53 typically WILD-TYPE but functionally "
                                     "silenced by HPV E6 — mutation is NOT the driver",
        "burden_score": 80,
        "kenya_context": "Leading female cancer in Kenya; HPV vaccination + screening "
                         "are higher-impact than TP53 testing here.",
        "sources": ["GLOBOCAN", "WHO cervical cancer (Africa)",
                    "HPV E6/p53 literature"],
    },
}

# Representative country-level burden for the choropleth (0-100).
# Aggregated from the profiles above; illustrative, not measured rates.
AFRICAN_COUNTRY_BURDEN: Dict[str, int] = {
    "Nigeria": 88, "Kenya": 86, "Senegal": 78, "Gambia": 82,
    "Mozambique": 80, "Ghana": 74, "Tanzania": 76, "Malawi": 72,
    "Ethiopia": 70, "Uganda": 68, "South Africa": 75, "Zambia": 66,
    "Guinea": 64, "Mali": 60, "Zimbabwe": 58, "Egypt": 55,
}


@dataclass
class AtlasProfile:
    """Structured atlas response for one query."""
    query: str
    matched_profiles: List[Dict]
    regions: List[str]
    countries: List[str]
    cancers: List[str]
    key_mutations: List[str]
    environmental_drivers: List[str]
    sources: List[str]
    disclaimer: str = DISCLAIMER


class AfricanTP53Atlas:
    """Regional TP53 cancer-genomics atlas for African populations."""

    def __init__(self) -> None:
        self._profiles = AFRICAN_TP53_PROFILES
        self._audit_log = Path("logs/african_atlas.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover - filesystem dependent
            log.warning(f"African Atlas audit dir unavailable: {e}")

    # ── internal helpers ──────────────────────────────────────────
    @staticmethod
    def _codon(mutation: str) -> str:
        digits = "".join(c for c in str(mutation or "") if c.isdigit())
        return digits

    def _match(self, mutation: Optional[str], region: Optional[str],
               cancer: Optional[str]) -> List[str]:
        """Return ids of profiles matching any provided filter."""
        mut = str(mutation or "").strip().upper()
        mut_codon = self._codon(mut)
        reg = str(region or "").strip().lower()
        can = str(cancer or "").strip().lower()
        hits: List[str] = []
        for pid, p in self._profiles.items():
            ok = False
            if mut:
                for km in p["key_mutations"]:
                    if km.upper() == mut or (mut_codon and self._codon(km) == mut_codon):
                        ok = True
                        break
            if not ok and reg:
                ok = any(reg in r.lower() for r in p["regions"]) or \
                     any(reg in c.lower() for c in p["countries"])
            if not ok and can:
                ok = can in p["dominant_cancer"].lower()
            if ok:
                hits.append(pid)
        return hits

    # ── main entry ────────────────────────────────────────────────
    def profile(self, mutation: Optional[str] = None,
                region: Optional[str] = None,
                cancer_type: Optional[str] = None) -> Dict:
        """Return the African TP53 epidemiology for a mutation / region / cancer.

        Never returns empty: with no/unknown filters it returns a curated
        continental overview (all profiles).
        """
        ids = self._match(mutation, region, cancer_type)
        broadened = False
        if not ids:
            ids = list(self._profiles.keys())  # zero-result fallback: full atlas
            broadened = True

        matched = [{"id": pid, **self._profiles[pid]} for pid in ids]

        def _uniq(seq):
            out: List = []
            for x in seq:
                if x and x not in out:
                    out.append(x)
            return out

        result = AtlasProfile(
            query=" | ".join(filter(None, [mutation, region, cancer_type])) or "continental overview",
            matched_profiles=matched,
            regions=_uniq([r for p in matched for r in p["regions"]]),
            countries=_uniq([c for p in matched for c in p["countries"]]),
            cancers=_uniq([p["dominant_cancer"] for p in matched]),
            key_mutations=_uniq([m for p in matched for m in p["key_mutations"]]),
            environmental_drivers=_uniq([p["environmental_driver"] for p in matched]),
            sources=_uniq([s for p in matched for s in p["sources"]]),
        )
        self._audit(f"profiled:{result.query} -> {len(matched)} profile(s)"
                    f"{' [broadened]' if broadened else ''}")
        return {
            "atlas": asdict(result),
            "broadened": broadened,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": f"{len(matched)} African TP53 profile(s) for '{result.query}'",
        }

    def country_burden(self) -> Dict[str, int]:
        """Representative country-level TP53-cancer burden (for the map)."""
        return dict(AFRICAN_COUNTRY_BURDEN)

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"African Atlas audit failed: {e}")


# Module-level singleton + convenience function
_atlas = AfricanTP53Atlas()


def atlas_profile(mutation: Optional[str] = None, region: Optional[str] = None,
                  cancer_type: Optional[str] = None) -> Dict:
    return _atlas.profile(mutation=mutation, region=region, cancer_type=cancer_type)
