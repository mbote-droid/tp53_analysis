"""
============================================================
TP53 RAG Platform - ClinicalTrials.gov Matcher
agents/clinical_trials.py
============================================================
Finds active clinical trials relevant to a TP53 mutation + cancer type,
prioritising trials with Kenyan / African sites (then international
trials open to African patients).

Offline-first design (same pattern as the ChEMBL client):
  * Live: queries the ClinicalTrials.gov v2 REST API for RECRUITING
    studies, parses them, and ranks African sites first.
  * Fallback: if the API is unreachable, returns a curated set of REAL
    TP53-pathway trial programmes as ClinicalTrials.gov *search links*
    (no fabricated NCT IDs) — honest, verifiable, never empty.

Parsing is a pure function so it is unit-testable without the network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "clinical_trials_matcher"
CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
DISCLAIMER = ("Trial data from ClinicalTrials.gov (live) or curated search "
              "pointers — verify eligibility and recruiting status before acting.")

# African countries (for site prioritisation). Kenya is ranked highest.
AFRICAN_COUNTRIES = {
    "Kenya", "Nigeria", "South Africa", "Ghana", "Tanzania", "Uganda",
    "Ethiopia", "Egypt", "Morocco", "Senegal", "Rwanda", "Zambia",
    "Zimbabwe", "Malawi", "Mozambique", "Cameroon", "Botswana", "Namibia",
    "Mali", "Sudan", "Tunisia", "Algeria", "Ivory Coast", "Côte d'Ivoire",
}


def _ctgov_search_url(term: str, cond: str = "") -> str:
    q = term.replace(" ", "+")
    c = f"&cond={cond.replace(' ', '+')}" if cond else ""
    return f"https://clinicaltrials.gov/search?term={q}{c}"


# Curated REAL TP53-pathway trial programmes (offline fallback). These are
# search pointers to live ClinicalTrials.gov queries — no invented NCT IDs.
CURATED_TRIAL_PROGRAMS = [
    {"drug": "Eprenetapopt (APR-246)", "note": "Mutant p53 reactivator"},
    {"drug": "Rezatapopt (PC14586)", "note": "Y220C p53 stabiliser"},
    {"drug": "Idasanutlin", "note": "MDM2 inhibitor"},
    {"drug": "Navtemadlin (AMG-232)", "note": "MDM2 inhibitor"},
    {"drug": "Milademetan", "note": "MDM2 inhibitor"},
]


def _norm_phase(phases) -> str:
    """ClinicalTrials.gov phase list -> readable string."""
    if not phases:
        return "N/A"
    nums = []
    for p in phases:
        digits = "".join(c for c in str(p) if c.isdigit())
        if digits:
            nums.append(digits)
    return ("Phase " + "/".join(nums)) if nums else "N/A"


def _phase_rank(phase_str: str) -> int:
    digits = [int(c) for c in phase_str if c.isdigit()]
    return max(digits) if digits else 0


def parse_studies(payload: Optional[dict]) -> List[Dict]:
    """Parse a ClinicalTrials.gov v2 response into normalised trial records.

    Pure & defensive — tolerates None / missing keys, never raises.
    """
    records: List[Dict] = []
    if not isinstance(payload, dict):
        return records
    for study in payload.get("studies", []) or []:
        ps = (study or {}).get("protocolSection", {}) if isinstance(study, dict) else {}
        idm = ps.get("identificationModule", {}) or {}
        nct = idm.get("nctId")
        if not nct:
            continue
        locs = ps.get("contactsLocationsModule", {}).get("locations", []) or []
        countries = []
        for l in locs:
            c = (l or {}).get("country")
            if c and c not in countries:
                countries.append(c)
        african = [c for c in countries if c in AFRICAN_COUNTRIES]
        phase = _norm_phase(ps.get("designModule", {}).get("phases"))
        records.append({
            "nct_id": nct,
            "title": (idm.get("briefTitle") or "(untitled)").strip(),
            "status": ps.get("statusModule", {}).get("overallStatus", "UNKNOWN"),
            "phase": phase,
            "phase_rank": _phase_rank(phase),
            "conditions": ps.get("conditionsModule", {}).get("conditions", []) or [],
            "countries": countries,
            "african_sites": african,
            "african_priority": bool(african),
            "kenya_site": "Kenya" in african,
            "url": f"https://clinicaltrials.gov/study/{nct}",
            "source": "ctgov-live",
        })
    return records


@dataclass
class TrialSearch:
    query: str
    trials: List[Dict]
    live: bool
    african_count: int
    kenya_count: int
    disclaimer: str = DISCLAIMER


class ClinicalTrialsMatcher:
    """Match TP53 mutation + cancer to recruiting clinical trials."""

    def __init__(self, cache_ttl: int = 1800, timeout: float = 10.0) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._cache: Dict[str, tuple] = {}
        self._audit_log = Path("logs/clinical_trials.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Trials audit dir unavailable: {e}")

    def _get_json(self, params: Dict) -> Optional[dict]:
        try:
            import requests
            resp = requests.get(CTGOV_API, params=params, timeout=self._timeout,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"ClinicalTrials.gov fetch failed: {e}")
            return None

    def _curated_fallback(self, mutation: str, cancer: str) -> List[Dict]:
        out = []
        for prog in CURATED_TRIAL_PROGRAMS:
            out.append({
                "nct_id": None,
                "title": f"Search: {prog['drug']} — {prog['note']}",
                "status": "SEARCH",
                "phase": "N/A", "phase_rank": 0,
                "conditions": [cancer] if cancer else [],
                "countries": [], "african_sites": [],
                "african_priority": False, "kenya_site": False,
                "url": _ctgov_search_url(prog["drug"], cancer),
                "source": "curated-search",
            })
        return out

    @staticmethod
    def _rank_key(t: Dict):
        # Kenya first, then other African, then phase, then recruiting.
        return (
            2 if t.get("kenya_site") else 1 if t.get("african_priority") else 0,
            t.get("phase_rank", 0),
            1 if t.get("status") == "RECRUITING" else 0,
        )

    def search(self, mutation: Optional[str] = None, cancer_type: Optional[str] = None,
               status: str = "RECRUITING", max_results: int = 20,
               use_live: bool = True, phases_234_only: bool = True) -> Dict:
        """Find recruiting trials for a mutation + cancer. Never empty."""
        mutation = (mutation or "").strip()
        cancer = (cancer_type or "").strip()
        term = " ".join(filter(None, ["TP53", mutation])) or "TP53"
        query = " | ".join(filter(None, [mutation, cancer])) or "TP53"

        trials: List[Dict] = []
        live_ok = False
        if use_live:
            params = {"query.term": term, "pageSize": max_results, "format": "json"}
            if cancer:
                params["query.cond"] = cancer
            if status:
                params["filter.overallStatus"] = status
            cache_key = str(sorted(params.items()))
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[0]) < self._ttl:
                trials = cached[1]
                live_ok = True
            else:
                payload = self._get_json(params)
                if payload is not None:
                    trials = parse_studies(payload)
                    self._cache[cache_key] = (time.time(), trials)
                    live_ok = True

        if phases_234_only and trials:
            filtered = [t for t in trials if t.get("phase_rank", 0) >= 2]
            if filtered:  # don't go empty if filtering removes everything
                trials = filtered

        if not trials:  # graceful fallback — never empty
            trials = self._curated_fallback(mutation, cancer)

        trials.sort(key=self._rank_key, reverse=True)
        trials = trials[:max_results]

        result = TrialSearch(
            query=query, trials=trials, live=live_ok,
            african_count=sum(1 for t in trials if t.get("african_priority")),
            kenya_count=sum(1 for t in trials if t.get("kenya_site")),
        )
        self._audit(f"search:{query} -> {len(trials)} trial(s) "
                    f"(live={live_ok}, african={result.african_count})")
        return {
            **asdict(result),
            "count": len(trials),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": f"{len(trials)} trial(s) for '{query}' "
                       f"({result.african_count} African, {result.kenya_count} Kenyan)",
        }

    def _audit(self, msg: str) -> None:
        try:
            import json as _json
            entry = _json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Trials audit failed: {e}")


_matcher = ClinicalTrialsMatcher()


def match_trials(mutation: Optional[str] = None, cancer_type: Optional[str] = None,
                 use_live: bool = True) -> Dict:
    return _matcher.search(mutation=mutation, cancer_type=cancer_type, use_live=use_live)
