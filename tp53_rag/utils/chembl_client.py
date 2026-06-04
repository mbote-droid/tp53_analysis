"""
============================================================
TP53 RAG Platform - ChEMBL client (TP53-pathway drug discovery)
utils/chembl_client.py
============================================================
Provides real drug data for compounds targeting the TP53 pathway
(p53 reactivators, MDM2/MDMX inhibitors) from the ChEMBL database.

Offline-first design:
  * A curated set of real TP53-pathway drugs (name, mechanism, clinical
    phase) ships in-code and ALWAYS works — cloud, offline, or API down.
  * When online, the live ChEMBL REST API augments this with additional
    compounds; results are cached (30 min TTL). Any network failure falls
    back silently to the curated set (graceful degradation, never empty).

Parsing is a pure function so it is unit-testable without the network.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from utils.logger import log

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# ChEMBL target ids for the druggable TP53 pathway (avoids a search call).
TARGET_IDS: Dict[str, str] = {
    "MDM2": "CHEMBL5023",      # E3 ligase — main druggable node
    "TP53": "CHEMBL4096",      # Cellular tumor antigen p53
    "MDM4": "CHEMBL1293296",   # MDMX
}

# Human-readable clinical phase (ChEMBL max_phase scale).
PHASE_LABEL = {4: "Approved", 3: "Phase III", 2: "Phase II",
               1: "Phase I", 0: "Preclinical"}


def _chembl_search_url(name: str) -> str:
    return f"https://www.ebi.ac.uk/chembl/g/#search_results/all/query={name}"


# Curated real TP53-pathway drugs (always-available fallback).
CURATED_TP53_DRUGS: List[Dict] = [
    {"name": "Eprenetapopt (APR-246)", "mechanism": "Mutant p53 reactivator (thiol refolding)",
     "target": "TP53", "max_phase": 3, "source": "curated"},
    {"name": "Idasanutlin (RG7388)", "mechanism": "MDM2 inhibitor", "target": "MDM2",
     "max_phase": 3, "source": "curated"},
    {"name": "Navtemadlin (AMG-232)", "mechanism": "MDM2 inhibitor", "target": "MDM2",
     "max_phase": 2, "source": "curated"},
    {"name": "Milademetan", "mechanism": "MDM2 inhibitor", "target": "MDM2",
     "max_phase": 2, "source": "curated"},
    {"name": "Rezatapopt (PC14586)", "mechanism": "Y220C p53 pocket stabiliser",
     "target": "TP53", "max_phase": 2, "source": "curated"},
    {"name": "COTI-2", "mechanism": "Mutant p53 reactivator (metallochaperone)",
     "target": "TP53", "max_phase": 1, "source": "curated"},
    {"name": "ALRN-6924", "mechanism": "Dual MDM2/MDMX stapled-peptide inhibitor",
     "target": "MDM4", "max_phase": 1, "source": "curated"},
    {"name": "Nutlin-3a", "mechanism": "MDM2 inhibitor (tool compound)", "target": "MDM2",
     "max_phase": 0, "source": "curated"},
]


def parse_mechanisms(payload: Optional[dict], target: str) -> List[Dict]:
    """Parse a ChEMBL /mechanism response into normalised drug records.

    Pure & defensive — tolerates None / missing keys, never raises.
    """
    records: List[Dict] = []
    if not isinstance(payload, dict):
        return records
    for m in payload.get("mechanisms", []) or []:
        if not isinstance(m, dict):
            continue
        cid = m.get("molecule_chembl_id")
        name = (m.get("mechanism_of_action") or "").strip()
        if not cid:
            continue
        phase = m.get("max_phase")
        try:
            phase = int(phase) if phase is not None else None
        except (TypeError, ValueError):
            phase = None
        records.append({
            "name": cid,  # molecule id; display name resolved separately if needed
            "chembl_id": cid,
            "mechanism": name or "(mechanism not annotated)",
            "target": target,
            "max_phase": phase,
            "source": "chembl-live",
        })
    return records


class ChEMBLClient:
    """Fetch TP53-pathway drugs from ChEMBL, with caching + curated fallback."""

    def __init__(self, cache_ttl: int = 1800, timeout: float = 8.0) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._cache: Dict[str, tuple] = {}  # key -> (ts, data)

    def _get_json(self, url: str) -> Optional[dict]:
        """GET + parse JSON. Returns None on any failure (graceful)."""
        try:
            import requests
            resp = requests.get(url, timeout=self._timeout,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"ChEMBL fetch failed ({url.split('?')[0]}): {e}")
            return None

    def _cached(self, key: str) -> Optional[list]:
        hit = self._cache.get(key)
        if hit and (time.time() - hit[0]) < self._ttl:
            return hit[1]
        return None

    def fetch_target_drugs(self, target: str = "MDM2", limit: int = 15) -> List[Dict]:
        """Live ChEMBL mechanism query for one target. [] on failure."""
        tid = TARGET_IDS.get(target.upper())
        if not tid:
            return []
        cache_key = f"mech:{tid}:{limit}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached
        url = (f"{CHEMBL_BASE}/mechanism?target_chembl_id={tid}"
               f"&format=json&limit={int(limit)}")
        recs = parse_mechanisms(self._get_json(url), target.upper())
        self._cache[cache_key] = (time.time(), recs)
        return recs

    def compounds(self, targets: Optional[List[str]] = None,
                  use_live: bool = True, limit: int = 15) -> Dict:
        """Main entry: curated TP53-pathway drugs, augmented with live ChEMBL.

        Always returns a non-empty list. `live` indicates whether any live
        ChEMBL data was successfully merged in.
        """
        targets = targets or ["MDM2", "TP53"]
        merged: List[Dict] = list(CURATED_TP53_DRUGS)
        live_ok = False
        if use_live:
            seen = {d["name"].lower() for d in merged}
            for t in targets:
                for rec in self.fetch_target_drugs(t, limit=limit):
                    live_ok = True
                    if rec["name"].lower() not in seen:
                        merged.append(rec)
                        seen.add(rec["name"].lower())

        for d in merged:  # annotate human-readable phase
            d["phase_label"] = PHASE_LABEL.get(d.get("max_phase"), "Unknown")
            d.setdefault("chembl_url", _chembl_search_url(d["name"]))

        merged.sort(key=lambda d: (d.get("max_phase") or -1), reverse=True)
        return {
            "compounds": merged,
            "live": live_ok,
            "count": len(merged),
            "source": "chembl-live+curated" if live_ok else "curated",
        }


_client = ChEMBLClient()


def tp53_pathway_drugs(use_live: bool = True) -> Dict:
    return _client.compounds(use_live=use_live)
