"""
============================================================
Precision Onco Africa - AlphaFold structure client (real structure)
utils/alphafold_client.py
============================================================
Fetches the real AlphaFold-predicted structure of human p53 (UniProt P04637)
and its per-residue confidence (pLDDT, stored in the PDB B-factor column).

Offline-first (same pattern as the other API clients):
  * Fetched server-side (no browser CORS issue) and cached (30 min TTL).
  * On any network failure it returns an 'unavailable' result with a clear note
    — the Structure tab then falls back to the existing experimental-PDB viewer.
  * Nothing is fabricated: if AlphaFold is unreachable, no structure is invented.

Parsing (pLDDT from PDB B-factors) is a pure function, unit-testable offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import log

# AlphaFold DB. The exact file version changes over time (v4 -> v6 -> ...), so
# we resolve the current pdbUrl from the API (version-proof) and only fall back
# to a direct file URL if the API is unreachable.
AF_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
AF_FILE_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
TP53_UNIPROT = "P04637"

# TP53 DNA-binding hotspot residues to report confidence for.
HOTSPOT_RESIDUES = (175, 220, 245, 248, 249, 273, 282)


def parse_plddt(pdb_text: Optional[str]) -> Dict:
    """Parse per-residue pLDDT (CA B-factor) from PDB text. Pure, never raises.

    Returns {per_residue: {resSeq: plddt}, mean, bands, n}. AlphaFold confidence
    bands: very high (>90), confident (70-90), low (50-70), very low (<=50).
    """
    empty = {"per_residue": {}, "mean": None, "bands": {}, "n": 0}
    if not pdb_text:
        return empty
    per_res: Dict[int, float] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":          # one pLDDT per residue (CA atom)
            continue
        try:
            resseq = int(line[22:26])
            b = float(line[60:66])
        except (ValueError, IndexError):
            continue
        per_res[resseq] = b
    if not per_res:
        return empty
    vals = list(per_res.values())
    mean = round(sum(vals) / len(vals), 1)
    bands = {
        "very_high": sum(1 for v in vals if v > 90),
        "confident": sum(1 for v in vals if 70 < v <= 90),
        "low": sum(1 for v in vals if 50 < v <= 70),
        "very_low": sum(1 for v in vals if v <= 50),
    }
    return {"per_residue": per_res, "mean": mean, "bands": bands, "n": len(vals)}


def plddt_band(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score > 90:
        return "very high"
    if score > 70:
        return "confident"
    if score > 50:
        return "low"
    return "very low"


@dataclass
class AlphaFoldStructure:
    """Real AlphaFold structure + confidence. Always populated (never raises)."""
    uniprot: str = TP53_UNIPROT
    model_url: str = ""
    pdb_text: str = ""
    mean_plddt: Optional[float] = None
    bands: Dict = field(default_factory=dict)
    n_residues: int = 0
    hotspot_plddt: Dict = field(default_factory=dict)
    per_residue: Dict = field(default_factory=dict)
    available: bool = False
    method: str = "unavailable"
    notes: str = ""

    def to_dict(self) -> Dict:
        # Compact view: drop the bulky PDB text + per-residue map.
        d = self.__dict__.copy()
        d["pdb_bytes"] = len(self.pdb_text)
        d.pop("pdb_text", None)
        d.pop("per_residue", None)
        return d


class AlphaFoldClient:
    """Fetch the AlphaFold structure for a UniProt accession, with caching."""

    def __init__(self, cache_ttl: int = 1800, timeout: float = 12.0) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._cache: Dict[str, tuple] = {}

    def _get_text(self, url: str) -> Optional[str]:
        try:
            import requests
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.warning(f"AlphaFold fetch failed ({url}): {e}")
            return None

    def _get_json(self, url: str):
        try:
            import requests
            resp = requests.get(url, timeout=self._timeout,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"AlphaFold API failed ({url}): {e}")
            return None

    def _resolve_pdb_url(self, uniprot: str) -> str:
        """Current pdbUrl from the API (version-proof); fall back to a file URL."""
        meta = self._get_json(AF_API_URL.format(acc=uniprot))
        if isinstance(meta, list) and meta and isinstance(meta[0], dict):
            url = meta[0].get("pdbUrl")
            if url:
                return url
        return AF_FILE_URL.format(acc=uniprot)

    def get_structure(self, uniprot: str = TP53_UNIPROT,
                      use_live: bool = True) -> AlphaFoldStructure:
        res = AlphaFoldStructure(uniprot=uniprot)
        res.model_url = AF_FILE_URL.format(acc=uniprot)

        if not use_live:
            res.notes = "Live fetch disabled — enable it to load the AlphaFold model."
            return res

        cached = self._cache.get(uniprot)
        if cached and (time.time() - cached[0]) < self._ttl:
            return cached[1]

        res.model_url = self._resolve_pdb_url(uniprot)
        pdb_text = self._get_text(res.model_url)
        if not pdb_text:
            res.notes = ("AlphaFold DB unreachable — showing the curated "
                         "experimental-structure viewer instead.")
            return res

        parsed = parse_plddt(pdb_text)
        res.pdb_text = pdb_text
        res.mean_plddt = parsed["mean"]
        res.bands = parsed["bands"]
        res.n_residues = parsed["n"]
        res.per_residue = parsed["per_residue"]
        res.hotspot_plddt = {
            r: parsed["per_residue"].get(r) for r in HOTSPOT_RESIDUES
        }
        res.available = True
        res.method = "alphafold_live"
        self._cache[uniprot] = (time.time(), res)
        return res


_client = AlphaFoldClient()


def get_tp53_structure(use_live: bool = True) -> AlphaFoldStructure:
    return _client.get_structure(TP53_UNIPROT, use_live=use_live)
