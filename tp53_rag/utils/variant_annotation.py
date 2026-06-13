"""
============================================================
TP53 RAG Platform - Variant Annotation (real bioinformatics)
utils/variant_annotation.py
============================================================
Turns a TP53 variant (protein change like R175H, an rsID, or HGVS) into a
real, multi-source functional annotation:

  * Molecular consequence + SIFT / PolyPhen  — Ensembl VEP REST API
  * ClinVar significance, gnomAD allele freq, CADD, dbNSFP predictors
    — MyVariant.info (aggregates ClinVar/gnomAD/dbNSFP in one call)

Offline-first (same pattern as the ChEMBL/PubMed clients):
  * A curated set of well-established TP53 hotspot annotations ships in-code
    and ALWAYS works (cloud, offline, API down) — never fabricated, only
    long-established facts (consequence class, SIFT/PolyPhen direction,
    ClinVar significance, rsID, HGVS c.).
  * When online, the live APIs supply exact, current values (gnomAD AF, CADD,
    VEP consequence) which override the curated base. A `method` field reports
    whether the result is "live" or "curated_fallback".

Local VEP/SnpEff are intentionally NOT used: their multi-GB caches don't fit the
8 GB / offline-first target. The REST route gives the same real data.

Parsing is pure (no network) so it is fully unit-testable.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import log

ENSEMBL_VEP_BASE = "https://rest.ensembl.org/vep/human"
MYVARIANT_BASE = "https://myvariant.info/v1/variant"

# Canonical TP53 transcript used for HGVS-c annotation.
TP53_REFSEQ = "NM_000546.6"

# ── Curated, well-established hotspot annotations (always-available base) ──
# Values are long-standing facts (rsID, HGVS c., conformational/contact class,
# SIFT/PolyPhen direction, ClinVar significance). gnomAD is given qualitatively
# ("absent/ultra-rare") because these are SOMATIC hotspots, essentially absent
# from population databases — the live path fills in the exact figure.
CURATED_HOTSPOTS: Dict[str, Dict] = {
    "R175H": {"hgvs_c": "c.524G>A", "rsid": "rs28934578", "class": "conformational"},
    "G245S": {"hgvs_c": "c.733G>A", "rsid": "rs28934575", "class": "conformational"},
    "R248Q": {"hgvs_c": "c.743G>A", "rsid": "rs11540652", "class": "contact"},
    "R248W": {"hgvs_c": "c.742C>T", "rsid": "rs121912651", "class": "contact"},
    "R249S": {"hgvs_c": "c.746G>T", "rsid": "rs121912664", "class": "conformational"},
    "R273H": {"hgvs_c": "c.818G>A", "rsid": "rs28934576", "class": "contact"},
    "R273C": {"hgvs_c": "c.817C>T", "rsid": "rs121913343", "class": "contact"},
    "R282W": {"hgvs_c": "c.844C>T", "rsid": "rs28934574", "class": "conformational"},
    "Y220C": {"hgvs_c": "c.659A>G", "rsid": "rs121912666", "class": "conformational"},
}

# Established functional direction for confirmed pathogenic hotspots.
_HOTSPOT_BASE = {
    "consequence": "missense_variant",
    "impact": "MODERATE",
    "sift": "deleterious",
    "polyphen": "probably_damaging",
    "clinvar": "Pathogenic",
    "gnomad_af": "absent / ultra-rare (somatic hotspot)",
}

_PROTEIN_RE = re.compile(r'\b([A-Z])\s*(\d{1,3})\s*([A-Z*])\b', re.I)
_RSID_RE = re.compile(r'\b(rs\d+)\b', re.I)


@dataclass
class AnnotationResult:
    """Structured, source-tagged variant annotation. Never empty."""
    query: str
    protein_change: str = ""
    rsid: str = ""
    hgvs_c: str = ""
    gene: str = "TP53"
    consequence: str = "unknown"
    impact: str = "unknown"
    sift: str = "unknown"
    polyphen: str = "unknown"
    cadd_phred: Optional[float] = None
    clinvar_significance: str = "not_provided"
    gnomad_af: str = "unknown"
    structural_class: str = ""
    method: str = "curated_fallback"   # "live" | "curated_fallback"
    sources: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def normalise_protein_change(text: str) -> str:
    """Extract a canonical protein change key (e.g. 'R175H') from free text."""
    if not text:
        return ""
    m = _PROTEIN_RE.search(text.replace("p.", ""))
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2)}{m.group(3).upper()}"


def parse_vep(payload) -> Dict:
    """Parse an Ensembl VEP REST response. Pure, defensive, never raises."""
    out: Dict = {}
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return out
    out["consequence"] = payload.get("most_severe_consequence") or ""
    tcs = payload.get("transcript_consequences") or []
    # Prefer a TP53 transcript with SIFT/PolyPhen annotations.
    best = None
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        if (tc.get("gene_symbol") or "").upper() == "TP53":
            best = tc
            if "sift_prediction" in tc or "polyphen_prediction" in tc:
                break
    if best is None and tcs:
        best = tcs[0] if isinstance(tcs[0], dict) else None
    if isinstance(best, dict):
        out["impact"] = best.get("impact") or ""
        out["sift"] = best.get("sift_prediction") or ""
        out["polyphen"] = best.get("polyphen_prediction") or ""
        out["gene"] = best.get("gene_symbol") or "TP53"
    return {k: v for k, v in out.items() if v}


def parse_myvariant(payload) -> Dict:
    """Parse a MyVariant.info response (ClinVar/gnomAD/CADD/dbNSFP). Pure."""
    out: Dict = {}
    if not isinstance(payload, dict):
        return out

    # ClinVar significance (shape varies: rcv may be list or dict)
    clinvar = payload.get("clinvar")
    if isinstance(clinvar, dict):
        rcv = clinvar.get("rcv")
        sig = None
        if isinstance(rcv, list) and rcv:
            sig = (rcv[0] or {}).get("clinical_significance")
        elif isinstance(rcv, dict):
            sig = rcv.get("clinical_significance")
        if sig:
            out["clinvar_significance"] = sig

    # CADD phred
    cadd = payload.get("cadd")
    if isinstance(cadd, dict) and cadd.get("phred") is not None:
        try:
            out["cadd_phred"] = round(float(cadd["phred"]), 1)
        except (TypeError, ValueError):
            pass

    # gnomAD allele frequency (genome preferred, else exome)
    for key in ("gnomad_genome", "gnomad_exome"):
        g = payload.get(key)
        if isinstance(g, dict):
            af = g.get("af")
            af_val = af.get("af") if isinstance(af, dict) else af
            if af_val is not None:
                try:
                    out["gnomad_af"] = f"{float(af_val):.2e}"
                    break
                except (TypeError, ValueError):
                    pass

    # dbNSFP SIFT/PolyPhen (fallback if VEP lacked them)
    dbnsfp = payload.get("dbnsfp")
    if isinstance(dbnsfp, dict):
        sift = dbnsfp.get("sift")
        if isinstance(sift, dict) and sift.get("pred"):
            out.setdefault("sift", sift["pred"])
        pp = dbnsfp.get("polyphen2")
        if isinstance(pp, dict):
            hdiv = pp.get("hdiv")
            if isinstance(hdiv, dict) and hdiv.get("pred"):
                out.setdefault("polyphen", hdiv["pred"])
    return out


class VariantAnnotator:
    """Real multi-source TP53 variant annotation with offline-first fallback."""

    def __init__(self, cache_ttl: int = 1800, timeout: float = 8.0) -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._cache: Dict[str, tuple] = {}

    def _get_json(self, url: str) -> Optional[object]:
        try:
            import requests
            resp = requests.get(url, timeout=self._timeout,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"Variant annotation fetch failed ({url.split('?')[0]}): {e}")
            return None

    def _cached(self, key: str):
        hit = self._cache.get(key)
        if hit and (time.time() - hit[0]) < self._ttl:
            return hit[1]
        return None

    def _fetch_live(self, rsid: str) -> Dict:
        """Query Ensembl VEP + MyVariant by rsID. {} if nothing usable."""
        key = f"live:{rsid}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        merged: Dict = {}
        vep = parse_vep(self._get_json(f"{ENSEMBL_VEP_BASE}/id/{rsid}?content-type=application/json"))
        mv = parse_myvariant(self._get_json(f"{MYVARIANT_BASE}/{rsid}"))
        merged.update(vep)
        # MyVariant fills ClinVar/gnomAD/CADD; don't let it override VEP's
        # SIFT/PolyPhen (VEP is the primary source for those).
        for k, v in mv.items():
            if k in ("sift", "polyphen"):
                merged.setdefault(k, v)
            else:
                merged[k] = v
        sources = []
        if vep:
            sources.append("Ensembl VEP")
        if mv:
            sources.append("MyVariant.info (ClinVar/gnomAD/dbNSFP)")
        merged["_sources"] = sources
        self._cache[key] = (time.time(), merged)
        return merged

    def annotate(self, variant: str, use_live: bool = True) -> AnnotationResult:
        """Annotate a TP53 variant. Always returns a populated result."""
        variant = (variant or "").strip()
        res = AnnotationResult(query=variant)

        # Resolve identifiers
        rsid_match = _RSID_RE.search(variant)
        key = normalise_protein_change(variant)
        curated = CURATED_HOTSPOTS.get(key, {})
        res.protein_change = key
        res.rsid = (rsid_match.group(1).lower() if rsid_match else curated.get("rsid", ""))
        res.hgvs_c = (f"{TP53_REFSEQ}:{curated['hgvs_c']}" if curated.get("hgvs_c") else "")
        res.structural_class = curated.get("class", "")

        # Curated base for known pathogenic hotspots
        if curated:
            res.consequence = _HOTSPOT_BASE["consequence"]
            res.impact = _HOTSPOT_BASE["impact"]
            res.sift = _HOTSPOT_BASE["sift"]
            res.polyphen = _HOTSPOT_BASE["polyphen"]
            res.clinvar_significance = _HOTSPOT_BASE["clinvar"]
            res.gnomad_af = _HOTSPOT_BASE["gnomad_af"]
            res.sources.append("curated TP53 hotspot reference")
        else:
            res.notes = ("Not a curated hotspot — live annotation required for "
                         "full functional prediction.")

        # Live override
        if use_live and res.rsid:
            live = self._fetch_live(res.rsid)
            if live:
                for attr in ("consequence", "impact", "sift", "polyphen",
                             "gnomad_af", "clinvar_significance"):
                    if live.get(attr):
                        setattr(res, attr, live[attr])
                if live.get("cadd_phred") is not None:
                    res.cadd_phred = live["cadd_phred"]
                if live.get("gene"):
                    res.gene = live["gene"]
                res.sources.extend(live.get("_sources", []))
                res.method = "live"

        if not res.protein_change and not res.rsid:
            res.notes = ("Could not parse a TP53 protein change (e.g. R175H) or "
                         "rsID from the input.")
        return res


_annotator = VariantAnnotator()


def annotate_variant(variant: str, use_live: bool = True) -> Dict:
    """Convenience: annotate and return a plain dict (UI/JSON friendly)."""
    return _annotator.annotate(variant, use_live=use_live).to_dict()
