"""
============================================================
Precision Onco Africa - PubMed Inline Citations
utils/pubmed_citations.py
============================================================
Fetches real PubMed references for a TP53 mutation / claim via the NCBI
Entrez E-utilities (esearch + esummary) and formats them as inline
[PMID: …] citations with links to the abstract.

Offline-first & honest:
  * Live: real PMIDs + titles from PubMed (cached, graceful timeout).
  * Fallback: if the API is unreachable, returns a single PubMed *search
    pointer* (a real query URL) — NO fabricated PMIDs.

Parsing is pure (testable without the network).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from utils.logger import log

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search_url(term: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={str(term).replace(' ', '+')}"


def pubmed_abstract_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def format_citation(rec: Dict) -> str:
    """Inline citation string, e.g. 'Smith et al., Nature 2020 [PMID: 123]'."""
    authors = rec.get("authors") or []
    first = authors[0] if authors else ""
    who = f"{first} et al." if len(authors) > 1 else (first or "Anon")
    src = rec.get("source", "")
    yr = rec.get("year", "")
    bits = " ".join(b for b in [src, str(yr)] if b)
    return f"{who}, {bits} [PMID: {rec.get('pmid','?')}]".strip()


def parse_esearch(payload: Optional[dict]) -> List[str]:
    """Extract PMIDs from an esearch JSON response. Defensive."""
    if not isinstance(payload, dict):
        return []
    ids = payload.get("esearchresult", {}).get("idlist", [])
    return [str(i) for i in ids if i]


def parse_esummary(payload: Optional[dict]) -> List[Dict]:
    """Parse esummary JSON into citation records. Defensive."""
    records: List[Dict] = []
    if not isinstance(payload, dict):
        return records
    result = payload.get("result", {})
    for uid in result.get("uids", []) or []:
        item = result.get(str(uid), {})
        if not isinstance(item, dict):
            continue
        authors = [a.get("name", "") for a in (item.get("authors") or [])
                   if isinstance(a, dict) and a.get("name")]
        pubdate = str(item.get("pubdate", "")).strip()
        year = pubdate.split(" ")[0] if pubdate else ""
        records.append({
            "pmid": str(uid),
            "title": (item.get("title") or "").strip().rstrip("."),
            "authors": authors,
            "source": item.get("source", ""),
            "year": year,
            "url": pubmed_abstract_url(uid),
        })
    return records


class PubMedClient:
    """Query PubMed for citations, with caching + a search-pointer fallback."""

    def __init__(self, cache_ttl: int = 1800, timeout: float = 8.0,
                 email: str = "") -> None:
        self._ttl = cache_ttl
        self._timeout = timeout
        self._email = email
        self._cache: Dict[str, tuple] = {}

    def _get_json(self, endpoint: str, params: Dict) -> Optional[dict]:
        try:
            import requests
            if self._email:
                params = {**params, "email": self._email}
            resp = requests.get(f"{EUTILS}/{endpoint}", params=params,
                                timeout=self._timeout,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"PubMed {endpoint} failed: {e}")
            return None

    def cite(self, query: str, max_results: int = 3, use_live: bool = True) -> Dict:
        """Return citations for a query term. Never empty (search-pointer fallback)."""
        term = str(query or "TP53").strip()
        search_term = term if "tp53" in term.lower() else f"TP53 {term}"
        cache_key = f"{search_term}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._ttl:
            return cached[1]

        records: List[Dict] = []
        live_ok = False
        if use_live:
            try:
                es = self._get_json("esearch.fcgi", {
                    "db": "pubmed", "term": search_term,
                    "retmax": max_results, "retmode": "json", "sort": "relevance"})
                pmids = parse_esearch(es)
                if pmids:
                    summ = self._get_json("esummary.fcgi", {
                        "db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
                    records = parse_esummary(summ)
                    live_ok = bool(records)
            except Exception as e:  # never crash — fall back to the search pointer
                log.warning(f"PubMed cite failed: {e}")
                records = []

        if not records:  # honest fallback: a real search link, no fake PMIDs
            records = [{
                "pmid": None,
                "title": f"Search PubMed for: {search_term}",
                "authors": [], "source": "PubMed search", "year": "",
                "url": pubmed_search_url(search_term),
            }]

        result = {
            "query": search_term,
            "citations": records,
            "inline": [format_citation(r) for r in records if r.get("pmid")],
            "live": live_ok,
            "count": len(records),
            "status": "success",
            "message": (f"{len(records)} PubMed citation(s) for '{search_term}'"
                        if live_ok else "Live PubMed unavailable — search pointer provided"),
        }
        self._cache[cache_key] = (time.time(), result)
        return result


_client = PubMedClient()


def pubmed_cite(query: str, max_results: int = 3, use_live: bool = True) -> Dict:
    return _client.cite(query, max_results=max_results, use_live=use_live)
