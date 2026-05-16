"""
agents/gene_expression.py — Gene Expression & Cellular Behaviour Agent (Agent #12)
=====================================================================================
Predicts gene expression changes and downstream cellular behaviour resulting from
TP53 mutations. Integrates pathway analysis, transcription factor disruption,
tumour microenvironment (TME) modelling, and Kenya-context clinical translation.

HIPAA compliant — no raw PII in logs or LLM prompts.
Reusable cache + rate limiter (utils/cache.py).
Rate-limited — 20 calls/min.
HL7 FHIR R4 structured output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from utils.llm_cache import get_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    log.warning("utils.cache not found — caching disabled")

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# Clinical Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_ID = "gene_expression"

# ── Downstream gene expression changes per TP53 mutation ─────────────────────
# Sources: TCGA, IARC TP53 DB, MSigDB hallmark gene sets
MUTATION_EXPRESSION_MAP: dict[str, dict] = {
    "R175H": {
        "upregulated": [
            "MDM2", "VEGFA", "MYC", "CCND1", "BCL2", "CDH2",
            "SNAI1", "TWIST1", "VIM",           # EMT drivers
            "PD-L1", "IL-6", "CXCL8",           # TME immunosuppression
        ],
        "downregulated": [
            "CDKN1A", "BAX", "PUMA", "NOXA",    # Apoptosis loss
            "CDH1", "PTEN",                      # Tumour suppressors
            "GADD45A", "RRM2B",                  # DNA-damage response
        ],
        "pathways_activated": [
            "PI3K/AKT/mTOR", "RAS/MAPK/ERK", "NF-κB",
            "Wnt/β-catenin", "EMT", "Angiogenesis",
        ],
        "pathways_suppressed": [
            "Intrinsic apoptosis", "G1/S checkpoint",
            "Nucleotide excision repair",
        ],
        "cellular_behaviours": [
            "Enhanced proliferation (CDK4/6 axis active)",
            "Apoptosis resistance (BCL2 overexpression)",
            "Epithelial-mesenchymal transition (EMT)",
            "Angiogenesis promotion via VEGFA",
            "Immune evasion via PD-L1 upregulation",
            "Metastatic invasion via MMP upregulation",
        ],
        "tme_profile": {
            "immune_infiltration": "Low",
            "M2_macrophages":      "High",
            "T_cell_exclusion":    True,
            "cytokine_milieu":     "Immunosuppressive (IL-6, IL-10, TGF-β high)",
        },
        "drug_sensitivity": {
            "APR-246":    "High — reactivates p53 folding",
            "Doxorubicin":"Moderate — partially bypasses p53",
            "Olaparib":   "Low (unless BRCA co-mutation)",
            "Cisplatin":  "Moderate",
        },
        "transcription_factors_disrupted": [
            "p53 (WT function lost)", "p63 (cross-reactive inhibition)",
            "p73 (dominant-negative inhibition)",
        ],
    },
    "R248W": {
        "upregulated": [
            "MDM2", "RAD51", "MYC", "CCNE1", "MMP9",
            "PD-L1", "IL-8", "CXCL5",
            "PCNA", "TOP2A",                     # Replication stress markers
        ],
        "downregulated": [
            "CDKN1A", "BAX", "FAS", "APAF1",
            "MLH1", "MSH2",                      # MMR loss
            "BRCA1",
        ],
        "pathways_activated": [
            "Homologous recombination (aberrant)", "RAS/MAPK",
            "DNA replication stress", "NF-κB", "EMT",
        ],
        "pathways_suppressed": [
            "Mismatch repair", "Intrinsic apoptosis",
            "G2/M checkpoint",
        ],
        "cellular_behaviours": [
            "Aggressive replication with genomic instability",
            "Mismatch repair deficiency — hypermutation risk",
            "High invasiveness via MMP9",
            "Cisplatin resistance via RAD51 overexpression",
            "Immune desert phenotype",
        ],
        "tme_profile": {
            "immune_infiltration": "Very Low",
            "M2_macrophages":      "High",
            "T_cell_exclusion":    True,
            "cytokine_milieu":     "Cold tumour — poor immunotherapy response",
        },
        "drug_sensitivity": {
            "APR-246":    "High",
            "Cisplatin":  "Low (RAD51-mediated resistance)",
            "Doxorubicin":"Moderate",
            "Pembrolizumab":"Low (cold TME)",
        },
        "transcription_factors_disrupted": [
            "p53", "p73 (dominant-negative)", "NRF2 (indirect suppression)",
        ],
    },
    "R273H": {
        "upregulated": [
            "MDM2", "MYC", "CCND1", "MCM2", "CDK2",
            "VEGFC", "PDGFRA",                   # Alternative angiogenesis
            "IL-6", "TNF-α",
        ],
        "downregulated": [
            "CDKN1A", "PTEN", "RB1",
            "DAPK1",                             # Autophagy suppressed
            "RASSF1A",
        ],
        "pathways_activated": [
            "PI3K/AKT", "CDK4/6 cell cycle", "PDGF signalling",
            "Autophagy suppression", "NF-κB",
        ],
        "pathways_suppressed": [
            "RB1 tumour suppression", "Autophagy",
            "S-phase checkpoint",
        ],
        "cellular_behaviours": [
            "Uncontrolled S-phase entry",
            "PTEN loss → PI3K hyperactivation",
            "Autophagy suppression → therapy resistance",
            "PDGF-driven stromal remodelling",
        ],
        "tme_profile": {
            "immune_infiltration": "Moderate",
            "M2_macrophages":      "Moderate",
            "T_cell_exclusion":    False,
            "cytokine_milieu":     "Mixed — some inflammatory signalling",
        },
        "drug_sensitivity": {
            "Cisplatin":   "Low–Moderate",
            "Paclitaxel":  "Moderate",
            "Everolimus":  "Moderate (mTOR inhibitor — PI3K active)",
            "APR-246":     "Low",
        },
        "transcription_factors_disrupted": [
            "p53", "E2F1 (RB1 loss)", "FOXO3a (AKT phosphorylation)",
        ],
    },
    "G245S": {
        "upregulated": [
            "MDM2", "CCNB1", "CDC20", "PLK1",   # Mitotic drivers
            "AURKA", "AURKB",                    # Chromosomal instability
            "HMGA2",
        ],
        "downregulated": [
            "CDKN1A", "GADD45A", "14-3-3σ",     # G2/M checkpoint loss
            "BAX", "PUMA",
        ],
        "pathways_activated": [
            "Mitotic spindle checkpoint bypass",
            "Chromosomal instability (CIN)",
            "APC/C–CDC20 axis", "PLK1/Aurora kinase",
        ],
        "pathways_suppressed": [
            "G2/M checkpoint", "Spindle assembly checkpoint",
            "Intrinsic apoptosis",
        ],
        "cellular_behaviours": [
            "Chromosomal instability — aneuploidy",
            "Mitotic bypass — rapid tumour evolution",
            "PLK1-driven therapy resistance",
            "Reduced sensitivity to DNA-damaging agents",
        ],
        "tme_profile": {
            "immune_infiltration": "Low",
            "M2_macrophages":      "Moderate",
            "T_cell_exclusion":    True,
            "cytokine_milieu":     "Mildly immunosuppressive",
        },
        "drug_sensitivity": {
            "Paclitaxel":  "Low (mitotic bypass)",
            "Doxorubicin": "Moderate",
            "PLK1 inhibitor (BI-6727)": "Potential — PLK1 overexpressed",
            "Aurora kinase inhibitor":  "Potential",
        },
        "transcription_factors_disrupted": [
            "p53", "FOXM1 (indirectly activated)", "NF-Y (G2/M genes)",
        ],
    },
    "R249S": {
        "upregulated": [
            "MDM2", "IGF1R", "HIF-1α",          # Hypoxia adaptation
            "AFP",                               # Hepatocellular marker
            "GPC3", "DKK1",
        ],
        "downregulated": [
            "CDKN1A", "BAX",
            "SOCS1", "SOCS3",                   # JAK/STAT dysregulation
        ],
        "pathways_activated": [
            "HIF-1α/hypoxia", "IGF1R/AKT", "JAK/STAT3",
            "Wnt/β-catenin (hepatic context)",
        ],
        "pathways_suppressed": [
            "Apoptosis", "JAK/STAT negative feedback",
        ],
        "cellular_behaviours": [
            "Hypoxia adaptation — HIF-1α stabilisation",
            "IGF1R-driven survival signalling",
            "Aflatoxin/hepatocellular context — AFP elevation",
            "STAT3 constitutive activation",
        ],
        "tme_profile": {
            "immune_infiltration": "Low",
            "M2_macrophages":      "High",
            "T_cell_exclusion":    True,
            "cytokine_milieu":     "HIF-1α driven — hypoxic immunosuppression",
        },
        "drug_sensitivity": {
            "Sorafenib":   "Moderate (HCC context)",
            "Lenvatinib":  "Moderate",
            "Doxorubicin": "Low (hypoxia resistance)",
            "APR-246":     "Low",
        },
        "transcription_factors_disrupted": [
            "p53", "HIF-1α (constitutively active)", "STAT3",
        ],
    },
    "R282W": {
        "upregulated": [
            "MDM2", "MYC", "CCND1", "E2F1",
            "RAD54L",                            # HR pathway
            "DNMT3A", "EZH2",                   # Epigenetic silencing
        ],
        "downregulated": [
            "CDKN1A", "BAX", "PUMA",
            "RASSF1A", "APC",                   # Wnt suppression lost
        ],
        "pathways_activated": [
            "Epigenetic silencing (EZH2/DNMT3A)",
            "E2F transcription", "Wnt/β-catenin",
            "Homologous recombination (aberrant)",
        ],
        "pathways_suppressed": [
            "Apoptosis", "Senescence",
            "APC/Wnt negative regulation",
        ],
        "cellular_behaviours": [
            "Epigenetic reprogramming via EZH2/DNMT3A",
            "Stem-like tumour cell phenotype",
            "E2F1-driven aggressive replication",
            "Chemotherapy resistance via HR upregulation",
        ],
        "tme_profile": {
            "immune_infiltration": "Low",
            "M2_macrophages":      "High",
            "T_cell_exclusion":    True,
            "cytokine_milieu":     "Immunosuppressive + epigenetically silenced",
        },
        "drug_sensitivity": {
            "EZH2 inhibitor (tazemetostat)": "Potential",
            "Decitabine (DNMT inhibitor)":   "Potential",
            "Doxorubicin": "Moderate",
            "APR-246":     "Low",
        },
        "transcription_factors_disrupted": [
            "p53", "E2F1 (activated by RB1 bypass)",
            "PRC2 complex (EZH2-driven silencing)",
        ],
    },
}

# Co-mutation synthetic lethality
SYNTHETIC_LETHALITY_MAP: dict[str, list[dict]] = {
    "BRCA1": [{"drug": "Olaparib",   "rationale": "PARP inhibitor — HR deficiency"}],
    "BRCA2": [{"drug": "Olaparib",   "rationale": "PARP inhibitor — HR deficiency"}],
    "ATM":   [{"drug": "Olaparib",   "rationale": "ATM-null + PARP synergy"}],
    "PTEN":  [{"drug": "Everolimus", "rationale": "PI3K/mTOR inhibitor — PTEN loss"}],
    "RB1":   [{"drug": "Palbociclib","rationale": "CDK4/6 inhibitor — RB1 null bypass"}],
    "MLH1":  [{"drug": "Pembrolizumab","rationale":"MSI-H → immunotherapy eligible"}],
    "MSH2":  [{"drug": "Pembrolizumab","rationale":"MSI-H → immunotherapy eligible"}],
}

RATE_LIMIT_CALLS  = 20
RATE_LIMIT_WINDOW = 60

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpressionProfile:
    mutation:                  str
    upregulated:               list[str]
    downregulated:             list[str]
    pathways_activated:        list[str]
    pathways_suppressed:       list[str]
    cellular_behaviours:       list[str]
    tme_profile:               dict
    drug_sensitivity:          dict
    transcription_factors:     list[str]
    synthetic_lethality_hits:  list[dict]  = field(default_factory=list)

@dataclass
class GeneExpressionResult:
    agent_id:        str = AGENT_ID
    timestamp:       str = ""
    mutation:        str = ""
    patient_hash:    str = ""
    profile:         Optional[ExpressionProfile] = None
    query_response:  str = ""
    fhir_report:     dict = field(default_factory=dict)
    cache_hit:       bool = False
    error:           Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable utilities (same pattern as liquid_biopsy)
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_calls: int = RATE_LIMIT_CALLS,
                 window: int = RATE_LIMIT_WINDOW) -> None:
        self._lock  = threading.Lock()
        self._calls: list[float] = []
        self.max_calls = max_calls
        self.window    = window

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < self.window]
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


_SAFE_QUERY  = re.compile(r"^[\w\s\?\.\,\-\'\"\/\(\)]+$")
_INJECTION   = re.compile(
    r"(drop\s+table|delete\s+from|<script|import\s+os|__import__|eval\()",
    re.IGNORECASE,
)
_HGVS_LOOSE  = re.compile(r"^[A-Z]\d+[A-Z\*]$")   # e.g. R175H, G245S

def sanitise_query(query: str) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    q = query.strip()
    if len(q) > 2000:
        raise ValueError("Query exceeds 2000 character limit")
    if _INJECTION.search(q):
        raise ValueError("Potentially unsafe content detected in query")
    if not _SAFE_QUERY.match(q):
        raise ValueError("Query contains disallowed characters")
    return q

def sanitise_mutation(mutation: str) -> str:
    m = mutation.strip().upper()
    if not _HGVS_LOOSE.match(m):
        raise ValueError(
            f"Invalid mutation format '{m}'. Expected HGVS short (e.g. R175H)."
        )
    return m

def hash_patient_id(patient_id: str) -> str:
    return "PAT-" + hashlib.sha256(patient_id.encode()).hexdigest()[:12].upper()


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis engine
# ═══════════════════════════════════════════════════════════════════════════════

def get_expression_profile(
    mutation: str,
    co_mutations: Optional[list[str]] = None,
) -> ExpressionProfile:
    """
    Return full expression + cellular behaviour profile for a TP53 mutation.
    co_mutations: list of co-occurring gene alterations (e.g. ["BRCA1", "PTEN"])
    """
    data = MUTATION_EXPRESSION_MAP.get(mutation)
    if data is None:
        raise ValueError(
            f"Mutation '{mutation}' not in knowledge base. "
            f"Supported: {list(MUTATION_EXPRESSION_MAP.keys())}"
        )

    # Synthetic lethality from co-mutations
    sl_hits: list[dict] = []
    for gene in (co_mutations or []):
        gene_upper = gene.strip().upper()
        if gene_upper in SYNTHETIC_LETHALITY_MAP:
            for entry in SYNTHETIC_LETHALITY_MAP[gene_upper]:
                sl_hits.append({
                    "co_mutation": gene_upper,
                    "drug":        entry["drug"],
                    "rationale":   entry["rationale"],
                })

    return ExpressionProfile(
        mutation                 = mutation,
        upregulated              = data["upregulated"],
        downregulated            = data["downregulated"],
        pathways_activated       = data["pathways_activated"],
        pathways_suppressed      = data["pathways_suppressed"],
        cellular_behaviours      = data["cellular_behaviours"],
        tme_profile              = data["tme_profile"],
        drug_sensitivity         = data["drug_sensitivity"],
        transcription_factors    = data["transcription_factors_disrupted"],
        synthetic_lethality_hits = sl_hits,
    )


def build_fhir_observation(
    patient_hash: str,
    profile: ExpressionProfile,
) -> dict:
    """HL7 FHIR R4 Observation — gene expression analysis result."""
    return {
        "resourceType": "Observation",
        "status":       "final",
        "category": [{
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/observation-category",
                "code":    "laboratory",
                "display": "Laboratory",
            }]
        }],
        "code": {
            "coding": [{
                "system":  "http://loinc.org",
                "code":    "69548-6",
                "display": "Genetic variant assessment",
            }],
            "text": f"TP53 {profile.mutation} Expression Profile",
        },
        "subject": {"reference": f"Patient/{patient_hash}"},
        "component": [
            {
                "code":         {"text": "Upregulated genes"},
                "valueString":  ", ".join(profile.upregulated),
            },
            {
                "code":         {"text": "Downregulated genes"},
                "valueString":  ", ".join(profile.downregulated),
            },
            {
                "code":         {"text": "Activated pathways"},
                "valueString":  ", ".join(profile.pathways_activated),
            },
            {
                "code":         {"text": "Suppressed pathways"},
                "valueString":  ", ".join(profile.pathways_suppressed),
            },
            {
                "code":         {"text": "Cellular behaviours"},
                "valueString":  " | ".join(profile.cellular_behaviours),
            },
            {
                "code":         {"text": "TME immune infiltration"},
                "valueString":  profile.tme_profile.get("immune_infiltration", "Unknown"),
            },
            {
                "code":         {"text": "TME cytokine milieu"},
                "valueString":  profile.tme_profile.get("cytokine_milieu", "Unknown"),
            },
            {
                "code":         {"text": "Drug sensitivity"},
                "valueString":  json.dumps(profile.drug_sensitivity),
            },
            {
                "code":         {"text": "Synthetic lethality targets"},
                "valueString":  json.dumps(profile.synthetic_lethality_hits) or "None",
            },
            {
                "code":         {"text": "Disrupted transcription factors"},
                "valueString":  ", ".join(profile.transcription_factors),
            },
        ],
    }


def _build_context(profile: ExpressionProfile, query: str) -> str:
    """Build LLM-safe context string — no PII."""
    sl_text = (
        "\n".join(
            f"  {h['co_mutation']} → {h['drug']} ({h['rationale']})"
            for h in profile.synthetic_lethality_hits
        )
        or "  None detected"
    )
    sens = "\n".join(
        f"  {drug}: {level}"
        for drug, level in profile.drug_sensitivity.items()
    )
    return f"""
Gene Expression & Cellular Behaviour Analysis
TP53 Mutation: {profile.mutation}
Query: {query}

UPREGULATED GENES:
  {', '.join(profile.upregulated)}

DOWNREGULATED GENES:
  {', '.join(profile.downregulated)}

ACTIVATED PATHWAYS:
  {' | '.join(profile.pathways_activated)}

SUPPRESSED PATHWAYS:
  {' | '.join(profile.pathways_suppressed)}

PREDICTED CELLULAR BEHAVIOURS:
  {chr(10).join('  • ' + b for b in profile.cellular_behaviours)}

TUMOUR MICROENVIRONMENT:
  Immune infiltration : {profile.tme_profile.get('immune_infiltration')}
  M2 macrophages      : {profile.tme_profile.get('M2_macrophages')}
  T-cell exclusion    : {profile.tme_profile.get('T_cell_exclusion')}
  Cytokine milieu     : {profile.tme_profile.get('cytokine_milieu')}

DRUG SENSITIVITY:
{sens}

SYNTHETIC LETHALITY OPPORTUNITIES:
{sl_text}

DISRUPTED TRANSCRIPTION FACTORS:
  {', '.join(profile.transcription_factors)}
""".strip()


def _llm_query(query: str, context: str) -> str:
    try:
        from rag_chain import TP53RAGChain
        chain  = TP53RAGChain()
        result = chain.query(f"{query}\n\nContext:\n{context}")
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)
    except Exception as exc:
        log.warning("LLM unavailable (%s) — using structured fallback", exc)
        return f"[Structured Gene Expression Analysis]\n\n{context}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Agent Class
# ═══════════════════════════════════════════════════════════════════════════════

class GeneExpressionAgent:
    """
    Gene Expression & Cellular Behaviour Prediction Agent.

    Usage
    -----
    agent = GeneExpressionAgent()

    # Query by mutation
    result = agent.analyse(
        mutation    = "R175H",
        query       = "What cellular behaviours does R175H drive?",
        patient_id  = "SAM-NBI-01",        # optional — hashed internally
        co_mutations= ["BRCA1", "PTEN"],   # optional
    )

    print(result.query_response)
    print(result.profile.cellular_behaviours)
    """

    def __init__(self, ttl: int = 1800) -> None:
        self._rate  = RateLimiter()
        self._cache = get_cache(ttl=ttl) if _CACHE_AVAILABLE else None

    def analyse(
        self,
        mutation:     str,
        query:        str = "Explain the gene expression changes and cellular behaviour for this mutation",
        patient_id:   Optional[str] = None,
        co_mutations: Optional[list[str]] = None,
    ) -> GeneExpressionResult:
        """Main entry point."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 1. Rate limit
        if not self._rate.allow():
            return GeneExpressionResult(
                timestamp = ts,
                error     = "Rate limit exceeded — max 20 calls/min",
            )

        # 2. Sanitise inputs
        try:
            mutation = sanitise_mutation(mutation)
            query    = sanitise_query(query)
        except ValueError as exc:
            return GeneExpressionResult(timestamp=ts, error=str(exc))

        patient_hash = hash_patient_id(patient_id) if patient_id else "UNKNOWN"
        cache_key    = f"{mutation}::{query}::{','.join(sorted(co_mutations or []))}"

        # 3. Cache lookup
        if self._cache:
            cached = self._cache.get(AGENT_ID, cache_key)
            if cached:
                result              = GeneExpressionResult(**cached)
                result.cache_hit    = True
                return result

        # 4. Build expression profile
        try:
            profile = get_expression_profile(mutation, co_mutations)
        except ValueError as exc:
            return GeneExpressionResult(timestamp=ts, mutation=mutation, error=str(exc))

        # 5. Build LLM context + query
        context  = _build_context(profile, query)
        response = _llm_query(query, context)

        # 6. FHIR
        fhir = build_fhir_observation(patient_hash, profile)

        # 7. Assemble
        result = GeneExpressionResult(
            timestamp      = ts,
            mutation       = mutation,
            patient_hash   = patient_hash,
            profile        = profile,
            query_response = response,
            fhir_report    = fhir,
            cache_hit      = False,
        )

        # 8. Cache
        if self._cache:
            try:
                payload = asdict(result)
                self._cache.set(AGENT_ID, cache_key, payload)
            except Exception as exc:
                log.warning("Cache write failed: %s", exc)

        return result

    def compare_mutations(self, mutations: list[str]) -> dict:
        """
        Side-by-side comparison of expression profiles for multiple mutations.
        Returns dict keyed by mutation with pathway + behaviour highlights.
        """
        out = {}
        for m in mutations:
            try:
                m = sanitise_mutation(m)
                p = get_expression_profile(m)
                out[m] = {
                    "top_upregulated":   p.upregulated[:5],
                    "top_downregulated": p.downregulated[:5],
                    "key_pathways":      p.pathways_activated[:3],
                    "tme_infiltration":  p.tme_profile.get("immune_infiltration"),
                    "behaviours":        p.cellular_behaviours[:3],
                    "best_drug":         next(iter(p.drug_sensitivity), "Unknown"),
                }
            except ValueError as exc:
                out[m] = {"error": str(exc)}
        return out

    def synthetic_lethality_screen(
        self,
        mutation: str,
        co_mutations: list[str],
    ) -> list[dict]:
        """Standalone synthetic lethality query."""
        try:
            mutation = sanitise_mutation(mutation)
            profile  = get_expression_profile(mutation, co_mutations)
            return profile.synthetic_lethality_hits
        except ValueError as exc:
            return [{"error": str(exc)}]


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test / reverse-engineering suite
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== GeneExpressionAgent Self-Test ===\n")

    agent = GeneExpressionAgent()

    # Test 1: R175H profile
    result = agent.analyse("R175H", "What does R175H upregulate?", patient_id="SAM-001")
    assert result.error is None,                          "FAIL: R175H analysis errored"
    assert "MDM2" in result.profile.upregulated,          "FAIL: MDM2 not in upregulated"
    assert "CDKN1A" in result.profile.downregulated,      "FAIL: CDKN1A not downregulated"
    print("✅ Test 1 — R175H expression profile correct")

    # Test 2: APR-246 sensitivity present
    assert "APR-246" in result.profile.drug_sensitivity,  "FAIL: APR-246 missing"
    print("✅ Test 2 — Drug sensitivity includes APR-246")

    # Test 3: Synthetic lethality — BRCA1 co-mutation
    hits = agent.synthetic_lethality_screen("R175H", ["BRCA1", "PTEN"])
    assert any(h["drug"] == "Olaparib"   for h in hits),  "FAIL: Olaparib SL missing"
    assert any(h["drug"] == "Everolimus" for h in hits),  "FAIL: Everolimus SL missing"
    print(f"✅ Test 3 — Synthetic lethality: {[h['drug'] for h in hits]}")

    # Test 4: TME profile present
    tme = result.profile.tme_profile
    assert "immune_infiltration" in tme,                  "FAIL: TME missing"
    print(f"✅ Test 4 — TME profile: {tme['immune_infiltration']} infiltration")

    # Test 5: FHIR structure valid
    fhir = result.fhir_report
    assert fhir["resourceType"] == "Observation",         "FAIL: FHIR resourceType"
    assert len(fhir["component"]) >= 8,                   "FAIL: FHIR component count"
    print("✅ Test 5 — FHIR Observation structure valid")

    # Test 6: PII not in FHIR
    fhir_str = json.dumps(fhir)
    assert "SAM-001" not in fhir_str,                     "FAIL: PII in FHIR"
    print("✅ Test 6 — HIPAA: PII not in FHIR report")

    # Test 7: Invalid mutation
    result_bad = agent.analyse("NOTAVARIANT", "test")
    assert result_bad.error is not None,                  "FAIL: bad mutation not caught"
    print("✅ Test 7 — Invalid mutation format rejected")

    # Test 8: Injection attack
    result_inj = agent.analyse("R175H", "DROP TABLE users; --")
    assert result_inj.error is not None,                  "FAIL: injection not blocked"
    print("✅ Test 8 — SQL injection blocked")

    # Test 9: Rate limiter
    limiter = RateLimiter(max_calls=2, window=60)
    assert limiter.allow() and limiter.allow(),           "FAIL: allow"
    assert not limiter.allow(),                           "FAIL: rate limit not enforced"
    print("✅ Test 9 — Rate limiter enforced")

    # Test 10: Multi-mutation comparison
    comparison = agent.compare_mutations(["R175H", "R248W", "G245S"])
    assert len(comparison) == 3,                          "FAIL: comparison count"
    assert "top_upregulated" in comparison["R175H"],      "FAIL: comparison structure"
    print(f"✅ Test 10 — Multi-mutation comparison: {list(comparison.keys())}")

    # Test 11: All 6 mutations covered
    for mut in ["R175H", "R248W", "R273H", "G245S", "R249S", "R282W"]:
        r = agent.analyse(mut, "Summarise")
        assert r.error is None, f"FAIL: {mut} errored"
    print("✅ Test 11 — All 6 TP53 hotspots covered")

    print("\n=== All 11 tests passed ===\n")
