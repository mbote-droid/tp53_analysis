"""
agents/liquid_biopsy.py — Liquid Biopsy Agent (Agent #11)
=============================================================
Analyses circulating tumour DNA (ctDNA) from liquid biopsy CSV data.
Tracks VAF trends, flags resistance, scores treatment eligibility,
and generates HL7-FHIR-structured clinical reports.

HIPAA compliant — no raw PII in logs or LLM prompts.
Reusable cache layer (utils/cache.py).
Rate-limited — 20 calls/min.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Optional deps (graceful degradation) ────────────────────────────────────
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
# Constants & clinical knowledge base
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_ID = "liquid_biopsy"

VAF_THRESHOLDS = {
    "minimal":    (0.0,  5.0),   # Low tumour burden
    "low":        (5.0,  15.0),  # Monitoring zone
    "moderate":   (15.0, 30.0),  # Treatment response concern
    "high":       (30.0, 60.0),  # Active disease
    "critical":   (60.0, 100.0), # Urgent intervention
}

# VAF rise that triggers a resistance alert (percentage points)
RESISTANCE_DELTA_THRESHOLD = 5.0

# TP53 hotspot → clinical significance
MUTATION_SIGNIFICANCE = {
    "R175H": {
        "class":       "GOF",
        "oncogenicity":"high",
        "apr246":      True,
        "mdm2_score":  3,
        "note":        "Most common TP53 GOF; conformational mutant; APR-246 candidate",
    },
    "R248W": {
        "class":       "GOF",
        "oncogenicity":"high",
        "apr246":      True,
        "mdm2_score":  3,
        "note":        "DNA-contact mutant; high metastatic risk; APR-246 candidate",
    },
    "R273H": {
        "class":       "GOF",
        "oncogenicity":"high",
        "apr246":      False,
        "mdm2_score":  2,
        "note":        "DNA-contact mutant; cisplatin resistance association",
    },
    "G245S": {
        "class":       "LOF",
        "oncogenicity":"moderate",
        "apr246":      False,
        "mdm2_score":  2,
        "note":        "Structural mutant; reduced apoptotic signalling",
    },
    "R249S": {
        "class":       "LOF",
        "oncogenicity":"moderate",
        "apr246":      False,
        "mdm2_score":  1,
        "note":        "Associated with aflatoxin exposure; hepatocellular carcinoma link",
    },
    "R282W": {
        "class":       "GOF",
        "oncogenicity":"high",
        "apr246":      False,
        "mdm2_score":  2,
        "note":        "Structural mutant; chemotherapy resistance",
    },
}

# Kenya-specific drug availability (KEML 2023)
KEML_AVAILABILITY = {
    "APR-246":       "Clinical trial only — KNH / AKUH Nairobi",
    "Doxorubicin":   "National referral + county hospitals",
    "Paclitaxel":    "National referral + level-5 county hospitals",
    "Cisplatin":     "National referral + level-5 county hospitals",
    "Carboplatin":   "National referral + level-5 county hospitals",
    "5-Fluorouracil":"Level-4+ county hospitals",
    "Olaparib":      "Import only — private facilities (Nairobi/Mombasa)",
    "Tamoxifen":     "National referral + most county hospitals",
}

RATE_LIMIT_CALLS  = 20
RATE_LIMIT_WINDOW = 60  # seconds

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BiopsyRecord:
    patient_id:    str
    sample_date:   str
    mutation:      str
    vaf:           float
    cancer_type:   str
    county:        str
    hospital:      str
    treatment:     str
    clinician_id:  str
    notes:         str

@dataclass
class ResistanceAlert:
    patient_hash:  str
    mutation:      str
    prev_vaf:      float
    curr_vaf:      float
    delta:         float
    severity:      str          # "warning" | "critical"
    message:       str

@dataclass
class PatientSummary:
    patient_hash:   str
    mutation:       str
    cancer_type:    str
    county:         str
    hospital:       str
    vaf_trend:      list[float]
    dates:          list[str]
    current_vaf:    float
    vaf_class:      str
    treatment:      str
    alerts:         list[ResistanceAlert]
    apr246_eligible:bool
    mdm2_score:     int
    keml_drugs:     list[str]
    clinical_note:  str

@dataclass
class AnalysisResult:
    agent_id:       str = AGENT_ID
    timestamp:      str = ""
    total_patients: int = 0
    summaries:      list[PatientSummary] = field(default_factory=list)
    alerts:         list[ResistanceAlert] = field(default_factory=list)
    fhir_report:    dict = field(default_factory=dict)
    query_response: str = ""
    cache_hit:      bool = False
    error:          Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Rate limiter (reusable)
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_calls: int = RATE_LIMIT_CALLS,
                 window: int = RATE_LIMIT_WINDOW) -> None:
        self._lock      = threading.Lock()
        self._calls:list[float] = []
        self.max_calls  = max_calls
        self.window     = window

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < self.window]
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


# ═══════════════════════════════════════════════════════════════════════════════
# Input sanitiser
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_QUERY = re.compile(r"^[\w\s\?\.\,\-\'\"\/\(\)]+$")
_INJECTION   = re.compile(
    r"(drop\s+table|delete\s+from|<script|import\s+os|__import__|eval\()",
    re.IGNORECASE,
)

def sanitise_query(query: str) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    q = query.strip()
    if len(q) > 2000:
        raise ValueError("Query exceeds 2000 character limit")
    if _INJECTION.search(q):
        raise ValueError("Potentially unsafe content in query")
    if not _SAFE_QUERY.match(q):
        raise ValueError("Query contains disallowed characters")
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# PII hasher
# ═══════════════════════════════════════════════════════════════════════════════

def hash_patient_id(patient_id: str) -> str:
    return "PAT-" + hashlib.sha256(patient_id.encode()).hexdigest()[:12].upper()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: Path) -> list[BiopsyRecord]:
    """Load and validate liquid biopsy CSV. Raises on bad schema."""
    required_cols = {
        "patient_id", "sample_date", "mutation", "vaf_percent",
        "cancer_type", "county", "hospital", "current_treatment",
        "clinician_id", "notes",
    }
    records: list[BiopsyRecord] = []

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = required_cols - cols
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for i, row in enumerate(reader, start=2):
            try:
                vaf = float(row["vaf_percent"])
                if not (0.0 <= vaf <= 100.0):
                    raise ValueError(f"VAF out of range at row {i}: {vaf}")
                records.append(BiopsyRecord(
                    patient_id   = row["patient_id"].strip(),
                    sample_date  = row["sample_date"].strip(),
                    mutation     = row["mutation"].strip(),
                    vaf          = vaf,
                    cancer_type  = row["cancer_type"].strip(),
                    county       = row["county"].strip(),
                    hospital     = row["hospital"].strip(),
                    treatment    = row["current_treatment"].strip(),
                    clinician_id = row["clinician_id"].strip(),
                    notes        = row["notes"].strip(),
                ))
            except (ValueError, KeyError) as exc:
                log.warning("Skipping malformed row %d: %s", i, exc)

    log.info("Loaded %d records from %s", len(records), csv_path)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical analysis engine
# ═══════════════════════════════════════════════════════════════════════════════

def classify_vaf(vaf: float) -> str:
    for label, (lo, hi) in VAF_THRESHOLDS.items():
        if lo <= vaf < hi:
            return label
    return "critical"


def detect_resistance(records: list[BiopsyRecord]) -> list[ResistanceAlert]:
    """
    Group records by patient+mutation, sort by date, flag VAF rises
    above RESISTANCE_DELTA_THRESHOLD as resistance events.
    """
    groups: dict[str, list[BiopsyRecord]] = defaultdict(list)
    for r in records:
        key = f"{r.patient_id}::{r.mutation}"
        groups[key].append(r)

    alerts: list[ResistanceAlert] = []
    for key, recs in groups.items():
        recs_sorted = sorted(recs, key=lambda x: x.sample_date)
        for prev, curr in zip(recs_sorted, recs_sorted[1:]):
            delta = curr.vaf - prev.vaf
            if delta >= RESISTANCE_DELTA_THRESHOLD:
                severity = "critical" if curr.vaf >= 30.0 else "warning"
                alerts.append(ResistanceAlert(
                    patient_hash = hash_patient_id(curr.patient_id),
                    mutation     = curr.mutation,
                    prev_vaf     = prev.vaf,
                    curr_vaf     = curr.vaf,
                    delta        = round(delta, 2),
                    severity     = severity,
                    message      = (
                        f"{curr.mutation} VAF rose {delta:.1f}pp "
                        f"({prev.vaf:.1f}% → {curr.vaf:.1f}%) — "
                        f"possible treatment resistance"
                    ),
                ))
    return alerts


def build_patient_summaries(
    records: list[BiopsyRecord],
    alerts: list[ResistanceAlert],
) -> list[PatientSummary]:
    groups: dict[str, list[BiopsyRecord]] = defaultdict(list)
    for r in records:
        groups[r.patient_id].append(r)

    alert_map: dict[str, list[ResistanceAlert]] = defaultdict(list)
    for a in alerts:
        alert_map[a.patient_hash].append(a)

    summaries: list[PatientSummary] = []
    for pid, recs in groups.items():
        recs_sorted  = sorted(recs, key=lambda x: x.sample_date)
        latest       = recs_sorted[-1]
        phash        = hash_patient_id(pid)
        sig          = MUTATION_SIGNIFICANCE.get(latest.mutation, {})
        vaf_trend    = [r.vaf for r in recs_sorted]
        current_vaf  = latest.vaf

        # KEML drug suggestions
        keml = []
        if sig.get("apr246"):
            keml.append("APR-246 (trial)")
        keml += ["Doxorubicin", "Paclitaxel", "Cisplatin"]

        note = sig.get("note", "No specific clinical note for this mutation.")
        if current_vaf >= 30.0:
            note += " ⚠ High VAF — escalate therapy review."

        summaries.append(PatientSummary(
            patient_hash    = phash,
            mutation        = latest.mutation,
            cancer_type     = latest.cancer_type,
            county          = latest.county,
            hospital        = latest.hospital,
            vaf_trend       = vaf_trend,
            dates           = [r.sample_date for r in recs_sorted],
            current_vaf     = current_vaf,
            vaf_class       = classify_vaf(current_vaf),
            treatment       = latest.treatment,
            alerts          = alert_map.get(phash, []),
            apr246_eligible = bool(sig.get("apr246")),
            mdm2_score      = sig.get("mdm2_score", 0),
            keml_drugs      = keml,
            clinical_note   = note,
        ))

    # Sort: most critical first
    summaries.sort(key=lambda s: s.current_vaf, reverse=True)
    return summaries


# ═══════════════════════════════════════════════════════════════════════════════
# FHIR report builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_fhir_report(summaries: list[PatientSummary]) -> dict:
    """
    HL7 FHIR R4 DiagnosticReport structure.
    All patient references use hashed IDs — HIPAA compliant.
    """
    return {
        "resourceType": "Bundle",
        "type":         "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "status":       "final",
                    "code": {
                        "coding": [{
                            "system":  "http://loinc.org",
                            "code":    "85319-2",
                            "display": "TP53 gene mutation analysis in Blood or Tissue",
                        }]
                    },
                    "subject": {"reference": f"Patient/{s.patient_hash}"},
                    "result": [
                        {
                            "display": f"{s.mutation} VAF {s.current_vaf:.1f}% "
                                       f"[{s.vaf_class}] — {s.cancer_type}",
                        }
                    ],
                    "conclusion": s.clinical_note,
                    "extension": [
                        {"url": "apr246_eligible",  "valueBoolean": s.apr246_eligible},
                        {"url": "mdm2_inhibitor_score", "valueInteger": s.mdm2_score},
                        {"url": "resistance_alerts",
                         "valueString": "; ".join(a.message for a in s.alerts) or "None"},
                        {"url": "keml_drugs",
                         "valueString": ", ".join(s.keml_drugs)},
                        {"url": "county",
                         "valueString": s.county},
                    ],
                }
            }
            for s in summaries
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LLM query (plugs into existing RAG chain or falls back to structured output)
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_query(query: str, context: str) -> str:
    """
    Attempt LLM call via existing TP53RAGChain.
    Falls back to structured summary if RAG unavailable.
    """
    try:
        from rag_chain import TP53RAGChain   # project import
        chain  = TP53RAGChain()
        result = chain.query(f"{query}\n\nContext:\n{context}")
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)
    except Exception as exc:
        log.warning("LLM unavailable (%s) — using structured fallback", exc)
        return f"[Structured Analysis]\n{context}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Agent Class
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidBiopsyAgent:
    """
    Liquid Biopsy ctDNA Analysis Agent.

    Parameters
    ----------
    csv_path : Path
        Path to liquid_biopsy_tracker.csv
    ttl : int
        Cache TTL in seconds (default 1800)
    """

    def __init__(
        self,
        csv_path: Path = Path("data/liquid_biopsy_tracker.csv"),
        ttl: int = 1800,
    ) -> None:
        self.csv_path    = Path(csv_path)
        self._rate       = RateLimiter()
        self._cache      = get_cache(ttl=ttl) if _CACHE_AVAILABLE else None
        self._records:   list[BiopsyRecord] = []
        self._summaries: list[PatientSummary] = []
        self._alerts:    list[ResistanceAlert] = []
        self._loaded     = False

    # ── Data loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load CSV and run full clinical analysis pipeline."""
        self._records   = load_csv(self.csv_path)
        self._alerts    = detect_resistance(self._records)
        self._summaries = build_patient_summaries(self._records, self._alerts)
        self._loaded    = True
        log.info(
            "LiquidBiopsyAgent ready — %d patients, %d alerts",
            len(self._summaries), len(self._alerts),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, query: str = "Summarise all patient ctDNA results") -> AnalysisResult:
        """
        Main entry point. Accepts a natural-language query about the cohort.
        Returns AnalysisResult with structured data + LLM narrative.
        """
        # 1. Rate limit
        if not self._rate.allow():
            return AnalysisResult(
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error     = "Rate limit exceeded — max 20 calls/min",
            )

        # 2. Sanitise
        try:
            query = sanitise_query(query)
        except ValueError as exc:
            return AnalysisResult(
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error     = str(exc),
            )

        # 3. Cache lookup
        if self._cache:
            cached = self._cache.get(AGENT_ID, query)
            if cached:
                result = AnalysisResult(**cached)
                result.cache_hit = True
                return result

        # 4. Load data
        try:
            self._ensure_loaded()
        except Exception as exc:
            return AnalysisResult(
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error     = f"Data load failed: {exc}",
            )

        # 5. Build context for LLM (no raw PII)
        context = self._build_context()

        # 6. LLM query
        response = _llm_query(query, context)

        # 7. FHIR report
        fhir = build_fhir_report(self._summaries)

        # 8. Assemble result
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = AnalysisResult(
            timestamp      = ts,
            total_patients = len(self._summaries),
            summaries      = self._summaries,
            alerts         = self._alerts,
            fhir_report    = fhir,
            query_response = response,
            cache_hit      = False,
        )

        # 9. Cache result (serialise summaries as dicts)
        if self._cache:
            try:
                self._cache.set(AGENT_ID, query, {
                    "timestamp":      ts,
                    "total_patients": result.total_patients,
                    "query_response": result.query_response,
                    "alerts":         [asdict(a) for a in self._alerts],
                    "fhir_report":    fhir,
                    "cache_hit":      False,
                    "error":          None,
                    "agent_id":       AGENT_ID,
                    "summaries":      [],   # summaries omitted from cache (size)
                })
            except Exception as exc:
                log.warning("Cache write failed: %s", exc)

        return result

    def get_patient(self, patient_hash: str) -> Optional[PatientSummary]:
        """Retrieve a single patient summary by hashed ID."""
        self._ensure_loaded()
        for s in self._summaries:
            if s.patient_hash == patient_hash:
                return s
        return None

    def critical_alerts(self) -> list[ResistanceAlert]:
        """Return only CRITICAL severity alerts."""
        self._ensure_loaded()
        return [a for a in self._alerts if a.severity == "critical"]

    def apr246_candidates(self) -> list[PatientSummary]:
        """Return patients eligible for APR-246 trial."""
        self._ensure_loaded()
        return [s for s in self._summaries if s.apr246_eligible]

    def stats(self) -> dict:
        """Cohort-level statistics — safe for dashboard display."""
        self._ensure_loaded()
        mutation_counts: dict[str, int] = defaultdict(int)
        county_counts:   dict[str, int] = defaultdict(int)
        for s in self._summaries:
            mutation_counts[s.mutation] += 1
            county_counts[s.county]     += 1
        return {
            "total_patients":    len(self._summaries),
            "total_alerts":      len(self._alerts),
            "critical_alerts":   sum(1 for a in self._alerts if a.severity == "critical"),
            "apr246_candidates": len(self.apr246_candidates()),
            "mutation_breakdown": dict(mutation_counts),
            "county_breakdown":   dict(county_counts),
            "avg_vaf":            round(
                sum(s.current_vaf for s in self._summaries) / max(len(self._summaries), 1),
                2,
            ),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_context(self) -> str:
        """Build LLM context from summaries — PII replaced with hashes."""
        lines = [
            f"Liquid Biopsy Cohort Analysis — {len(self._summaries)} patients\n",
            "=" * 60,
        ]
        for s in self._summaries:
            lines += [
                f"\nPatient: {s.patient_hash}",
                f"  Mutation     : {s.mutation} ({s.cancer_type})",
                f"  County       : {s.county} | Hospital: {s.hospital}",
                f"  VAF trend    : {s.vaf_trend} → current {s.current_vaf:.1f}% [{s.vaf_class}]",
                f"  Treatment    : {s.treatment}",
                f"  APR-246      : {'Eligible' if s.apr246_eligible else 'Not eligible'}",
                f"  MDM2 score   : {s.mdm2_score}/10",
                f"  KEML options : {', '.join(s.keml_drugs)}",
                f"  Clinical note: {s.clinical_note}",
            ]
            if s.alerts:
                lines.append(f"  ⚠ ALERTS     : {'; '.join(a.message for a in s.alerts)}")
        if self._alerts:
            lines += [
                "\n" + "=" * 60,
                f"RESISTANCE ALERTS ({len(self._alerts)} total):",
            ]
            for a in self._alerts:
                lines.append(f"  [{a.severity.upper()}] {a.message}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test / reverse-engineering suite
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tempfile, os
    logging.basicConfig(level=logging.INFO)
    print("\n=== LiquidBiopsyAgent Self-Test ===\n")

    # ── Build minimal mock CSV ────────────────────────────────────────────────
    MOCK_CSV = """patient_id,sample_date,mutation,vaf_percent,cancer_type,county,hospital,current_treatment,clinician_id,notes
SAM-NBI-01,2024-01-10,R175H,18.4,Breast Cancer,Nairobi,KNH,Doxorubicin,DR-001,Baseline
SAM-NBI-01,2024-03-15,R175H,27.3,Breast Cancer,Nairobi,KNH,Doxorubicin,DR-001,Rising VAF — resistance suspected
SAM-NBI-02,2024-02-01,R248W,8.2,Colorectal Cancer,Mombasa,Coast General,FOLFOX,DR-002,Monitoring
SAM-NBI-03,2024-01-20,G245S,55.1,Lung Cancer,Kisumu,Jaramogi Hospital,Cisplatin,DR-003,Advanced disease
SAM-NBI-04,2024-03-01,R273H,3.1,Ovarian Cancer,Uasin Gishu,Eldoret Teaching,Paclitaxel,DR-004,Early detection
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(MOCK_CSV)
        tmp_csv = Path(f.name)

    try:
        agent = LiquidBiopsyAgent(csv_path=tmp_csv)
        agent.load()

        # Test 1: Basic load
        assert len(agent._summaries) == 4, "FAIL: patient count"
        print("✅ Test 1 — CSV load (4 patients)")

        # Test 2: Resistance detection
        alerts = agent.critical_alerts()
        assert any(a.mutation == "R175H" for a in agent._alerts), \
            "FAIL: R175H resistance not detected"
        print(f"✅ Test 2 — Resistance detection ({len(agent._alerts)} alert(s))")

        # Test 3: APR-246 eligibility
        candidates = agent.apr246_candidates()
        assert any(c.mutation in ("R175H", "R248W") for c in candidates), \
            "FAIL: APR-246 candidates wrong"
        print(f"✅ Test 3 — APR-246 candidates ({len(candidates)})")

        # Test 4: VAF classification
        high_vaf = next(s for s in agent._summaries if s.mutation == "G245S")
        assert high_vaf.vaf_class in ("high", "critical"), "FAIL: VAF classification"
        print(f"✅ Test 4 — VAF classification: G245S → {high_vaf.vaf_class}")

        # Test 5: FHIR report structure
        fhir = build_fhir_report(agent._summaries)
        assert fhir["resourceType"] == "Bundle", "FAIL: FHIR resourceType"
        assert len(fhir["entry"]) == 4, "FAIL: FHIR entry count"
        print("✅ Test 5 — FHIR report structure valid")

        # Test 6: PII — raw patient ID must NOT appear in context
        context = agent._build_context()
        assert "SAM-NBI-01" not in context, "FAIL: PII leaked into context"
        print("✅ Test 6 — HIPAA: PII not in LLM context")

        # Test 7: Input sanitisation — injection attack
        result = agent.analyse("DROP TABLE query_cache; --")
        assert result.error is not None, "FAIL: injection not caught"
        print("✅ Test 7 — Injection attack rejected")

        # Test 8: Rate limiter (exhaust then block)
        limiter = RateLimiter(max_calls=3, window=60)
        assert limiter.allow() and limiter.allow() and limiter.allow(), "FAIL: allow"
        assert not limiter.allow(), "FAIL: rate limit not enforced"
        print("✅ Test 8 — Rate limiter enforced")

        # Test 9: Stats
        stats = agent.stats()
        assert stats["total_patients"] == 4, "FAIL: stats patient count"
        print(f"✅ Test 9 — Stats: {stats}")

        # Test 10: Missing CSV
        bad_agent = LiquidBiopsyAgent(csv_path=Path("/nonexistent/path.csv"))
        result = bad_agent.analyse("test")
        assert result.error is not None, "FAIL: missing CSV not handled"
        print("✅ Test 10 — Missing CSV handled gracefully")

        print("\n=== All 10 tests passed ===\n")

    finally:
        os.unlink(tmp_csv)
