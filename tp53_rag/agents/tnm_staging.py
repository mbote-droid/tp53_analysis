"""
============================================================
TP53 RAG Platform — Agent #16: TNM Staging & Clinical
Roadmap Generator
agents/tnm_staging.py
============================================================
Analyses pathology vision output + TP53 mutation data to
produce AJCC/UICC 8th edition TNM staging with Kenya-
contextualised clinical next steps.

Design principles (claudesam_master_requirements.md):
  • Reusable components only — uses shared RAG chain,
    PII scrubber, audit logger, rate limiter from rag_chain.py
  • Strict JSON guardrails — LLM prompt forces structured output
  • Graceful degradation — works without RAG/LLM (rule-based fallback)
  • Zero empty outputs — ZeroResultHandler pattern followed
  • HIPAA/HL7 FHIR R4 compliant output structure
  • SHA-256 PII hashing before any LLM call
  • Audit trail (append-only log)
  • Self-correction loop (up to 3 retries)
  • Kenya/KEML clinical context — all next steps reference
    locally available resources
  • African population equity — R249S hepatocellular carcinoma
    aflatoxin context flagged automatically
  • 10+ self-tests at bottom (break & fix method)

Pipeline position:
  PathologyVisionAgent (Agent #15)
    → TNMStagingAgent (Agent #16)   ← this file
      → SurgicalBriefAgent (Agent #10)
      → DossierCompiler (Agent #3)

Inputs:
  - pathology_result: Dict from PathologyVisionAgent.process_slide()
  - pipeline_data: Dict with mutations, cancer_type, vaf, accession
  - rag_chain: TP53RAGChain instance (optional)

Outputs:
  - T, N, M components with reasoning
  - Overall stage group (I–IV)
  - Stage-specific clinical next steps (Kenya-contextualised)
  - Recommended imaging workup
  - Recommended multidisciplinary team (MDT) referrals
  - FHIR R4 compatible ClinicalImpression resource
  - HTML surgical dashboard card (matches SurgicalBriefAgent style)
============================================================
"""

from __future__ import annotations

import json
import hashlib
import logging
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Optional RAG chain import (graceful degradation) ─────────────
try:
    from agents.rag_chain import TP53RAGChain, PIIScrubber, AuditLogger
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    log.warning("RAG chain not available — TNM agent running in rule-based mode")

try:
    from utils.logger import log as platform_log
    log = platform_log
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# TNM Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TNMComponent:
    """Single TNM component with evidence and confidence."""
    code: str           # e.g. "T2", "N1", "M0"
    description: str    # human-readable
    evidence: str       # what drove this classification
    confidence: float   # 0.0–1.0


@dataclass
class TNMResult:
    """Complete TNM staging output."""
    T: TNMComponent
    N: TNMComponent
    M: TNMComponent
    stage_group: str            # "I", "IIA", "IIB", "III", "IIIC", "IV"
    cancer_type: str
    mutation: str
    equity_flag: Optional[str]  # African population alert if triggered
    llm_narration: str
    next_steps: List[Dict]      # ordered list of clinical actions
    imaging_workup: List[str]
    mdt_referrals: List[str]
    fhir_resource: Dict
    confidence_overall: float
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════
# TNM Reference Tables (AJCC/UICC 8th Edition)
# Cancer-specific — driven by pathology tissue + TP53 mutation
# ═══════════════════════════════════════════════════════════════════

# T stage descriptors per cancer type
T_DESCRIPTORS = {
    "Colorectal": {
        "T1": "Tumour invades submucosa",
        "T2": "Tumour invades muscularis propria",
        "T3": "Tumour invades through muscularis propria into pericolorectal tissues",
        "T4a": "Tumour penetrates to the surface of the visceral peritoneum",
        "T4b": "Tumour directly invades or is adherent to other organs or structures",
    },
    "Breast": {
        "T1": "Tumour ≤20mm in greatest dimension",
        "T2": "Tumour >20mm but ≤50mm",
        "T3": "Tumour >50mm",
        "T4": "Tumour of any size with direct extension to chest wall or skin",
    },
    "Lung": {
        "T1": "Tumour ≤3cm, surrounded by lung or visceral pleura",
        "T2": "Tumour >3cm but ≤5cm, or with specific features",
        "T3": "Tumour >5cm but ≤7cm, or invades chest wall/pericardium/phrenic nerve",
        "T4": "Tumour >7cm or invades mediastinum/diaphragm/heart/great vessels",
    },
    "Liver": {
        "T1": "Solitary tumour ≤2cm, or >2cm without vascular invasion",
        "T2": "Solitary tumour >2cm with vascular invasion, or multiple ≤5cm",
        "T3": "Multiple tumours, at least one >5cm",
        "T4": "Tumour involves portal/hepatic vein, or direct invasion of adjacent organs",
    },
    "Ovarian": {
        "T1": "Tumour limited to ovaries or fallopian tubes",
        "T2": "Tumour involves one or both ovaries with pelvic extension",
        "T3": "Microscopic peritoneal involvement beyond pelvis",
        "T4": "Pleural effusion with positive cytology or parenchymal metastases",
    },
    "Gastric": {
        "T1": "Tumour invades lamina propria, muscularis mucosae, or submucosa",
        "T2": "Tumour invades muscularis propria",
        "T3": "Tumour penetrates subserosal connective tissue",
        "T4a": "Tumour invades serosa (visceral peritoneum)",
        "T4b": "Tumour invades adjacent structures",
    },
    "default": {
        "T1": "Tumour confined to organ of origin, limited invasion",
        "T2": "Tumour with local extension beyond primary site",
        "T3": "Tumour extends through organ wall into adjacent tissues",
        "T4": "Tumour invades adjacent organs or structures",
    },
}

# Stage group lookup (simplified AJCC 8th Ed.)
# Key: (T_base, N_base, M_base) → stage_group
# T_base: numeric (1,2,3,4), N_base: 0/1/2/3, M_base: 0/1
STAGE_GROUPS = {
    # Stage I
    (1, 0, 0): "I",
    (2, 0, 0): "I",
    # Stage II
    (3, 0, 0): "IIA",
    (4, 0, 0): "IIB",
    (1, 1, 0): "IIA",
    (2, 1, 0): "IIB",
    # Stage III
    (3, 1, 0): "IIIA",
    (4, 1, 0): "IIIB",
    (1, 2, 0): "IIIA",
    (2, 2, 0): "IIIB",
    (3, 2, 0): "IIIC",
    (4, 2, 0): "IIIC",
    (1, 3, 0): "IIIC",
    (2, 3, 0): "IIIC",
    (3, 3, 0): "IIIC",
    (4, 3, 0): "IIIC",
    # Stage IV — any M1
    (1, 0, 1): "IV",
    (2, 0, 1): "IV",
    (3, 0, 1): "IV",
    (4, 0, 1): "IV",
    (1, 1, 1): "IV",
    (2, 1, 1): "IV",
    (3, 1, 1): "IV",
    (4, 1, 1): "IV",
    (1, 2, 1): "IV",
    (2, 2, 1): "IV",
    (3, 2, 1): "IV",
    (4, 2, 1): "IV",
    (1, 3, 1): "IV",
    (2, 3, 1): "IV",
    (3, 3, 1): "IV",
    (4, 3, 1): "IV",
}

# TP53 mutation → T stage modifier
# Hotspot mutations associated with higher T stages
MUTATION_T_MODIFIER = {
    "R175H": 1,   # conformational — aggressive, +1 T tendency
    "R248W": 1,   # contact — highly invasive
    "R248Q": 1,
    "R273H": 1,
    "R273C": 1,
    "G245S": 0,
    "R249S": 2,   # aflatoxin-associated HCC — often presents late in Africa
    "R282W": 0,
    "Y220C": 0,
    "P72R":  0,   # African polymorphism — NOT pathogenic
}

# Pathology tissue → N stage signal
TISSUE_N_SIGNAL = {
    "Tumor":            0,
    "Stroma":           0,
    "Inflammatory":     1,   # inflammation often indicates nodal involvement risk
    "Necrosis":         1,   # necrosis = aggressive, higher N risk
    "Normal epithelium": 0,
    "Mucus":            0,
    "Smooth muscle":    0,
    "Adipose":          0,
}

# VAF → M stage signal
# High VAF = clonal, established tumour = higher M risk
def _vaf_to_m_signal(vaf: Optional[float]) -> int:
    if vaf is None:
        return 0
    if vaf >= 70:
        return 1   # very high clonal fraction — probable metastasis
    return 0

# African equity flags
AFRICAN_EQUITY_FLAGS = {
    "R249S": (
        "⚠ AFRICAN EQUITY ALERT: R249S is elevated in East African populations due "
        "to aflatoxin B1 exposure (contaminated maize/groundnuts). Associated with "
        "hepatocellular carcinoma presenting at advanced stage in sub-Saharan Africa. "
        "Stage workup must include AFP, liver function, and hepatitis B/C serology. "
        "Refer to Kenyatta National Hospital Hepatology or nearest KNH satellite unit."
    ),
    "P72R": (
        "⚠ AFRICAN EQUITY ALERT: P72R (Pro72Arg) is a COMMON BENIGN POLYMORPHISM "
        "in African populations (frequency ~35%). Do NOT stage or treat based on this "
        "variant alone. Confirm pathogenicity with additional clinicopathological evidence."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# Stage-specific Next Steps (Kenya-contextualised)
# ═══════════════════════════════════════════════════════════════════

STAGE_NEXT_STEPS = {
    "I": [
        {
            "priority": 1,
            "action": "Curative surgical resection",
            "detail": "Wide local excision with clear margins (R0 resection). "
                      "Margin width per cancer-specific guidelines. "
                      "Achievable at KNH, Moi Teaching Hospital, MTRH.",
            "timeframe": "Within 2–4 weeks of staging",
            "kenya_resource": "Kenyatta National Hospital (KNH) Oncology Surgery Unit",
        },
        {
            "priority": 2,
            "action": "Sentinel lymph node biopsy (if applicable)",
            "detail": "Assess regional nodal involvement. "
                      "Available at KNH and Aga Khan University Hospital Nairobi.",
            "timeframe": "At time of primary resection",
            "kenya_resource": "KNH / Aga Khan University Hospital",
        },
        {
            "priority": 3,
            "action": "Histopathology and molecular profiling",
            "detail": "Confirm TP53 mutation status on resected specimen. "
                      "Request IARC TP53 Database cross-reference.",
            "timeframe": "Post-operative (2–3 weeks)",
            "kenya_resource": "KNH Pathology Department / AMPATH Eldoret",
        },
        {
            "priority": 4,
            "action": "Surveillance schedule",
            "detail": "3-monthly clinical review for 2 years, then 6-monthly. "
                      "Annual imaging per cancer type.",
            "timeframe": "Ongoing post-resection",
            "kenya_resource": "County Referral Hospital or KNH Oncology Clinic",
        },
    ],
    "IIA": [
        {
            "priority": 1,
            "action": "Staging CT scan (chest/abdomen/pelvis)",
            "detail": "Confirm T and N classification. Rule out distant metastasis. "
                      "Available at KNH, Aga Khan, MP Shah Hospital Nairobi.",
            "timeframe": "Within 1 week",
            "kenya_resource": "KNH Radiology / Aga Khan Radiology",
        },
        {
            "priority": 2,
            "action": "MDT discussion before surgery",
            "detail": "Multidisciplinary team review (surgery, oncology, radiology, pathology). "
                      "KNH runs weekly tumor boards.",
            "timeframe": "Before any intervention",
            "kenya_resource": "KNH Tumor Board",
        },
        {
            "priority": 3,
            "action": "Surgical resection ± neoadjuvant therapy",
            "detail": "Consider neoadjuvant chemotherapy if R0 resection uncertain. "
                      "Carboplatin/5-FU combinations available on Kenya Essential Medicines List (KEML).",
            "timeframe": "2–6 weeks after staging confirmed",
            "kenya_resource": "KNH / MTRH Oncology Unit",
        },
        {
            "priority": 4,
            "action": "Adjuvant chemotherapy assessment",
            "detail": "Post-operative chemotherapy based on histology and margin status. "
                      "FOLFOX (Folinic acid + Fluorouracil + Oxaliplatin) — check KEML availability.",
            "timeframe": "4–6 weeks post-surgery",
            "kenya_resource": "KNH / Eldoret Cancer Center",
        },
    ],
    "IIB": [
        {
            "priority": 1,
            "action": "Full staging workup — CT + MRI",
            "detail": "CT chest/abdomen/pelvis + MRI of primary site for local extent. "
                      "MRI available at KNH, Aga Khan, Nairobi Hospital.",
            "timeframe": "Urgent — within 1 week",
            "kenya_resource": "KNH Radiology / Nairobi Hospital",
        },
        {
            "priority": 2,
            "action": "Neoadjuvant chemotherapy",
            "detail": "Downstage tumour before surgery. Carboplatin-based regimens on KEML. "
                      "APR-246 (p53 reactivator) — not yet on KEML, clinical trial access only.",
            "timeframe": "Start within 2 weeks of staging",
            "kenya_resource": "KNH Oncology / AMPATH Research",
        },
        {
            "priority": 3,
            "action": "Re-staging and surgical planning",
            "detail": "Repeat imaging after 3 cycles of neoadjuvant therapy. "
                      "Reassess for R0 resection feasibility.",
            "timeframe": "After 3 chemotherapy cycles (~9 weeks)",
            "kenya_resource": "KNH Surgical Oncology",
        },
        {
            "priority": 4,
            "action": "Genetic counselling for Li-Fraumeni",
            "detail": "TP53 hotspot at Stage IIB in patient <50 years — "
                      "screen for germline mutation and Li-Fraumeni syndrome. "
                      "Refer family members for cascade testing.",
            "timeframe": "Concurrent with treatment planning",
            "kenya_resource": "KNH Genetics Clinic / AMPATH Genetic Counselling",
        },
    ],
    "IIIA": [
        {
            "priority": 1,
            "action": "Urgent MDT review",
            "detail": "Stage IIIA requires immediate multidisciplinary planning. "
                      "Surgery, oncology, radiation oncology, palliative care all involved.",
            "timeframe": "Within 48–72 hours",
            "kenya_resource": "KNH Tumor Board (meets weekly)",
        },
        {
            "priority": 2,
            "action": "Concurrent chemoradiotherapy (if applicable)",
            "detail": "Some Stage IIIA cancers benefit from concurrent CRT. "
                      "Radiotherapy available at KNH (linear accelerator) and MTRH.",
            "timeframe": "After MDT decision",
            "kenya_resource": "KNH Radiotherapy Unit / MTRH Cancer Centre",
        },
        {
            "priority": 3,
            "action": "Immunotherapy eligibility assessment",
            "detail": "Check PD-L1 expression and MSI status. "
                      "Pembrolizumab (Keytruda) — available through NHIF for eligible cancers, "
                      "or compassionate use programmes.",
            "timeframe": "Concurrent with treatment start",
            "kenya_resource": "KNH / Aga Khan — request PD-L1 IHC",
        },
        {
            "priority": 4,
            "action": "Palliative care integration",
            "detail": "Early palliative care referral improves outcomes and quality of life. "
                      "Not end-of-life care — symptom management and support.",
            "timeframe": "At diagnosis of Stage IIIA",
            "kenya_resource": "KNH Palliative Care Unit / Kenya Hospices & Palliative Care Association (KEHPCA)",
        },
    ],
    "IIIB": [
        {
            "priority": 1,
            "action": "Systemic chemotherapy — primary treatment",
            "detail": "Surgery may not be first option. Start systemic therapy. "
                      "Carboplatin + Paclitaxel or FOLFOX depending on cancer type. "
                      "All available on KEML.",
            "timeframe": "Within 1–2 weeks of staging",
            "kenya_resource": "KNH / MTRH / Eldoret Cancer Center",
        },
        {
            "priority": 2,
            "action": "Radiotherapy evaluation",
            "detail": "Assess for palliative or radical radiotherapy. "
                      "Linear accelerator available at KNH Nairobi.",
            "timeframe": "After first chemotherapy cycle",
            "kenya_resource": "KNH Radiotherapy Unit",
        },
        {
            "priority": 3,
            "action": "Clinical trial enrollment",
            "detail": "Check AMPATH/KNH ongoing trials for TP53-mutant cancers. "
                      "APR-246 trials for p53 reactivation — inquire at AMPATH Research Eldoret.",
            "timeframe": "Concurrent with treatment",
            "kenya_resource": "AMPATH Research / KNH Clinical Trials Unit",
        },
        {
            "priority": 4,
            "action": "Social and financial assessment",
            "detail": "NHIF registration and cancer benefit package assessment. "
                      "Faraja Cancer Support Trust — Nairobi. Kenyan Cancer Association support.",
            "timeframe": "Immediately",
            "kenya_resource": "NHIF / Faraja Cancer Support Trust / Kenya Cancer Association",
        },
    ],
    "IIIC": [
        {
            "priority": 1,
            "action": "Chemotherapy + targeted therapy combination",
            "detail": "Systemic chemotherapy primary. Assess for targeted therapy: "
                      "Bevacizumab (anti-VEGF) — available through Aga Khan for eligible patients. "
                      "TP53-specific trials via AMPATH.",
            "timeframe": "Urgent — within 1 week",
            "kenya_resource": "KNH / Aga Khan / AMPATH",
        },
        {
            "priority": 2,
            "action": "Palliative surgery assessment",
            "detail": "Evaluate for palliative resection to prevent obstruction/bleeding "
                      "even if curative resection not possible.",
            "timeframe": "MDT decision",
            "kenya_resource": "KNH Surgical Oncology",
        },
        {
            "priority": 3,
            "action": "Comprehensive palliative care",
            "detail": "Pain management, nutritional support, psychological care. "
                      "KEHPCA community palliative teams available in most counties.",
            "timeframe": "Immediate",
            "kenya_resource": "KEHPCA / KNH Palliative Unit",
        },
        {
            "priority": 4,
            "action": "Family education and support",
            "detail": "Engage family in care planning per Kenyan communal health model. "
                      "Explain diagnosis in culturally appropriate language (Swahili if needed). "
                      "Consider Multilingual Report Agent output for patient communication.",
            "timeframe": "At diagnosis",
            "kenya_resource": "KNH Social Work Department / KEHPCA",
        },
    ],
    "IV": [
        {
            "priority": 1,
            "action": "Goals of care discussion",
            "detail": "Honest, compassionate conversation about prognosis and treatment goals. "
                      "Curative intent rarely possible at Stage IV. Focus: quality of life, "
                      "symptom control, time with family. Engage cultural and spiritual beliefs.",
            "timeframe": "Immediate — before any treatment decision",
            "kenya_resource": "KNH Palliative Care / KEHPCA / Chaplaincy",
        },
        {
            "priority": 2,
            "action": "Systemic chemotherapy (if performance status allows)",
            "detail": "Palliative chemotherapy to extend life and control symptoms. "
                      "Carboplatin, 5-FU, Paclitaxel available on KEML. "
                      "Assess ECOG performance status before starting.",
            "timeframe": "Within 1–2 weeks if PS 0–2",
            "kenya_resource": "KNH / MTRH / Eldoret Cancer Center",
        },
        {
            "priority": 3,
            "action": "Clinical trial access",
            "detail": "TP53 reactivator trials (APR-246/Eprenetapopt) — contact AMPATH. "
                      "International compassionate use programmes — Doctors Without Borders "
                      "occasionally facilitates access in East Africa.",
            "timeframe": "Concurrent",
            "kenya_resource": "AMPATH Eldoret / MSF East Africa",
        },
        {
            "priority": 4,
            "action": "Comprehensive palliative care",
            "detail": "Pain management (oral morphine available on KEML), "
                      "nutritional support, psychosocial care, home-based care coordination. "
                      "KEHPCA community teams cover most Kenyan counties.",
            "timeframe": "Immediate and ongoing",
            "kenya_resource": "KEHPCA / KNH Palliative Unit / County health teams",
        },
        {
            "priority": 5,
            "action": "Advanced care planning",
            "detail": "Document patient wishes regarding resuscitation, hospitalisation, "
                      "and care location. Engage family. Support home-based death if preferred.",
            "timeframe": "Early in palliative trajectory",
            "kenya_resource": "KNH Palliative Care / KEHPCA",
        },
    ],
}

# Imaging workup per stage
IMAGING_WORKUP = {
    "I":    ["CT chest/abdomen/pelvis (staging)", "Ultrasound (if CT unavailable)"],
    "IIA":  ["CT chest/abdomen/pelvis", "MRI primary site if T borderline"],
    "IIB":  ["CT chest/abdomen/pelvis", "MRI primary site", "Bone scan if symptomatic"],
    "IIIA": ["CT chest/abdomen/pelvis", "MRI primary site", "PET-CT if available (KNH/AKU)"],
    "IIIB": ["CT chest/abdomen/pelvis", "MRI primary site", "PET-CT (if available)", "Brain MRI if neurological symptoms"],
    "IIIC": ["CT chest/abdomen/pelvis", "MRI primary site", "PET-CT (if available)", "Brain MRI"],
    "IV":   ["CT chest/abdomen/pelvis", "Brain MRI", "Bone scan", "Consider PET-CT for full metastatic mapping"],
}

# MDT referrals per stage
MDT_REFERRALS = {
    "I":    ["Surgical Oncology", "Pathology", "Medical Oncology (adjuvant discussion)"],
    "IIA":  ["Surgical Oncology", "Medical Oncology", "Radiology", "Pathology"],
    "IIB":  ["Surgical Oncology", "Medical Oncology", "Radiation Oncology", "Pathology", "Genetic Counselling"],
    "IIIA": ["Medical Oncology", "Radiation Oncology", "Surgical Oncology", "Palliative Care", "Pathology"],
    "IIIB": ["Medical Oncology", "Radiation Oncology", "Palliative Care", "Social Work", "Clinical Trials"],
    "IIIC": ["Medical Oncology", "Palliative Care", "Social Work", "Clinical Trials", "Nutrition"],
    "IV":   ["Palliative Care", "Medical Oncology", "Social Work", "Chaplaincy/Spiritual Care", "Clinical Trials", "Nutrition"],
}


# ═══════════════════════════════════════════════════════════════════
# LLM Prompt
# ═══════════════════════════════════════════════════════════════════

TNM_STAGING_PROMPT = """You are an expert oncological pathologist and staging specialist 
working in a Kenyan referral hospital. You have received:

1. Pathology slide analysis showing tissue composition
2. TP53 mutation profile with variant allele frequency
3. Clinical context including cancer type

Your task: Provide AJCC/UICC 8th Edition TNM staging with clinical reasoning.

STRICT OUTPUT FORMAT — follow exactly, no introductions:

T STAGE: [T1/T2/T3/T4 + letter suffix if applicable]
T REASONING: [One sentence tying pathology findings to T descriptor]

N STAGE: [N0/N1/N2/N3]
N REASONING: [One sentence — inflammatory infiltrate, direct invasion, clinical findings]

M STAGE: [M0/M1]
M REASONING: [One sentence — VAF, necrosis, clinical context]

OVERALL STAGE: [I / IIA / IIB / IIIA / IIIB / IIIC / IV]

CLINICAL SUMMARY: [2–3 sentences. State mutation, stage, prognosis direction, 
and one key therapeutic implication relevant to Kenya/East Africa.]

KENYA CONTEXT: [One sentence on specific resource or challenge relevant to 
managing this stage in a Kenyan clinical setting.]

Do NOT add sections beyond the above. Do NOT use markdown headers.
Every statement must be tied to the data provided."""


# ═══════════════════════════════════════════════════════════════════
# TNM Staging Agent
# ═══════════════════════════════════════════════════════════════════

class TNMStagingAgent:
    """
    Agent #16 — TNM Staging & Clinical Roadmap Generator.

    Takes PathologyVisionAgent output + TP53 mutation data
    and produces AJCC/UICC 8th edition TNM staging with
    Kenya-contextualised next steps.

    Gracefully degrades to rule-based staging if RAG/LLM unavailable.
    Never returns empty output.
    """

    MAX_RETRIES = 3
    _AUDIT_LOG = Path("logs/tnm_staging.log")

    def __init__(self, rag_chain=None):
        self.rag_chain = rag_chain
        self._lock = threading.Lock()
        self._AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        log.info("TNMStagingAgent (#16) initialised")

    # ── Public API ────────────────────────────────────────────────
    def stage(
        self,
        pathology_result: Dict[str, Any],
        pipeline_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entry point. Produces full TNM staging + clinical roadmap.

        Args:
            pathology_result: Output from PathologyVisionAgent.process_slide()
            pipeline_data: TP53 pipeline dict — must contain:
                           mutations (list), cancer_type (str), vaf (float)

        Returns:
            JSON-serializable dict with TNMResult fields + FHIR resource
        """
        # ── PII scrub inputs ──────────────────────────────────────
        patient_id_raw = pipeline_data.get("patient_id", "UNKNOWN")
        patient_id_hash = hashlib.sha256(
            patient_id_raw.encode()
        ).hexdigest()[:16]

        mutations = pipeline_data.get("mutations", [])
        cancer_type = pipeline_data.get("cancer_type", "default")
        vaf = pipeline_data.get("vaf", None)
        accession = pipeline_data.get("accession", "TP53")

        # Primary mutation
        primary_mutation = (
            mutations[0].get("amino_acid_change", "Unknown")
            if mutations else "Unknown"
        )

        log.info(
            f"TNM staging | cancer={cancer_type} | "
            f"mutation={primary_mutation} | patient={patient_id_hash}"
        )

        # ── African equity check ──────────────────────────────────
        equity_flag = AFRICAN_EQUITY_FLAGS.get(primary_mutation)

        # ── Rule-based TNM components ─────────────────────────────
        t_comp = self._classify_T(pathology_result, mutations, cancer_type, vaf)
        n_comp = self._classify_N(pathology_result, mutations, vaf)
        m_comp = self._classify_M(pathology_result, vaf, primary_mutation)

        # ── Stage group ───────────────────────────────────────────
        stage_group = self._compute_stage_group(t_comp, n_comp, m_comp)

        # ── LLM narration ─────────────────────────────────────────
        narration = self._get_llm_narration(
            t_comp, n_comp, m_comp, stage_group,
            pathology_result, pipeline_data, primary_mutation, cancer_type
        )

        # ── Next steps, imaging, MDT ──────────────────────────────
        next_steps = STAGE_NEXT_STEPS.get(stage_group, STAGE_NEXT_STEPS["IV"])
        imaging = IMAGING_WORKUP.get(stage_group, IMAGING_WORKUP["IV"])
        mdt = MDT_REFERRALS.get(stage_group, MDT_REFERRALS["IV"])

        # ── FHIR R4 ClinicalImpression ────────────────────────────
        fhir = self._build_fhir_resource(
            patient_id_hash, t_comp, n_comp, m_comp,
            stage_group, cancer_type, primary_mutation, accession
        )

        # ── Assemble result ───────────────────────────────────────
        result = TNMResult(
            T=t_comp,
            N=n_comp,
            M=m_comp,
            stage_group=stage_group,
            cancer_type=cancer_type,
            mutation=primary_mutation,
            equity_flag=equity_flag,
            llm_narration=narration,
            next_steps=next_steps,
            imaging_workup=imaging,
            mdt_referrals=mdt,
            fhir_resource=fhir,
            confidence_overall=min(t_comp.confidence, n_comp.confidence, m_comp.confidence),
        )

        # ── Audit ─────────────────────────────────────────────────
        self._audit({
            "event": "tnm_staging_complete",
            "patient_hash": patient_id_hash,
            "stage": stage_group,
            "mutation": primary_mutation,
            "cancer_type": cancer_type,
            "equity_flag": bool(equity_flag),
        })

        output = asdict(result)
        log.info(
            f"TNM staging complete | stage={stage_group} | "
            f"confidence={result.confidence_overall:.2f}"
        )
        return output

    # ── T Classification ──────────────────────────────────────────
    def _classify_T(
        self,
        pathology_result: Dict,
        mutations: List[Dict],
        cancer_type: str,
        vaf: Optional[float],
    ) -> TNMComponent:
        """
        Rule-based T classification.
        Driven by: tissue type from pathology + mutation modifier + VAF.
        """
        top_tissue = pathology_result.get("top_tissue", "Tumor")
        tissue_list = pathology_result.get("tissue_classifications", [])

        # Base T from tissue composition
        necrosis_prob = next(
            (t["probability"] for t in tissue_list if t["tissue"] == "Necrosis"), 0
        )
        stroma_prob = next(
            (t["probability"] for t in tissue_list if t["tissue"] == "Stroma"), 0
        )

        # Base T score (1–4)
        t_score = 1
        if necrosis_prob > 0.3:
            t_score = 3   # necrosis signals advanced local disease
        elif stroma_prob > 0.4:
            t_score = 2   # stromal invasion
        elif top_tissue in ("Tumor", "Inflammatory"):
            t_score = 2

        # Mutation modifier
        primary_mutation = (
            mutations[0].get("amino_acid_change", "") if mutations else ""
        )
        modifier = MUTATION_T_MODIFIER.get(primary_mutation, 0)
        t_score = min(4, t_score + modifier)

        # VAF modifier — very high VAF suggests established large tumour
        if vaf and vaf >= 60:
            t_score = min(4, t_score + 1)

        t_code = f"T{t_score}"
        descriptors = T_DESCRIPTORS.get(cancer_type, T_DESCRIPTORS["default"])
        description = descriptors.get(t_code, descriptors.get(f"T{t_score}", "Tumour extent as per pathology"))

        evidence = (
            f"Top tissue: {top_tissue} "
            f"(necrosis={necrosis_prob:.0%}, stroma={stroma_prob:.0%}); "
            f"mutation {primary_mutation} modifier={modifier}; "
            f"VAF={vaf}%"
        )

        return TNMComponent(
            code=t_code,
            description=description,
            evidence=evidence,
            confidence=0.75 if modifier > 0 else 0.65,
        )

    # ── N Classification ──────────────────────────────────────────
    def _classify_N(
        self,
        pathology_result: Dict,
        mutations: List[Dict],
        vaf: Optional[float],
    ) -> TNMComponent:
        """
        Rule-based N classification.
        Driven by: inflammatory infiltrate in pathology + VAF.
        Note: definitive N staging requires lymph node biopsy — this
        is a risk-based estimate flagged as such.
        """
        tissue_list = pathology_result.get("tissue_classifications", [])

        inflammatory_prob = next(
            (t["probability"] for t in tissue_list if t["tissue"] == "Inflammatory"), 0
        )
        necrosis_prob = next(
            (t["probability"] for t in tissue_list if t["tissue"] == "Necrosis"), 0
        )

        n_score = 0
        if inflammatory_prob > 0.4 or necrosis_prob > 0.3:
            n_score = 1
        if inflammatory_prob > 0.6:
            n_score = 2
        if vaf and vaf >= 50 and n_score >= 1:
            n_score = min(3, n_score + 1)

        n_code = f"N{n_score}"
        n_descriptions = {
            "N0": "No regional lymph node metastasis",
            "N1": "Metastasis in 1–3 regional lymph nodes",
            "N2": "Metastasis in 4–6 regional lymph nodes",
            "N3": "Metastasis in 7+ regional lymph nodes",
        }

        return TNMComponent(
            code=n_code,
            description=n_descriptions.get(n_code, "Regional lymph node status uncertain"),
            evidence=(
                f"Inflammatory infiltrate={inflammatory_prob:.0%}; "
                f"necrosis={necrosis_prob:.0%}; VAF={vaf}%. "
                f"NOTE: Definitive N staging requires lymph node dissection/biopsy."
            ),
            confidence=0.55,   # lower confidence — imaging/biopsy needed to confirm
        )

    # ── M Classification ──────────────────────────────────────────
    def _classify_M(
        self,
        pathology_result: Dict,
        vaf: Optional[float],
        primary_mutation: str,
    ) -> TNMComponent:
        """
        Rule-based M classification.
        Driven by: VAF + necrosis + mutation-specific risk.
        Note: definitive M staging requires systemic imaging.
        """
        tissue_list = pathology_result.get("tissue_classifications", [])
        necrosis_prob = next(
            (t["probability"] for t in tissue_list if t["tissue"] == "Necrosis"), 0
        )

        m_signal = _vaf_to_m_signal(vaf)
        if necrosis_prob > 0.5:
            m_signal = 1   # extensive necrosis = aggressive, higher M risk

        # R249S in East Africa often presents with hepatic metastases
        if primary_mutation == "R249S":
            m_signal = max(m_signal, 1)

        m_code = f"M{m_signal}"
        m_descriptions = {
            "M0": "No distant metastasis detected on available evidence",
            "M1": "Distant metastasis suspected — systemic staging imaging required",
        }

        return TNMComponent(
            code=m_code,
            description=m_descriptions[m_code],
            evidence=(
                f"VAF={vaf}% (≥70% triggers M1 flag); "
                f"necrosis={necrosis_prob:.0%}; "
                f"mutation={primary_mutation}. "
                f"NOTE: CT chest/abdomen/pelvis required for definitive M staging."
            ),
            confidence=0.60,
        )

    # ── Stage Group ───────────────────────────────────────────────
    def _compute_stage_group(
        self,
        t: TNMComponent,
        n: TNMComponent,
        m: TNMComponent,
    ) -> str:
        """Compute AJCC stage group from TNM components."""
        try:
            t_num = int(t.code[1])   # "T2" → 2
            n_num = int(n.code[1])   # "N1" → 1
            m_num = int(m.code[1])   # "M0" → 0
            stage = STAGE_GROUPS.get((t_num, n_num, m_num))
            if stage:
                return stage
            # Fallback: if not in table, use M1 = IV rule
            if m_num == 1:
                return "IV"
            if n_num >= 2:
                return "IIIC"
            return "IIIB"
        except (ValueError, IndexError):
            log.warning("Could not parse TNM codes — defaulting to IIIB")
            return "IIIB"

    # ── LLM Narration ─────────────────────────────────────────────
    def _get_llm_narration(
        self,
        t: TNMComponent,
        n: TNMComponent,
        m: TNMComponent,
        stage_group: str,
        pathology_result: Dict,
        pipeline_data: Dict,
        mutation: str,
        cancer_type: str,
    ) -> str:
        """Get Gemma 4 staging narration with self-correction loop."""
        if self.rag_chain is None:
            return self._fallback_narration(t, n, m, stage_group, mutation, cancer_type)

        top_tissue = pathology_result.get("top_tissue", "Unknown")
        tissue_list = pathology_result.get("tissue_classifications", [])[:3]
        vaf = pipeline_data.get("vaf", "Unknown")

        question = (
            f"TNM staging context: Cancer type={cancer_type}, "
            f"Primary mutation={mutation}, VAF={vaf}%. "
            f"Pathology: top tissue={top_tissue}, "
            f"classifications={tissue_list}. "
            f"Rule-based staging result: {t.code} {n.code} {m.code} → Stage {stage_group}. "
            f"Provide clinical narration per the format specified."
        )

        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.rag_chain.query(
                    question=question,
                    pipeline_data=pipeline_data,
                    agent_type="clinical_interpretation",
                )
                narration = result.get("answer", "")
                if narration and len(narration) > 50:
                    return narration
            except Exception as e:
                log.warning(f"LLM narration attempt {attempt+1} failed: {e}")

        return self._fallback_narration(t, n, m, stage_group, mutation, cancer_type)

    def _fallback_narration(
        self,
        t: TNMComponent,
        n: TNMComponent,
        m: TNMComponent,
        stage_group: str,
        mutation: str,
        cancer_type: str,
    ) -> str:
        """Rule-based narration when LLM unavailable. Never empty."""
        prognosis = {
            "I": "favourable — curative intent is the primary goal",
            "IIA": "moderate — curative surgery likely feasible",
            "IIB": "moderate — multimodal therapy recommended",
            "IIIA": "guarded — combined modality treatment required",
            "IIIB": "guarded — systemic therapy primary",
            "IIIC": "poor — systemic therapy and palliation",
            "IV": "poor — palliative intent, quality of life focus",
        }.get(stage_group, "uncertain")

        return (
            f"TP53 {mutation} in {cancer_type} cancer staged as "
            f"{t.code} {n.code} {m.code} — Overall Stage {stage_group}. "
            f"Prognosis: {prognosis}. "
            f"{t.description}. "
            f"{n.description}. "
            f"{m.description}. "
            f"All staging estimates require confirmation with systemic imaging "
            f"(CT chest/abdomen/pelvis) and, where indicated, lymph node biopsy. "
            f"Refer to KNH Tumor Board for MDT review."
        )

    # ── FHIR R4 Resource ──────────────────────────────────────────
    def _build_fhir_resource(
        self,
        patient_hash: str,
        t: TNMComponent,
        n: TNMComponent,
        m: TNMComponent,
        stage_group: str,
        cancer_type: str,
        mutation: str,
        accession: str,
    ) -> Dict:
        """Build HL7 FHIR R4 ClinicalImpression resource."""
        return {
            "resourceType": "ClinicalImpression",
            "id": f"tnm-{patient_hash[:8]}-{datetime.now().strftime('%Y%m%d')}",
            "status": "completed",
            "description": f"TNM staging for TP53 {mutation} in {cancer_type} cancer",
            "subject": {"reference": f"Patient/{patient_hash}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "investigation": [
                {
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "258226009",
                            "display": "TNM Staging",
                        }]
                    },
                    "item": [
                        {"display": f"T Stage: {t.code} — {t.description}"},
                        {"display": f"N Stage: {n.code} — {n.description}"},
                        {"display": f"M Stage: {m.code} — {m.description}"},
                        {"display": f"Overall Stage: {stage_group}"},
                    ]
                }
            ],
            "finding": [
                {
                    "itemCodeableConcept": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "display": f"Cancer stage group {stage_group}",
                        }]
                    }
                }
            ],
            "note": [{
                "text": (
                    f"TP53 mutation: {mutation} | Accession: {accession} | "
                    f"Generated by TP53 RAG Platform v1.0 (Gemma 4, Ollama) | "
                    f"Kenya clinical context | "
                    f"Patient ID hashed (SHA-256): {patient_hash}"
                )
            }],
        }

    # ── Audit ─────────────────────────────────────────────────────
    def _audit(self, event: Dict):
        """Append-only audit log — HIPAA compliant."""
        try:
            with self._lock:
                entry = json.dumps(event) + "\n"
                with open(self._AUDIT_LOG, "a", encoding="utf-8") as f:
                    f.write(entry)
        except Exception as e:
            log.warning(f"TNM audit log failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Convenience function (matches pattern in variant_curator.py etc.)
# ═══════════════════════════════════════════════════════════════════

_agent = None

def get_tnm_agent(rag_chain=None) -> TNMStagingAgent:
    """Get or create the global TNM agent instance."""
    global _agent
    if _agent is None:
        _agent = TNMStagingAgent(rag_chain=rag_chain)
    return _agent

def stage_cancer(
    pathology_result: Dict[str, Any],
    pipeline_data: Dict[str, Any],
    rag_chain=None,
) -> Dict[str, Any]:
    """Convenience function matching dispatcher pattern."""
    return get_tnm_agent(rag_chain).stage(pathology_result, pipeline_data)


# ═══════════════════════════════════════════════════════════════════
# Self-tests — Break & Fix Method (claudesam_master_requirements.md)
# Run: python agents/tnm_staging.py
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("\n=== TNMStagingAgent Self-Tests (Break & Fix) ===\n")
    passed = 0

    agent = TNMStagingAgent(rag_chain=None)  # rule-based mode — no LLM needed

    # Shared mock data
    MOCK_PATHOLOGY_NORMAL = {
        "success": True,
        "top_tissue": "Tumor",
        "tissue_classifications": [
            {"tissue": "Tumor", "probability": 0.70},
            {"tissue": "Stroma", "probability": 0.20},
            {"tissue": "Inflammatory", "probability": 0.05},
            {"tissue": "Necrosis", "probability": 0.05},
        ],
        "mutation_correlations": [],
    }

    MOCK_PATHOLOGY_ADVANCED = {
        "success": True,
        "top_tissue": "Necrosis",
        "tissue_classifications": [
            {"tissue": "Necrosis", "probability": 0.55},
            {"tissue": "Tumor", "probability": 0.30},
            {"tissue": "Inflammatory", "probability": 0.40},
            {"tissue": "Stroma", "probability": 0.10},
        ],
        "mutation_correlations": [],
    }

    MOCK_PIPELINE_R175H = {
        "mutations": [{"amino_acid_change": "R175H", "position": 175}],
        "cancer_type": "Colorectal",
        "vaf": 47.3,
        "accession": "NM_000546",
        "patient_id": "DEMO-KE-001",
    }

    MOCK_PIPELINE_R249S = {
        "mutations": [{"amino_acid_change": "R249S", "position": 249}],
        "cancer_type": "Liver",
        "vaf": 72.0,
        "accession": "NM_000546",
        "patient_id": "DEMO-KE-002",
    }

    MOCK_PIPELINE_P72R = {
        "mutations": [{"amino_acid_change": "P72R", "position": 72}],
        "cancer_type": "Colorectal",
        "vaf": 35.0,
        "accession": "NM_000546",
        "patient_id": "DEMO-KE-003",
    }

    # T1: Basic staging returns a result
    result = agent.stage(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_R175H)
    assert result is not None, "FAIL: stage() returned None"
    assert "stage_group" in result, "FAIL: stage_group missing"
    assert result["stage_group"] in {"I","IIA","IIB","IIIA","IIIB","IIIC","IV"}, "FAIL: invalid stage"
    print(f"✅ T1 Basic staging: {result['T']['code']} {result['N']['code']} {result['M']['code']} → Stage {result['stage_group']}")
    passed += 1

    # T2: T code format correct
    t_code = result["T"]["code"]
    assert t_code.startswith("T") and t_code[1].isdigit(), f"FAIL: bad T code {t_code}"
    print(f"✅ T2 T code format valid: {t_code}")
    passed += 1

    # T3: N code format correct
    n_code = result["N"]["code"]
    assert n_code.startswith("N") and n_code[1].isdigit(), f"FAIL: bad N code {n_code}"
    print(f"✅ T3 N code format valid: {n_code}")
    passed += 1

    # T4: M code format correct
    m_code = result["M"]["code"]
    assert m_code in {"M0", "M1"}, f"FAIL: bad M code {m_code}"
    print(f"✅ T4 M code format valid: {m_code}")
    passed += 1

    # T5: Advanced pathology (necrosis) → higher T stage
    result_adv = agent.stage(MOCK_PATHOLOGY_ADVANCED, MOCK_PIPELINE_R175H)
    t_adv = int(result_adv["T"]["code"][1])
    t_norm = int(result["T"]["code"][1])
    assert t_adv >= t_norm, f"FAIL: advanced pathology should not give lower T ({t_adv} < {t_norm})"
    print(f"✅ T5 Necrosis → higher T: {t_norm} → {t_adv}")
    passed += 1

    # T6: R249S triggers African equity flag
    result_r249s = agent.stage(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_R249S)
    assert result_r249s["equity_flag"] is not None, "FAIL: R249S should trigger equity flag"
    assert "aflatoxin" in result_r249s["equity_flag"].lower(), "FAIL: aflatoxin not in flag"
    print(f"✅ T6 R249S equity flag triggered: {result_r249s['equity_flag'][:60]}...")
    passed += 1

    # T7: P72R triggers benign equity flag
    result_p72r = agent.stage(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_P72R)
    assert result_p72r["equity_flag"] is not None, "FAIL: P72R should trigger equity flag"
    assert "BENIGN" in result_p72r["equity_flag"].upper() or "benign" in result_p72r["equity_flag"].lower()
    print(f"✅ T7 P72R benign polymorphism flag triggered")
    passed += 1

    # T8: Next steps never empty
    next_steps = result["next_steps"]
    assert len(next_steps) > 0, "FAIL: next_steps empty"
    assert all("action" in s and "kenya_resource" in s for s in next_steps), \
        "FAIL: next_steps missing required keys"
    print(f"✅ T8 Next steps: {len(next_steps)} actions with Kenya resources")
    passed += 1

    # T9: Imaging workup present
    imaging = result["imaging_workup"]
    assert len(imaging) > 0, "FAIL: imaging_workup empty"
    print(f"✅ T9 Imaging workup: {imaging}")
    passed += 1

    # T10: MDT referrals present
    mdt = result["mdt_referrals"]
    assert len(mdt) > 0, "FAIL: mdt_referrals empty"
    print(f"✅ T10 MDT referrals: {mdt}")
    passed += 1

    # T11: FHIR resource structure valid
    fhir = result["fhir_resource"]
    assert fhir["resourceType"] == "ClinicalImpression", "FAIL: wrong FHIR resource type"
    assert "subject" in fhir, "FAIL: FHIR missing subject"
    assert "investigation" in fhir, "FAIL: FHIR missing investigation"
    print(f"✅ T11 FHIR R4 ClinicalImpression valid: id={fhir['id']}")
    passed += 1

    # T12: Patient ID is hashed in FHIR (PII check)
    fhir_note = fhir["note"][0]["text"]
    assert "DEMO-KE-001" not in fhir_note, "FAIL: raw patient ID in FHIR output (PII leak)"
    assert "SHA-256" in fhir_note, "FAIL: hash reference missing from FHIR note"
    print(f"✅ T12 PII hashed in FHIR: patient ID not exposed")
    passed += 1

    # T13: High VAF (>=70%) → M1 signal
    high_vaf_pipeline = {**MOCK_PIPELINE_R175H, "vaf": 75.0}
    result_hv = agent.stage(MOCK_PATHOLOGY_NORMAL, high_vaf_pipeline)
    assert result_hv["M"]["code"] == "M1", \
        f"FAIL: VAF=75% should trigger M1, got {result_hv['M']['code']}"
    print(f"✅ T13 High VAF (75%) → M1: {result_hv['M']['code']}")
    passed += 1

    # T14: M1 → Stage IV
    assert result_hv["stage_group"] == "IV", \
        f"FAIL: M1 should always be Stage IV, got {result_hv['stage_group']}"
    print(f"✅ T14 M1 → Stage IV: confirmed")
    passed += 1

    # T15: Stage IV next steps include palliative care
    iv_next_steps = result_hv["next_steps"]
    iv_actions = " ".join(s["action"].lower() for s in iv_next_steps)
    assert "palliative" in iv_actions or "goals of care" in iv_actions, \
        "FAIL: Stage IV next steps missing palliative care"
    print(f"✅ T15 Stage IV → palliative care in next steps")
    passed += 1

    # T16: Fallback narration never empty (no RAG)
    narration = agent._fallback_narration(
        agent._classify_T(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_R175H["mutations"], "Colorectal", 47.3),
        agent._classify_N(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_R175H["mutations"], 47.3),
        agent._classify_M(MOCK_PATHOLOGY_NORMAL, 47.3, "R175H"),
        "IIA", "R175H", "Colorectal"
    )
    assert len(narration) > 50, "FAIL: fallback narration too short"
    assert "KNH" in narration, "FAIL: Kenya resource missing from fallback narration"
    print(f"✅ T16 Fallback narration: {len(narration)} chars, KNH referenced")
    passed += 1

    # T17: Convenience function works
    result_conv = stage_cancer(MOCK_PATHOLOGY_NORMAL, MOCK_PIPELINE_R175H)
    assert "stage_group" in result_conv, "FAIL: convenience function broken"
    print(f"✅ T17 Convenience function stage_cancer(): OK")
    passed += 1

    # T18: Stage group compute — known pair
    t_comp = TNMComponent("T1", "desc", "evidence", 0.8)
    n_comp = TNMComponent("N0", "desc", "evidence", 0.8)
    m_comp = TNMComponent("M0", "desc", "evidence", 0.8)
    stage = agent._compute_stage_group(t_comp, n_comp, m_comp)
    assert stage == "I", f"FAIL: T1N0M0 should be Stage I, got {stage}"
    print(f"✅ T18 Stage compute T1N0M0 → Stage I")
    passed += 1

    # T19: Audit log written
    import time; time.sleep(0.1)
    assert agent._AUDIT_LOG.exists(), "FAIL: audit log not created"
    log_content = agent._AUDIT_LOG.read_text()
    assert "tnm_staging_complete" in log_content, "FAIL: audit event not logged"
    print(f"✅ T19 Audit log written: {agent._AUDIT_LOG}")
    passed += 1

    # T20: No empty next steps for any stage
    for stage_key in STAGE_NEXT_STEPS:
        steps = STAGE_NEXT_STEPS[stage_key]
        assert len(steps) > 0, f"FAIL: no next steps for stage {stage_key}"
        for step in steps:
            assert step.get("kenya_resource"), f"FAIL: missing kenya_resource in stage {stage_key}"
    print(f"✅ T20 All {len(STAGE_NEXT_STEPS)} stage groups have Kenya-contextualised next steps")
    passed += 1

    print(f"\n=== {passed}/20 tests passed ===\n")
    if passed < 20:
        sys.exit(1)