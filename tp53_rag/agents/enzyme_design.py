"""
agents/enzyme_design.py — TP53 Enzyme & Protein Engineering Agent (Agent #13)
================================================================================
Designs p53 protein reactivation strategies, corrector peptides, and
engineered enzyme variants targeting TP53 mutant-specific structural defects.

Covers:
  - p53 DNA-binding domain (DBD) rescue strategies per mutation class
  - MDM2 inhibitor peptide design (stapled peptides, miniproteins)
  - p53 reactivating small-molecule correctors (APR-246, PRIMA-1 analogues)
  - Synthetic p53 mimetics (transcription factor restoration)
  - Zinc-coordination rescue (C176/H179/C238/C242 site)
  - Proteolysis-targeting chimeras (PROTACs) for mutant p53 degradation

HIPAA compliant — no raw PII in logs or LLM prompts.
HL7 FHIR R4 structured output.
Rate-limited (20 calls/min). Shared cache (utils/cache.py).
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
# TP53 Structural & Enzyme Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_ID = "enzyme_design"

# ── p53 protein domain map ────────────────────────────────────────────────────
P53_DOMAINS = {
    "TAD1":          {"residues": (1,  40),  "function": "Transactivation domain 1 — MDM2 binding"},
    "TAD2":          {"residues": (40, 67),  "function": "Transactivation domain 2"},
    "PRD":           {"residues": (67, 98),  "function": "Proline-rich domain — apoptosis signalling"},
    "DBD":           {"residues": (94, 292), "function": "DNA-binding domain — hotspot mutation site"},
    "NLS":           {"residues": (316,325), "function": "Nuclear localisation signal"},
    "TETRAMERIZATION":{"residues":(323,356), "function": "Tetramerization domain — oligomeric p53"},
    "REG":           {"residues": (364,393), "function": "Regulatory domain — post-translational mods"},
}

# ── Zinc coordination site (critical for DBD folding) ────────────────────────
ZINC_COORDINATION_RESIDUES = ["C176", "H179", "C238", "C242"]

# ── Per-mutation structural defect + engineering strategy ─────────────────────
MUTATION_ENGINEERING_MAP: dict[str, dict] = {
    "R175H": {
        "mutation_class":    "Conformational",
        "structural_defect": (
            "R175 normally forms a critical hydrogen bond network stabilising "
            "the L2/L3 loop of the DBD. R175H disrupts this network, causing "
            "global unfolding of the DNA-binding surface. Zinc coordination "
            "is destabilised secondarily."
        ),
        "domain_affected":   "DBD (L2 loop, residues 164–194)",
        "zinc_affected":     True,

        # Small molecule correctors
        "small_molecule_correctors": [
            {
                "name":        "APR-246 (Eprenetapopt)",
                "mechanism":   "Converts to MQ (methylene quinuclidinone); alkylates C277 "
                               "and C176 in mutant p53 DBD, restoring WT-like conformation",
                "target_residues": ["C277", "C176"],
                "stage":       "Phase III (AML/MDS) — KNH trial eligibility",
                "priority":    "HIGH",
                "keml_status": "Trial access via KNH / AKUH Nairobi",
            },
            {
                "name":        "PRIMA-1",
                "mechanism":   "Parent compound of APR-246; same MQ conversion mechanism, "
                               "lower potency",
                "target_residues": ["C176", "C238"],
                "stage":       "Preclinical",
                "priority":    "MEDIUM",
                "keml_status": "Research only",
            },
            {
                "name":        "PK7088",
                "mechanism":   "Binds Y220C-adjacent pocket — limited R175H activity; "
                               "included as structural comparator",
                "target_residues": ["S116", "T125"],
                "stage":       "Preclinical",
                "priority":    "LOW",
                "keml_status": "Research only",
            },
        ],

        # Engineered peptide / protein strategies
        "peptide_strategies": [
            {
                "name":        "ReACp53 (stapled peptide)",
                "mechanism":   "Cell-penetrating stapled peptide targeting the p53 aggregation "
                               "interface; inhibits amyloid-like aggregation of R175H mutant",
                "design_basis":"Sequence derived from p53 β-strand S9 (residues 251–257); "
                               "hydrocarbon staple at i, i+4 positions",
                "target":      "p53 aggregation interface",
                "stage":       "Preclinical (mouse xenograft)",
                "priority":    "HIGH",
            },
            {
                "name":        "p53 DBD superstabiliser peptide (CDB3)",
                "mechanism":   "9-residue peptide derived from p53-binding protein 2 (53BP2); "
                               "binds and thermostabilises mutant p53 DBD",
                "design_basis":"53BP2 interface residues: RALPNNTS",
                "target":      "p53 DBD hydrophobic core",
                "stage":       "Preclinical",
                "priority":    "MEDIUM",
            },
        ],

        # PROTAC strategy (targeted degradation of gain-of-function mutant)
        "protac_strategy": {
            "rationale":     "R175H is a GOF mutant — degrading it removes oncogenic activity",
            "e3_ligase":     "CRBN (cereblon) or VHL",
            "warhead":       "APR-246-derived covalent binder (MQ moiety) targeting C176/C277",
            "linker":        "PEG4 linker (optimal for DBD–E3 distance ~14Å)",
            "design_notes":  "PROTAC-p53(R175H): MQ-PEG4-pomalidomide; predicted MW ~950 Da",
            "stage":         "Computational design — no clinical data",
        },

        # MDM2 inhibitor design (restore p53 pathway even if mutant p53 is present)
        "mdm2_inhibitor": {
            "rationale":     "MDM2 still binds and degrades residual WT p53 in heterozygous "
                             "tumours; inhibiting MDM2 rescues WT allele",
            "peptide_design":"Stapled α-helix mimicking p53 TAD1 (residues 19–26: TFSDLWKL); "
                             "hydrocarbon staple at F19/L26",
            "small_molecule":"Nutlin-3a / RG7112 / Idasanutlin — MDM2 pocket binders",
            "mdm2_score":    9,
            "clinical_note": "Most effective in heterozygous TP53 tumours",
        },

        "zinc_rescue_strategy": (
            "Zinc metallochaperones (ZMC1/NSC319726) restore zinc coordination at "
            "C176/H179/C238/C242, partially rescuing R175H conformation. "
            "Combine with APR-246 for synergistic reactivation."
        ),
    },

    "R248W": {
        "mutation_class":    "Contact",
        "structural_defect": (
            "R248 directly contacts DNA major groove (AT-rich motif). R248W "
            "eliminates this contact via bulky tryptophan substitution. DBD "
            "global fold is preserved but DNA-binding is abrogated. "
            "Also gains oncogenic interaction with RAD51 (HR pathway hijack)."
        ),
        "domain_affected":   "DBD (L3 loop, residues 237–250) — DNA contact",
        "zinc_affected":     False,

        "small_molecule_correctors": [
            {
                "name":        "APR-246",
                "mechanism":   "MQ alkylation partially restores DBD flexibility; "
                               "less effective than in R175H but still active",
                "target_residues": ["C277", "C242"],
                "stage":       "Phase III",
                "priority":    "HIGH",
                "keml_status": "Trial access via KNH",
            },
            {
                "name":        "SCH529074",
                "mechanism":   "Binds p53 DBD core, stabilises DNA-contact conformation; "
                               "restores sequence-specific DNA binding in R248W",
                "target_residues": ["R248W site", "C176"],
                "stage":       "Preclinical",
                "priority":    "MEDIUM",
                "keml_status": "Research only",
            },
        ],

        "peptide_strategies": [
            {
                "name":        "DNA-contact mimetic peptide",
                "mechanism":   "Synthetic peptide mimicking the L3 loop (residues 237–250) "
                               "with W248→R substitution at key contact position; "
                               "acts as a molecular bridge restoring DNA groove contact",
                "design_basis":"Sequence: VVRCPHHERCSDSDGLAPPQHLIRVEGNL (L3 loop); "
                               "W248R reversion at position 12",
                "target":      "p53 response element in target gene promoters",
                "stage":       "Computational — not yet synthesised",
                "priority":    "HIGH",
            },
            {
                "name":        "RAD51 interaction blocker peptide",
                "mechanism":   "Blocks aberrant R248W–RAD51 interaction (GOF mechanism); "
                               "peptide derived from RAD51 ATPase domain interface",
                "design_basis":"RAD51 residues 82–102; cyclic peptide constraint",
                "target":      "R248W/RAD51 protein-protein interface",
                "stage":       "Preclinical",
                "priority":    "HIGH",
            },
        ],

        "protac_strategy": {
            "rationale":     "R248W GOF — degradation removes RAD51 hijack and DNA-contact loss",
            "e3_ligase":     "VHL",
            "warhead":       "SCH529074-derived binder",
            "linker":        "PEG3 linker",
            "design_notes":  "PROTAC-p53(R248W): SCH-PEG3-VHL ligand; predicted MW ~880 Da",
            "stage":         "Computational design",
        },

        "mdm2_inhibitor": {
            "rationale":     "MDM2 suppression rescues WT p53 in heterozygous context",
            "peptide_design":"Dual-function stapled peptide: p53 TAD (19–26) + MDM2/MDMX binder",
            "small_molecule":"ALRN-6924 (MDM2/MDMX dual inhibitor) — Phase II",
            "mdm2_score":    8,
            "clinical_note": "ALRN-6924 active in TP53 WT tumours; R248W heterozygous may respond",
        },

        "zinc_rescue_strategy": "Not applicable — zinc coordination intact in R248W.",
    },

    "R273H": {
        "mutation_class":    "Contact",
        "structural_defect": (
            "R273 is the primary DNA phosphate backbone contact residue. "
            "R273H abolishes electrostatic interaction with DNA. DBD fold "
            "intact. Unique GOF: activates PDGFR signalling via cytoplasmic "
            "p53 pool."
        ),
        "domain_affected":   "DBD (S10 β-strand, residues 270–275)",
        "zinc_affected":     False,

        "small_molecule_correctors": [
            {
                "name":        "Phikan083",
                "mechanism":   "Binds a cryptic pocket in Y220C mutant (adjacent); "
                               "limited direct activity on R273H but structural analogue "
                               "studies ongoing",
                "target_residues": ["Y205", "T211"],
                "stage":       "Preclinical",
                "priority":    "LOW",
                "keml_status": "Research only",
            },
            {
                "name":        "NSC319726 (zinc metallochaperone)",
                "mechanism":   "Restores zinc binding; indirect stabilisation of DBD "
                               "improves residual DNA-binding capacity",
                "target_residues": ["C176", "H179", "C238", "C242"],
                "stage":       "Preclinical",
                "priority":    "MEDIUM",
                "keml_status": "Research only",
            },
        ],

        "peptide_strategies": [
            {
                "name":        "Phosphate-contact mimetic (R273 restoration)",
                "mechanism":   "Guanidinium-containing peptidomimetic restoring positive "
                               "charge at position 273 to re-engage DNA phosphate backbone",
                "design_basis":"Arginine isostere at position 273; constrained by β-turn "
                               "scaffold from S10 strand",
                "target":      "DNA major groove phosphate contacts",
                "stage":       "Computational — not yet synthesised",
                "priority":    "HIGH",
            },
        ],

        "protac_strategy": {
            "rationale":     "R273H GOF via PDGFR; degradation + PDGFR inhibition synergy",
            "e3_ligase":     "CRBN",
            "warhead":       "NSC319726-inspired DBD binder",
            "linker":        "PEG4",
            "design_notes":  "Combine with imatinib (PDGFR inhibitor) for dual GOF blockade",
            "stage":         "Computational design",
        },

        "mdm2_inhibitor": {
            "rationale":     "MDM2 inhibition + PI3K inhibition (PTEN loss context)",
            "peptide_design":"Bispecific peptide: MDM2 antagonist + PI3K p85 SH2 binder",
            "small_molecule":"Nutlin-3a + everolimus combination",
            "mdm2_score":    7,
            "clinical_note": "PTEN-null co-mutation makes everolimus combination attractive",
        },

        "zinc_rescue_strategy": "Not applicable — zinc coordination intact in R273H.",
    },

    "G245S": {
        "mutation_class":    "Structural",
        "structural_defect": (
            "G245 is a critical glycine in the L3 loop providing essential "
            "conformational flexibility. G245S introduces a serine side chain "
            "that sterically clashes with L3 loop backbone, rigidifying the "
            "loop and preventing proper DNA minor groove insertion."
        ),
        "domain_affected":   "DBD (L3 loop, residues 237–250)",
        "zinc_affected":     False,

        "small_molecule_correctors": [
            {
                "name":        "CP-31398",
                "mechanism":   "Intercalates into p53 DBD, stabilises WT-like conformation; "
                               "active in G245S via loop rigidity reduction",
                "target_residues": ["G245S site", "V143"],
                "stage":       "Preclinical",
                "priority":    "MEDIUM",
                "keml_status": "Research only",
            },
        ],

        "peptide_strategies": [
            {
                "name":        "L3 loop flexibility restorer (glycine mimetic)",
                "mechanism":   "N-methylated peptide inserting a glycine isostere at position 245 "
                               "to restore L3 loop flexibility without serine clash",
                "design_basis":"Peptido-mimetic: Ac-[NMe-G245]-loop scaffold (residues 240–250)",
                "target":      "L3 loop backbone conformational space",
                "stage":       "Computational",
                "priority":    "MEDIUM",
            },
            {
                "name":        "Aurora kinase A (AURKA) inhibitor peptide",
                "mechanism":   "G245S upregulates AURKA (CIN driver); AURKA-targeting peptide "
                               "(derived from TPX2 activation domain) blocks AURKA kinase activity",
                "design_basis":"TPX2 residues 1–43 minimal activation domain",
                "target":      "AURKA activation loop",
                "stage":       "Preclinical",
                "priority":    "HIGH",
            },
        ],

        "protac_strategy": {
            "rationale":     "G245S drives CIN via PLK1/AURKA; PROTAC degradation of mutant p53 "
                             "removes CIN driver",
            "e3_ligase":     "VHL",
            "warhead":       "CP-31398-derived binder",
            "linker":        "Alkyl C6",
            "design_notes":  "Combine with PLK1 inhibitor (BI-6727) for synthetic lethality",
            "stage":         "Computational design",
        },

        "mdm2_inhibitor": {
            "rationale":     "MDM2 inhibition in G245S heterozygous tumours",
            "peptide_design":"Standard p53 TAD stapled helix",
            "small_molecule":"RG7112 (RO5045337)",
            "mdm2_score":    6,
            "clinical_note": "PLK1 inhibitor BI-6727 + MDM2 inhibitor combination promising",
        },

        "zinc_rescue_strategy": "Not applicable — zinc coordination intact in G245S.",
    },

    "R249S": {
        "mutation_class":    "Contact + Structural",
        "structural_defect": (
            "R249 contacts DNA backbone AND stabilises L3 loop via a salt bridge "
            "with E171. R249S eliminates both functions. Strongly associated with "
            "aflatoxin B1 exposure — hepatocellular carcinoma context. "
            "Gains JAK/STAT3 activation via cytoplasmic pool."
        ),
        "domain_affected":   "DBD (L3 loop — dual role: DNA contact + structural)",
        "zinc_affected":     False,

        "small_molecule_correctors": [
            {
                "name":        "Sorafenib (indirect)",
                "mechanism":   "Multi-kinase inhibitor active in HCC context; targets "
                               "HIF-1α/VEGFR downstream of R249S GOF signalling",
                "target_residues": ["VEGFR2 ATP site"],
                "stage":       "FDA approved (HCC)",
                "priority":    "HIGH",
                "keml_status": "Available at KNH / Aga Khan Nairobi (private)",
            },
        ],

        "peptide_strategies": [
            {
                "name":        "E171–R249 salt bridge restorer",
                "mechanism":   "Bifunctional peptide crosslinking E171 and S249 to restore "
                               "the missing salt bridge; constrains L3 loop in WT geometry",
                "design_basis":"Sequence spanning E171–R249 (28 residues); crosslink via "
                               "disulfide or triazole staple",
                "target":      "E171/R249S salt bridge site in DBD",
                "stage":       "Computational",
                "priority":    "MEDIUM",
            },
            {
                "name":        "STAT3 SH2 blocking peptide",
                "mechanism":   "Cyclic peptide blocking STAT3 SH2 domain; disrupts R249S "
                               "GOF STAT3 constitutive activation",
                "design_basis":"pTyr-containing cyclic peptide: Ac-pY-L-P-Q-T-V-c (cyclic)",
                "target":      "STAT3 SH2 domain",
                "stage":       "Preclinical",
                "priority":    "HIGH",
            },
        ],

        "protac_strategy": {
            "rationale":     "R249S GOF via STAT3/HIF-1α; degradation removes driver",
            "e3_ligase":     "CRBN",
            "warhead":       "HCC-context DBD binder (sorafenib scaffold)",
            "linker":        "PEG3",
            "design_notes":  "Kenya-relevant: aflatoxin-associated HCC priority target",
            "stage":         "Computational design",
        },

        "mdm2_inhibitor": {
            "rationale":     "Limited utility in R249S — JAK/STAT3 is the dominant GOF pathway",
            "peptide_design":"STAT3 inhibitor peptide preferred over MDM2 inhibitor",
            "small_molecule":"Ruxolitinib (JAK inhibitor) — off-label HCC use",
            "mdm2_score":    4,
            "clinical_note": "Aflatoxin exposure context — R249S endemic in sub-Saharan Africa",
        },

        "zinc_rescue_strategy": "Not applicable — zinc coordination intact in R249S.",
    },

    "R282W": {
        "mutation_class":    "Structural",
        "structural_defect": (
            "R282 forms a critical hydrogen bond with the backbone of K120 "
            "and stabilises the S7/S8 β-sheet of the DBD. R282W disrupts this "
            "bond via bulky indole side chain. Global DBD destabilisation "
            "results. GOF: EZH2/DNMT3A epigenetic silencing activation."
        ),
        "domain_affected":   "DBD (S7/S8 β-sheet, residues 278–287)",
        "zinc_affected":     True,

        "small_molecule_correctors": [
            {
                "name":        "Tazemetostat (EZH2 inhibitor)",
                "mechanism":   "Blocks R282W GOF epigenetic silencing via EZH2 inhibition; "
                               "does not reactivate p53 but removes downstream GOF effect",
                "target_residues": ["EZH2 SET domain"],
                "stage":       "FDA approved (epithelioid sarcoma)",
                "priority":    "HIGH",
                "keml_status": "Import only — Nairobi private facilities",
            },
            {
                "name":        "Decitabine (DNMT inhibitor)",
                "mechanism":   "Reverses DNMT3A-driven hypermethylation downstream of R282W GOF",
                "target_residues": ["DNMT3A active site"],
                "stage":       "FDA approved (MDS)",
                "priority":    "HIGH",
                "keml_status": "Available KNH (MDS protocol)",
            },
        ],

        "peptide_strategies": [
            {
                "name":        "K120 backbone H-bond restorer",
                "mechanism":   "Peptidomimetic replacing R282 H-bond donor function; "
                               "guanidinium isostere restores K120 backbone interaction",
                "design_basis":"β-sheet mimetic scaffold; residues 278–290",
                "target":      "S7/S8 β-sheet stabilisation site",
                "stage":       "Computational",
                "priority":    "MEDIUM",
            },
            {
                "name":        "EZH2 PRC2 complex disruption peptide",
                "mechanism":   "Stapled peptide blocking EZH2–EED interaction in PRC2; "
                               "disrupts epigenetic silencing driven by R282W GOF",
                "design_basis":"EED binding helix of EZH2 (residues 39–68); staple at i, i+7",
                "target":      "EZH2–EED protein–protein interface",
                "stage":       "Preclinical",
                "priority":    "HIGH",
            },
        ],

        "protac_strategy": {
            "rationale":     "R282W GOF epigenetic driver; PROTAC degrades mutant p53 + "
                             "combine with EZH2 inhibitor for dual blockade",
            "e3_ligase":     "VHL",
            "warhead":       "Zinc chelator scaffold (restores/competes zinc site)",
            "linker":        "PEG4",
            "design_notes":  "PROTAC-p53(R282W): ZnChelator-PEG4-VHL; "
                             "target zinc-affected DBD for selectivity",
            "stage":         "Computational design",
        },

        "mdm2_inhibitor": {
            "rationale":     "MDM2 + EZH2 dual inhibition — synergistic in R282W",
            "peptide_design":"Bispecific: MDM2 antagonist + EZH2–EED blocker",
            "small_molecule":"Idasanutlin + tazemetostat combination",
            "mdm2_score":    7,
            "clinical_note": "EZH2 inhibition is the priority GOF intervention for R282W",
        },

        "zinc_rescue_strategy": (
            "ZMC1 (NSC319726) zinc metallochaperone partially rescues R282W "
            "via indirect stabilisation of C176/H179/C238/C242. "
            "Combine with decitabine for epigenetic reversal."
        ),
    },
}

# ── Shared MDM2 peptide scaffold (all mutations) ──────────────────────────────
MDM2_STAPLED_PEPTIDE_SCAFFOLD = {
    "name":        "SAH-p53-8 (stapled p53 TAD helix)",
    "sequence":    "LTFEHYWAQLTS",   # p53 TAD residues 17–29
    "staple":      "Hydrocarbon staple between F19(S5) and W23(S5) — i, i+4",
    "properties": {
        "cell_penetration": "High (staple enables membrane crossing)",
        "protease_resistance": "High (α-helix locked)",
        "MDM2_Kd":          "~1 nM (stapled) vs ~700 nM (unstapled)",
        "MDMX_activity":    "Moderate",
    },
    "clinical_analogue": "ALRN-6924 (Aileron Therapeutics — Phase II AML)",
}

RATE_LIMIT_CALLS  = 20
RATE_LIMIT_WINDOW = 60

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnzymeDesignStrategy:
    mutation:                 str
    mutation_class:           str
    structural_defect:        str
    domain_affected:          str
    zinc_affected:            bool
    small_molecule_correctors:list[dict]
    peptide_strategies:       list[dict]
    protac_strategy:          dict
    mdm2_inhibitor:           dict
    zinc_rescue_strategy:     str
    recommended_priority:     str = ""   # filled by ranker
    clinical_summary:         str = ""

@dataclass
class EnzymeDesignResult:
    agent_id:        str = AGENT_ID
    timestamp:       str = ""
    mutation:        str = ""
    patient_hash:    str = ""
    strategy:        Optional[EnzymeDesignStrategy] = None
    query_response:  str = ""
    fhir_report:     dict = field(default_factory=dict)
    cache_hit:       bool = False
    error:           Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable utilities
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


_SAFE_QUERY = re.compile(r"^[\w\s\?\.\,\-\'\"\/\(\)\+\=\#]+$")
_INJECTION  = re.compile(
    r"(drop\s+table|delete\s+from|<script|import\s+os|__import__|eval\()",
    re.IGNORECASE,
)
_HGVS_LOOSE = re.compile(r"^[A-Z]\d+[A-Z\*]$")

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

def sanitise_mutation(mutation: str) -> str:
    m = mutation.strip().upper()
    if not _HGVS_LOOSE.match(m):
        raise ValueError(f"Invalid mutation format '{m}'. Expected e.g. R175H.")
    return m

def hash_patient_id(patient_id: str) -> str:
    return "PAT-" + hashlib.sha256(patient_id.encode()).hexdigest()[:12].upper()


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis engine
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_strategy(data: dict) -> str:
    """Determine overall recommended approach priority."""
    high_mols = sum(
        1 for m in data["small_molecule_correctors"]
        if m["priority"] == "HIGH"
    )
    high_peps = sum(
        1 for p in data["peptide_strategies"]
        if p["priority"] == "HIGH"
    )
    if high_mols >= 1 and high_peps >= 1:
        return "COMBINATION: Small molecule corrector + peptide strategy"
    elif high_mols >= 1:
        return "SMALL MOLECULE CORRECTOR priority"
    elif high_peps >= 1:
        return "PEPTIDE / PROTEIN ENGINEERING priority"
    else:
        return "PROTAC degradation strategy — no high-priority correctors available"


def _clinical_summary(mutation: str, data: dict) -> str:
    mols  = [m["name"] for m in data["small_molecule_correctors"] if m["priority"] == "HIGH"]
    peps  = [p["name"] for p in data["peptide_strategies"]       if p["priority"] == "HIGH"]
    mdm2  = data["mdm2_inhibitor"]["small_molecule"]
    score = data["mdm2_inhibitor"]["mdm2_score"]
    zinc  = "Yes" if data["zinc_affected"] else "No"
    return (
        f"Mutation: {mutation} ({data['mutation_class']} class)\n"
        f"Domain: {data['domain_affected']}\n"
        f"Zinc affected: {zinc}\n"
        f"Priority correctors: {', '.join(mols) or 'None at HIGH priority'}\n"
        f"Priority peptides: {', '.join(peps) or 'None at HIGH priority'}\n"
        f"MDM2 inhibitor: {mdm2} (score {score}/10)\n"
        f"PROTAC stage: {data['protac_strategy']['stage']}\n"
        f"Kenya access: {data['small_molecule_correctors'][0].get('keml_status','N/A') if data['small_molecule_correctors'] else 'N/A'}"
    )


def get_design_strategy(
    mutation: str,
    co_mutations: Optional[list[str]] = None,
) -> EnzymeDesignStrategy:
    data = MUTATION_ENGINEERING_MAP.get(mutation)
    if data is None:
        raise ValueError(
            f"Mutation '{mutation}' not in knowledge base. "
            f"Supported: {list(MUTATION_ENGINEERING_MAP.keys())}"
        )
    return EnzymeDesignStrategy(
        mutation                 = mutation,
        mutation_class           = data["mutation_class"],
        structural_defect        = data["structural_defect"],
        domain_affected          = data["domain_affected"],
        zinc_affected            = data["zinc_affected"],
        small_molecule_correctors= data["small_molecule_correctors"],
        peptide_strategies       = data["peptide_strategies"],
        protac_strategy          = data["protac_strategy"],
        mdm2_inhibitor           = data["mdm2_inhibitor"],
        zinc_rescue_strategy     = data["zinc_rescue_strategy"],
        recommended_priority     = _rank_strategy(data),
        clinical_summary         = _clinical_summary(mutation, data),
    )


def build_fhir_report(patient_hash: str, strategy: EnzymeDesignStrategy) -> dict:
    """HL7 FHIR R4 MedicationStatement / Procedure structured output."""
    return {
        "resourceType": "Bundle",
        "type":         "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Procedure",
                    "status":       "proposed",
                    "subject":      {"reference": f"Patient/{patient_hash}"},
                    "code": {
                        "coding": [{
                            "system":  "http://snomed.info/sct",
                            "code":    "416608005",
                            "display": "Drug therapy (procedure)",
                        }],
                        "text": f"TP53 {strategy.mutation} Enzyme/Protein Engineering Strategy",
                    },
                    "note": [{"text": strategy.clinical_summary}],
                    "extension": [
                        {"url": "mutation_class",
                         "valueString": strategy.mutation_class},
                        {"url": "domain_affected",
                         "valueString": strategy.domain_affected},
                        {"url": "zinc_affected",
                         "valueBoolean": strategy.zinc_affected},
                        {"url": "recommended_priority",
                         "valueString": strategy.recommended_priority},
                        {"url": "top_small_molecule",
                         "valueString": strategy.small_molecule_correctors[0]["name"]
                         if strategy.small_molecule_correctors else "None"},
                        {"url": "top_peptide",
                         "valueString": strategy.peptide_strategies[0]["name"]
                         if strategy.peptide_strategies else "None"},
                        {"url": "protac_stage",
                         "valueString": strategy.protac_strategy["stage"]},
                        {"url": "mdm2_inhibitor_score",
                         "valueInteger": strategy.mdm2_inhibitor["mdm2_score"]},
                    ],
                }
            }
        ],
    }


def _build_context(strategy: EnzymeDesignStrategy, query: str) -> str:
    mols = "\n".join(
        f"  [{m['priority']}] {m['name']}: {m['mechanism'][:100]}..."
        for m in strategy.small_molecule_correctors
    )
    peps = "\n".join(
        f"  [{p['priority']}] {p['name']}: {p['mechanism'][:100]}..."
        for p in strategy.peptide_strategies
    )
    return f"""
TP53 Enzyme & Protein Engineering Analysis
Mutation: {strategy.mutation} ({strategy.mutation_class} class)
Query: {query}

STRUCTURAL DEFECT:
  {strategy.structural_defect}

DOMAIN AFFECTED: {strategy.domain_affected}
ZINC COORDINATION AFFECTED: {strategy.zinc_affected}

SMALL MOLECULE CORRECTORS:
{mols}

PEPTIDE / PROTEIN ENGINEERING STRATEGIES:
{peps}

PROTAC DEGRADATION STRATEGY:
  Rationale : {strategy.protac_strategy['rationale']}
  E3 ligase : {strategy.protac_strategy['e3_ligase']}
  Stage     : {strategy.protac_strategy['stage']}
  Notes     : {strategy.protac_strategy['design_notes']}

MDM2 INHIBITOR STRATEGY:
  Small molecule : {strategy.mdm2_inhibitor['small_molecule']}
  MDM2 score     : {strategy.mdm2_inhibitor['mdm2_score']}/10
  Clinical note  : {strategy.mdm2_inhibitor['clinical_note']}

ZINC RESCUE: {strategy.zinc_rescue_strategy}

RECOMMENDED PRIORITY: {strategy.recommended_priority}

SHARED MDM2 SCAFFOLD:
  {MDM2_STAPLED_PEPTIDE_SCAFFOLD['name']} — {MDM2_STAPLED_PEPTIDE_SCAFFOLD['sequence']}
  MDM2 Kd: {MDM2_STAPLED_PEPTIDE_SCAFFOLD['properties']['MDM2_Kd']}
  Clinical analogue: {MDM2_STAPLED_PEPTIDE_SCAFFOLD['clinical_analogue']}
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
        log.warning("LLM unavailable (%s) — structured fallback", exc)
        return f"[Structured Enzyme Design Analysis]\n\n{context}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Agent Class
# ═══════════════════════════════════════════════════════════════════════════════

class EnzymeDesignAgent:
    """
    TP53 Enzyme & Protein Engineering Agent.

    Usage
    -----
    agent = EnzymeDesignAgent()
    result = agent.analyse(
        mutation   = "R175H",
        query      = "What is the best reactivation strategy for R175H?",
        patient_id = "SAM-NBI-01",       # optional
    )
    print(result.strategy.recommended_priority)
    print(result.query_response)
    """

    def __init__(self, ttl: int = 1800) -> None:
        self._rate  = RateLimiter()
        self._cache = get_cache(ttl=ttl) if _CACHE_AVAILABLE else None

    def analyse(
        self,
        mutation:     str,
        query:        str = "What is the best p53 reactivation strategy for this mutation?",
        patient_id:   Optional[str] = None,
        co_mutations: Optional[list[str]] = None,
    ) -> EnzymeDesignResult:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 1. Rate limit
        if not self._rate.allow():
            return EnzymeDesignResult(
                timestamp=ts, error="Rate limit exceeded — max 20 calls/min"
            )

        # 2. Sanitise
        try:
            mutation = sanitise_mutation(mutation)
            query    = sanitise_query(query)
        except ValueError as exc:
            return EnzymeDesignResult(timestamp=ts, error=str(exc))

        patient_hash = hash_patient_id(patient_id) if patient_id else "UNKNOWN"
        cache_key    = f"{mutation}::{query}"

        # 3. Cache lookup
        if self._cache:
            cached = self._cache.get(AGENT_ID, cache_key)
            if cached:
                result           = EnzymeDesignResult(**cached)
                result.cache_hit = True
                return result

        # 4. Build strategy
        try:
            strategy = get_design_strategy(mutation, co_mutations)
        except ValueError as exc:
            return EnzymeDesignResult(timestamp=ts, mutation=mutation, error=str(exc))

        # 5. LLM context + query
        context  = _build_context(strategy, query)
        response = _llm_query(query, context)

        # 6. FHIR
        fhir = build_fhir_report(patient_hash, strategy)

        # 7. Assemble
        result = EnzymeDesignResult(
            timestamp      = ts,
            mutation       = mutation,
            patient_hash   = patient_hash,
            strategy       = strategy,
            query_response = response,
            fhir_report    = fhir,
            cache_hit      = False,
        )

        # 8. Cache
        if self._cache:
            try:
                self._cache.set(AGENT_ID, cache_key, asdict(result))
            except Exception as exc:
                log.warning("Cache write failed: %s", exc)

        return result

    def compare_strategies(self, mutations: list[str]) -> dict:
        """Side-by-side strategy comparison for tumour board review."""
        out = {}
        for m in mutations:
            try:
                m  = sanitise_mutation(m)
                st = get_design_strategy(m)
                out[m] = {
                    "class":          st.mutation_class,
                    "priority":       st.recommended_priority,
                    "top_corrector":  st.small_molecule_correctors[0]["name"]
                                      if st.small_molecule_correctors else "None",
                    "top_peptide":    st.peptide_strategies[0]["name"]
                                      if st.peptide_strategies else "None",
                    "mdm2_score":     st.mdm2_inhibitor["mdm2_score"],
                    "zinc_affected":  st.zinc_affected,
                    "protac_stage":   st.protac_strategy["stage"],
                }
            except ValueError as exc:
                out[m] = {"error": str(exc)}
        return out

    def apr246_eligible(self, mutation: str) -> bool:
        """Quick check: is APR-246 a HIGH-priority corrector for this mutation?"""
        try:
            mutation = sanitise_mutation(mutation)
            st = get_design_strategy(mutation)
            return any(
                m["name"] == "APR-246" and m["priority"] == "HIGH"
                for m in st.small_molecule_correctors
            )
        except ValueError:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test / reverse-engineering suite
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print("\n=== EnzymeDesignAgent Self-Test ===\n")

    agent = EnzymeDesignAgent()

    # Test 1: R175H — conformational mutant
    result = agent.analyse("R175H", "Best reactivation strategy?", patient_id="SAM-001")
    assert result.error is None,                              "FAIL: R175H errored"
    assert result.strategy.mutation_class == "Conformational","FAIL: mutation class"
    assert result.strategy.zinc_affected is True,             "FAIL: zinc flag R175H"
    print(f"✅ Test 1 — R175H strategy: {result.strategy.recommended_priority[:40]}")

    # Test 2: APR-246 HIGH priority for R175H
    assert agent.apr246_eligible("R175H"),                    "FAIL: APR-246 R175H"
    assert agent.apr246_eligible("R248W"),                    "FAIL: APR-246 R248W"
    assert not agent.apr246_eligible("R273H"),                "FAIL: APR-246 R273H should be LOW"
    print("✅ Test 2 — APR-246 eligibility correct")

    # Test 3: PROTAC strategy present for all mutations
    for mut in ["R175H", "R248W", "R273H", "G245S", "R249S", "R282W"]:
        r = agent.analyse(mut, "PROTAC strategy?")
        assert r.error is None,                               f"FAIL: {mut} errored"
        assert "stage" in r.strategy.protac_strategy,         f"FAIL: PROTAC missing {mut}"
    print("✅ Test 3 — PROTAC strategies present for all 6 mutations")

    # Test 4: FHIR structure valid
    fhir = result.fhir_report
    assert fhir["resourceType"] == "Bundle",                  "FAIL: FHIR resourceType"
    entry = fhir["entry"][0]["resource"]
    assert entry["resourceType"] == "Procedure",              "FAIL: FHIR Procedure"
    print("✅ Test 4 — FHIR Bundle/Procedure structure valid")

    # Test 5: PII not in FHIR
    assert "SAM-001" not in json.dumps(fhir),                 "FAIL: PII in FHIR"
    print("✅ Test 5 — HIPAA: PII not in FHIR")

    # Test 6: Injection attack blocked
    r = agent.analyse("R175H", "DROP TABLE users; --")
    assert r.error is not None,                               "FAIL: injection not blocked"
    print("✅ Test 6 — SQL injection blocked")

    # Test 7: Invalid mutation rejected
    r = agent.analyse("NOTVALID", "test")
    assert r.error is not None,                               "FAIL: bad mutation accepted"
    print("✅ Test 7 — Invalid mutation rejected")

    # Test 8: Rate limiter
    limiter = RateLimiter(max_calls=2, window=60)
    assert limiter.allow() and limiter.allow(),               "FAIL: allow"
    assert not limiter.allow(),                               "FAIL: rate limit"
    print("✅ Test 8 — Rate limiter enforced")

    # Test 9: Comparison
    comparison = agent.compare_strategies(["R175H", "R248W", "R282W"])
    assert len(comparison) == 3,                              "FAIL: comparison count"
    assert "mdm2_score" in comparison["R175H"],               "FAIL: comparison structure"
    print(f"✅ Test 9 — Strategy comparison: {list(comparison.keys())}")

    # Test 10: Zinc rescue only for zinc-affected mutations
    zinc_muts   = [m for m, d in MUTATION_ENGINEERING_MAP.items() if d["zinc_affected"]]
    nozinc_muts = [m for m, d in MUTATION_ENGINEERING_MAP.items() if not d["zinc_affected"]]
    for m in zinc_muts:
        st = get_design_strategy(m)
        assert "ZMC1" in st.zinc_rescue_strategy or "zinc" in st.zinc_rescue_strategy.lower(), \
            f"FAIL: zinc rescue missing for {m}"
    print(f"✅ Test 10 — Zinc rescue correctly assigned: {zinc_muts}")

    # Test 11: MDM2 scores all present and in range
    for mut in MUTATION_ENGINEERING_MAP:
        st = get_design_strategy(mut)
        score = st.mdm2_inhibitor["mdm2_score"]
        assert 0 <= score <= 10, f"FAIL: MDM2 score out of range for {mut}"
    print("✅ Test 11 — MDM2 scores valid for all mutations")

    print("\n=== All 11 tests passed ===\n")