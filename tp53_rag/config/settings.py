"""
============================================================
TP53 RAG Platform - Configuration
============================================================
Central config loaded from .env — single source of truth
for all platform settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Inference mode ────────────────────────────────────────
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-2-2b-it")

# ── Ollama / Gemma 4 ─────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tp53_knowledge_base")

# ── NCBI ─────────────────────────────────────────────────
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# ── RAG hyperparameters ───────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))

# ── API ───────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(LOGS_DIR / "rag_platform.log")

# ── TP53 domain knowledge sources ────────────────────────
# These are the authoritative sources that ground Gemma 4's
# responses in verified clinical and research data.
TP53_KNOWLEDGE_SOURCES = {
    "iarc_tp53": {
        "description": "IARC TP53 Database - somatic mutations in human cancer",
        "url": "https://tp53.isb-cgc.org/",
        "priority": "critical",
    },
    "clinvar_tp53": {
        "description": "ClinVar TP53 pathogenic variants",
        "url": "https://www.ncbi.nlm.nih.gov/clinvar/?term=TP53",
        "priority": "critical",
    },
    "uniprot_p53": {
        "description": "UniProt P04637 - Cellular tumor antigen p53",
        "url": "https://www.uniprot.org/uniprot/P04637",
        "priority": "high",
    },
    "pfam_p53": {
        "description": "Pfam TP53 domain architecture",
        "url": "https://pfam.xfam.org/protein/P04637",
        "priority": "high",
    },
}

# ── Multi-agent function registry ─────────────────────────
# Maps natural language intent → agent function name
# This is what transforms single-tool into platform AI
AGENT_REGISTRY = {
    "mutation_analysis": {
        "description": "Detect, explain and clinically contextualise TP53 mutations",
        "keywords": ["mutation", "variant", "snv", "snp", "change", "substitution"],
    },
    "orf_analysis": {
        "description": "Discover and interpret open reading frames",
        "keywords": ["orf", "reading frame", "open reading", "translation start"],
    },
    "phylogenetic_analysis": {
        "description": "Cross-species TP53 conservation and evolutionary analysis",
        "keywords": ["phylo", "species", "conservation", "evolution", "orthologs", "tree"],
    },
    "domain_annotation": {
        "description": "Protein domain structure and functional annotation",
        "keywords": ["domain", "pfam", "interpro", "motif", "structure", "binding"],
    },
    "clinical_interpretation": {
        "description": "Clinical significance and cancer association of findings",
        "keywords": ["clinical", "cancer", "pathogenic", "benign", "significance", "prognosis"],
    },
    "sequence_fetch": {
        "description": "Fetch and validate TP53 sequences from NCBI",
        "keywords": ["fetch", "accession", "ncbi", "sequence", "download", "retrieve"],
    },
    "report_generation": {
        "description": "Generate comprehensive analysis report from all findings",
        "keywords": ["report", "summary", "comprehensive", "all", "complete", "full"],
    },
}