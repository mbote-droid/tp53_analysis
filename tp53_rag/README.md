# 🧬 TP53 RAG Platform — Multi-Omics AI for Precision Oncology

> **Gemma 4 Good Hackathon Submission** — A local-first, privacy-preserving multi-agent AI platform for TP53 analysis

[![Gemma 4](https://img.shields.io/badge/Powered%20by-Gemma%204-blue)](https://ai.google.dev/gemma)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-green)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://tp53-rag.streamlit.app)

TP53 RAG Platform is an enterprise-grade, multi-agent AI system for genomic researchers, oncologists, and pharmaceutical companies. Combines local-first Gemma 4 4B inference, 14 specialized AI agents, multi-omics integration, and HIPAA-compliant FHIR R4 output.

## 🎯 What's New (RAG Platform)

**This is a complete rewrite from the basic bioinformatics pipeline to an enterprise AI platform:**

| Feature | Classic Pipeline | 🆕 RAG Platform |
|---------|-----------------|-----------------|
| **Inference** | NCBI API calls | Local Gemma 4 (no data leaks) |
| **Agents** | Single sequential analysis | 14 specialized multi-agents |
| **Voice Input** | ❌ | ✅ Whisper transcription |
| **Drug Discovery** | Manual literature review | Automated APR-246 & KEML screening |
| **Clinical Output** | CSV + PNG | FHIR R4 compliant dossiers |
| **Speed** | 2-5 minutes | <30 seconds (with voice cache) |
| **HIPAA** | ❌ | ✅ PII scrubbing + audit logging |
| **GPU Ready** | ❌ | ✅ llama.cpp optimization |

## 🏗️ Architecture

**14 AI Agents + 1 Orchestrator:**

```
USER INPUT (Text/Voice/VCF)
    ↓
DISPATCHER (Routes to agents)
    ├→ Agent 1: Variant Curator (ClinVar/COSMIC classification)
    ├→ Agent 2: Drug Discovery (APR-246, KEML, therapeutic targeting)
    ├→ Agent 3: Immunogenicity (TME profiling, checkpoint response)
    ├→ Agent 4: Gene Expression (Pathway analysis, RNA-seq)
    ├→ Agent 5: Enzyme Design (PROTAC, molecular glues, zinc rescue)
    ├→ Agent 6: Liquid Biopsy (ctDNA VAF trends, resistance)
    ├→ Agent 7: Dossier Compiler (FHIR R4 + academic/pharma reports)
    ├→ Agent 8: Surgical Brief (Clinical interpretation for oncology)
    ├→ Agent 9: Auditor (Quality control & fact-checking)
    ├→ Agent 10: African Drift (Regional variant prevalence)
    ├→ Agent 11: Multilingual (Query translation & cross-language support)
    ├→ Agent 12: PDF Report (Enterprise dossier generation)
    ├→ Agent 13: Structure Viz (3D protein visualization, Mol*)
    └→ Agent 14: Clinical Interpretation (Prognosis & cancer associations)
    ↓
GEMMA 4 4B (Local inference via llama.cpp)
    ↓
CHROMADB RAG (TP53 knowledge base + HNSW indexing)
    ↓
FHIR R4 + PDF + JSON REPORT
```

## 🚀 Core Features

✅ **Local Inference**: Gemma 4 4B via llama.cpp (8GB RAM compatible, no API calls)  
✅ **Voice Input**: Real-time transcription via Whisper  
✅ **Hybrid Search**: Keyword + semantic retrieval from RAG store  
✅ **Semantic Cache**: Avoid redundant LLM calls  
✅ **Self-Correction**: Automatic retry + fallback logic  
✅ **PII Scrubbing**: HIPAA-compliant output filtering  
✅ **JSON Guardrails**: Strict output formatting (no hallucinations)  
✅ **FHIR R4 Export**: HL7 clinical interoperability  
✅ **n8n Workflows**: Visual node-based automation  
✅ **Docker-Ready**: Single-command deployment  

## 📊 Use Cases

| User | Workflow |
|------|----------|
| **Oncologist** | Upload patient TP53 variant → Get clinical interpretation + drug recommendations |
| **Researcher** | Record voice query → Instant literature-grounded answer + sources |
| **Pharma R&D** | Batch analyze mutations → Identify synthetic lethal targets → Generate IND dossiers |
| **Academic Lab** | Local deployment → No cloud costs, full data privacy |

## 🛠️ Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- NCBI Email (for sequence fetching)

### Installation

```bash
git clone https://github.com/yourusername/tp53_analysis.git
cd tp53_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env → set ENTREZ_EMAIL & LLAMA_CPP_HOST
```

### Launch Web Interface

```bash
# Terminal 1: Start llama.cpp inference server
wget https://huggingface.co/Google/gemma-4-e4b-GGUF/resolve/main/gemma-4-e4b-Q4_K_M.gguf
./llama-server -m gemma-4-e4b-Q4_K_M.gguf -c 8192 --timeout 300 --threads 4 --parallel 2

# Terminal 2: Build knowledge base (first time only)
cd tp53_rag
python main.py build

# Terminal 3: Launch web app
streamlit run app_rag.py
```

Visit: `http://localhost:8501`

**Features on web app:**
- 🎯 Quick Query — Text-based RAG questions
- 🎤 Voice Input — Whisper transcription + auto-analysis
- 📊 Analysis — Multi-agent dispatcher dashboard
- 📋 History — Query tracking
- ⚙️ Settings — HIPAA controls

### Docker (One-Command Deployment)

```bash
docker-compose up
```

Access:
- **Web**: http://localhost:8501
- **API**: http://localhost:8000/docs
- **n8n**: http://localhost:5678

## 📚 Usage

### CLI Commands

```bash
# Interactive Q&A with RAG
python tp53_rag/main.py query

# Run full demo (all 9 agents)
python tp53_rag/main.py demo

# Test individual agent
python tp53_rag/main.py test-rag
python tp53_rag/main.py test-variant

# REST API server
python tp53_rag/main.py serve

# 3D structure visualization
python tp53_rag/main.py visualise --accession NM_000546

# List all agents
python tp53_rag/main.py list-agents
```

### Python API

```python
# Variant classification
from tp53_rag.agents.variant_curator import VariantCurator
curator = VariantCurator()
result = curator.classify("R175H")
print(result['pathogenicity'], result['clinvar_class'])

# Drug discovery
from tp53_rag.agents.dispatcher import AgentDispatcher
from tp53_rag.knowledge_base.vector_store import TP53VectorStore
dispatcher = AgentDispatcher(vector_store=TP53VectorStore())
result = dispatcher.dispatch_single(
    agent_type="drug_discovery",
    custom_question="What drugs target R175H?"
)
print(result.answer)

# Immunogenicity prediction
from tp53_rag.agents.immunogenicity import ImmunogenicityPredictor
predictor = ImmunogenicityPredictor()
tme = predictor.predict("R248W")
print(tme['tme_status'], tme['checkpoint_recommendation'])
```

### Classic Bioinformatics Pipeline (Still Available)

```bash
# Full analysis with all steps
streamlit run app.py

# Or CLI
python main_tp53_analysis.py --accession NM_000546 --skip-phylo --skip-domains
```

## 📦 Project Structure

```
tp53_analysis/
├── tp53_rag/                      # Main RAG platform
│   ├── app_rag.py                 # 🎤 Streamlit web app (voice + text)
│   ├── main.py                    # CLI orchestrator & agent router
│   ├── agents/
│   │   ├── rag_chain.py           # LLM inference (llama.cpp + ChromaDB)
│   │   ├── dispatcher.py          # Multi-agent orchestration
│   │   ├── variant_curator.py     # ClinVar/COSMIC classification
│   │   ├── immunogenicity.py      # TME profiling
│   │   ├── dossier_compiler.py    # FHIR R4 + PDF reports
│   │   └── ...
│   ├── knowledge_base/
│   │   ├── ingestion.py           # Document processing
│   │   └── vector_store.py        # ChromaDB + HNSW
│   ├── utils/
│   │   ├── voice_transcriber.py   # 🎤 Whisper integration
│   │   ├── rag_cache.py           # Semantic caching
│   │   ├── pii_scrubber.py        # HIPAA scrubbing
│   │   └── logger.py              # Audit logging
│   └── ...
├── app.py                         # 🧬 Classic bioinformatics UI
├── main_tp53_analysis.py          # CLI genomic analysis
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

## 🎯 Hackathon Pitch

**Problem**: TP53 is mutated in 50% of human cancers but notoriously hard to drug.

**Solution**: A local-first, multi-agent AI platform that:
1. **Classifies variants** instantly (ClinVar/COSMIC)
2. **Maps structural defects** for drug design (AlphaFold)
3. **Finds synthetic lethal targets** (DepMap)
4. **Screens drug candidates** (ChEMBL)
5. **Predicts immune response** (TME profiling)
6. **Generates enterprise dossiers** (FHIR R4 + PDF)

**Why judges should care**:
- ✅ **Privacy-first**: Runs locally — no data leaks to cloud
- ✅ **Efficient**: Gemma 4 4B on 8GB RAM (edge-deployable)
- ✅ **Marketable**: Pharma companies will pay for this
- ✅ **Open-source**: LGPL license, fully reproducible

## 📚 Key Publications Integrated

- **IARC TP53 Database**: Clinical significance of mutations
- **COSMIC**: Somatic mutation burden across cancers
- **ClinVar**: Variant pathogenicity classifications
- **DepMap**: Synthetic lethality networks
- **AlphaFold**: Protein structure predictions
- **STRING**: Protein-protein interactions

## 🏥 Clinical Integration

### FHIR R4 Compliance
All outputs are HL7 FHIR R4 compatible for EHR integration:
```json
{
  "resourceType": "ClinicalImpression",
  "status": "completed",
  "code": {"coding": [{"system": "http://snomed.info/sct", "code": "..."}]}
}
```

### HIPAA Compliance
- ✅ PII scrubbing (automatic)
- ✅ Audit logging (HIPAA_AUDIT_LOG)
- ✅ Local inference (no data leaks)
- ✅ Encryption-ready (AWS HealthOmics compatible)

## ⚡ Performance

| Metric | Value |
|--------|-------|
| LLM Latency | 2-5s (Gemma 4 4B, CPU) |
| RAG Retrieval | <500ms (ChromaDB HNSW) |
| Variant Classification | <1s |
| Full Demo | ~30s (6 agents) |
| Memory | 2.8GB (Q4_K_M quant) |

## 🐳 Docker Deployment

```bash
docker-compose up
# Access: Web=:8501, API=:8000, n8n=:5678
```

## 📝 License

MIT License — See [LICENSE](LICENSE)

## 👨‍💻 Author

**Samuel Mbote**  
General Surgery Resident & Bioinformatics Developer  

## 🙏 Acknowledgments

- **Google Gemma Team** — Gemma 4 model
- **Meta LLaMA** — llama.cpp framework
- **Chroma** — Vector database
- **n8n** — Workflow automation

---

**Built for the Gemma 4 Good Hackathon 2026** 🚀
