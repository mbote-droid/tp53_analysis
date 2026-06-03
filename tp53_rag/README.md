# 🧬 TP53 RAG Platform — Multi-Omics AI for Precision Oncology

 — A local-first, privacy-preserving multi-agent AI platform for TP53 analysis

[![Gemma 4](https://img.shields.io/badge/Powered%20by-Gemma%204-blue)](https://ai.google.dev/gemma)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-green)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://tp53-rag.streamlit.app)

TP53 RAG Platform is an enterprise-grade, multi-agent AI system for genomic researchers, oncologists, and pharmaceutical companies. It combines local-first Gemma 4 inference, 16 specialized AI agents, multi-omics integration, and HIPAA-compliant FHIR R4 output — running entirely offline on commodity hardware (8GB RAM, no GPU).

> **Differentiator:** Beyond a generic RAG system, this platform embeds an **African cancer-genomics layer** — regional variant prevalence, an equity/bias drift detector, Kenya/KEML drug-availability context, and Swahili output — making it a clinically-grounded copilot for under-represented populations, not just another TP53 wrapper.

## 🎯 What's New (RAG Platform)

**This is a complete rewrite from the basic bioinformatics pipeline to an enterprise AI platform:**

| Feature | Classic Pipeline | 🆕 RAG Platform |
|---------|-----------------|-----------------|
| **Inference** | NCBI API calls | Local Gemma 4 (no data leaks) |
| **Agents** | Single sequential analysis | 16 specialized multi-agents |
| **Voice Input** | ❌ | ✅ Whisper transcription (text + voice, multimodal) |
| **Accuracy** | Unmeasured | ✅ Benchmarked vs ClinVar/IARC (see Benchmarking) |
| **Drug Discovery** | Manual literature review | Automated APR-246 & KEML screening |
| **Clinical Output** | CSV + PNG | FHIR R4 compliant dossiers |
| **Speed** | 2-5 minutes | <30 seconds (with voice cache) |
| **HIPAA** | ❌ | ✅ PII scrubbing + audit logging |
| **GPU Ready** | ❌ | ✅ llama.cpp optimization |

## 🏗️ Architecture

**16 AI Agents + 1 Orchestrator:**

```
USER INPUT (Text / Voice / VCF)
    ↓
DISPATCHER (parallel routing to agents)
    ├→ Agent 1:  Variant Curator (ClinVar/COSMIC/IARC classification)
    ├→ Agent 2:  Drug Discovery (APR-246, KEML, therapeutic targeting)
    ├→ Agent 3:  Immunogenicity (TME profiling, checkpoint response)
    ├→ Agent 4:  Gene Expression (Pathway analysis, RNA-seq)
    ├→ Agent 5:  Enzyme Design (PROTAC, molecular glues, zinc rescue)
    ├→ Agent 6:  Liquid Biopsy (ctDNA VAF trends, resistance)
    ├→ Agent 7:  Dossier Compiler (FHIR R4 + academic/pharma reports)
    ├→ Agent 8:  Surgical Brief (Clinical interpretation for oncology)
    ├→ Agent 9:  Auditor (Quality control, hallucination & bias checks)
    ├→ Agent 10: African Drift (Regional variant prevalence / equity)
    ├→ Agent 11: Multilingual (Swahili + cross-language support)
    ├→ Agent 12: PDF Report (Enterprise dossier generation)
    ├→ Agent 13: Structure Viz (3D protein visualization, Mol*/3Dmol)
    ├→ Agent 14: Clinical Interpretation (Prognosis & cancer associations)
    ├→ Agent 15: Pathology Vision (H&E slide tissue classification)
    └→ Agent 16: TNM Staging (AJCC clinical staging)
    ↓
GEMMA 4 (Local inference — Ollama / llama.cpp / Google AI Studio API)
    ↓
CHROMADB RAG (TP53 knowledge base + BM25 hybrid + HNSW indexing)
    ↓
FHIR R4 + PDF + JSON REPORT
```

## 🚀 Core Features

✅ **Local Inference**: Gemma 4 via Ollama / llama.cpp (8GB RAM, no GPU, no API calls)  
✅ **Dual-Mode**: Offline (Ollama) or cloud (Google AI Studio) via `INFERENCE_MODE`  
✅ **Multimodal Input**: Type *or* speak — Whisper transcription wired into the query + structure tabs  
✅ **Hybrid Search**: BM25 keyword + semantic vector retrieval, cross-encoder reranking  
✅ **Semantic Cache**: Cosine-similarity cache (0.92 threshold) to avoid redundant LLM calls  
✅ **Self-Correction**: Automatic retry + fallback logic (3 attempts)  
✅ **PII Scrubbing**: SHA-256 hashing — HIPAA-compliant output filtering  
✅ **JSON Guardrails**: Strict output formatting + post-response validation  
✅ **Accuracy Benchmark**: Curator scored against ClinVar/IARC ground truth (offline, repeatable)  
✅ **Animated Clinical UI**: Dark bioinformatics theme, animated VAF/hotspot charts, live agent-status board, animated dispatch network, auto-rotating domain-coloured 3D structure  
✅ **FHIR R4 Export**: HL7 clinical interoperability  
✅ **n8n Workflows**: Visual node-based automation with EHR alerting  
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
git clone https://github.com/mbote-droid/tp53_analysis.git
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
streamlit run tp53_rag/app.py
```

Visit: `http://localhost:8501` (or `8502` locally)

**Web app tabs:**
- 🔍 Query — Text-based RAG questions
- 🧬 Analysis — Multi-agent dispatcher with a live agent-status board
- 💊 Drug Discovery — Therapeutic targeting + KEML availability
- 📊 Visualization — Animated VAF timeline, hotspot chart, dispatch network
- 📋 Report — FHIR-aware clinical report generator
- 🔬 Structure — Auto-rotating, domain-coloured 3D protein + multimodal (voice) narration
- 🎤 Voice — Whisper transcription + auto-analysis
- 🛠 Debug — System status & cache stats
- 🔬 Pathology — H&E slide tissue classification
- 📍 TNM Staging — AJCC clinical staging

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

# Run full multi-agent demo
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
# Variant classification (rule-based, no LLM required)
from agents.variant_curator import VariantCurator
curator = VariantCurator()
result = curator.classify("R175H")
c = result["classification"]
print(c["clinical_significance"], c["iarc_classification"])  # -> pathogenic R1

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

## 🎯 Benchmarking

The Variant Curator is benchmarked against curated **ClinVar / IARC** ground truth so accuracy is measured, not assumed. The harness is offline (rule-based curator, no LLM needed) and fully opt-in — it never touches the live app.

```bash
python -m benchmarks.run_benchmark           # writes a timestamped markdown + JSON report
python -m benchmarks.run_benchmark --no-save # print only
```

Ground truth lives in [`benchmarks/ground_truth.json`](benchmarks/ground_truth.json) (7 pathogenic hotspots incl. R249S — the aflatoxin/sub-Saharan HCC hotspot — plus benign and VUS controls). Reported metrics: exact accuracy, bucketed concordance, IARC concordance, and precision/recall/F1 for pathogenic detection.

> Benchmarking immediately paid off: it caught a hotspot-key bug that was mislabelling every pathogenic hotspot as **VUS**. After the fix, exact accuracy rose **11% → 89%** and pathogenic-detection recall **0% → 100%**. The fix is locked by regression tests.

## 🧪 Testing

```bash
pytest tests/ -v                       # full suite
pytest tests/test_rag_platform.py -q   # unit + agent tests (no live LLM; mocked)
```

Tests run without a live model. Coverage includes the RAG chain, dispatcher, every agent, the visualization helpers (`utils/viz.py`), the benchmark scoring/runner, and a regression lock on the variant-classification fix.

## 📦 Project Structure

```
tp53_analysis/
└── tp53_rag/                      # Main RAG platform
    ├── app.py                     # 🎤 Streamlit web app (10 tabs, animated UI)
    ├── main.py                    # CLI orchestrator & agent router
    ├── agents/                    # 16 specialized agents
    │   ├── rag_chain.py           # LLM inference (Ollama/llama.cpp/API + ChromaDB)
    │   ├── dispatcher.py          # Parallel multi-agent orchestration
    │   ├── variant_curator.py     # ClinVar/COSMIC/IARC classification
    │   ├── immunogenicity.py      # TME profiling
    │   ├── dossier_compiler.py    # FHIR R4 + PDF reports
    │   ├── african_drift.py       # Regional/equity variant analysis
    │   ├── pathology_vision.py    # H&E slide classification
    │   ├── tnm_staging.py         # AJCC staging
    │   └── ...
    ├── knowledge_base/
    │   ├── ingestion.py           # Document processing
    │   └── vector_store.py        # ChromaDB + BM25 + HNSW
    ├── utils/
    │   ├── viz.py                 # 📊 Charts, dispatch network, 3D viewer (pure, tested)
    │   ├── voice_transcriber.py   # 🎤 Whisper integration
    │   ├── rag_cache.py           # Semantic caching
    │   ├── pii_scrubber.py        # HIPAA SHA-256 scrubbing
    │   └── hybrid_search.py       # BM25 + vector fusion
    ├── benchmarks/                # 🎯 Accuracy benchmark (ClinVar/IARC)
    │   ├── ground_truth.json
    │   ├── scoring.py
    │   └── run_benchmark.py
    ├── api/server.py              # FastAPI server (n8n integration)
    ├── tests/test_rag_platform.py # Unit + agent + benchmark tests
    ├── n8n_workflow.json          # Automation workflow (EHR alerting)
    ├── requirements.txt           # Dependencies
    ├── .env.example               # Environment template
    └── README.md                  # This file
```

## 🎯 Platform Pitch

**Problem**: TP53 is mutated in 50% of human cancers but notoriously hard to drug.

**Solution**: A local-first, multi-agent AI platform that:
1. **Classifies variants** instantly (ClinVar/COSMIC)
2. **Maps structural defects** for drug design (AlphaFold)
3. **Finds synthetic lethal targets** (DepMap)
4. **Screens drug candidates** (ChEMBL)
5. **Predicts immune response** (TME profiling)
6. **Generates enterprise dossiers** (FHIR R4 + PDF)

**Why this matters**:
- ✅ **Privacy-first**: Runs locally — no data leaks to cloud
- ✅ **Efficient**: Gemma 4 4B on 8GB RAM (edge-deployable)
- ✅ **Africa-relevant**: Regional variant data + KEML drug context — a genuine differentiator
- ✅ **Open-source**: MIT license, fully reproducible

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

**Dr Samuel Ngigi Mbote**  
General Surgery Resident (COSECSA) · IBM Certified AI Developer · Nairobi, Kenya  
Daktari Genomed Labs  

## 🙏 Acknowledgments

- **Google Gemma Team** — Gemma 4 model
- **Meta LLaMA** — llama.cpp framework
- **Chroma** — Vector database
- **n8n** — Workflow automation

---

**Built by Daktari Genomed Labs · Nairobi, Kenya** 🧬
