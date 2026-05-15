# TP53 RAG Platform 🧬
### Multi-Agent Bioinformatics AI — Gemma 4 Good Hackathon

[![Gemma 4](https://img.shields.io/badge/Powered%20by-Gemma%204-blue)](https://ai.google.dev/gemma)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-green)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)


> **Health & Sciences Track | Ollama Special Track | Safety & Trust Track**

> **Edge deployment optimised** — runs on `gemma4:e2b`, 
> targeting the E2B edge model criteria for resource-constrained 
> clinical settings.

A fully local, offline-capable multi-agent RAG platform that transforms the TP53 bioinformatics 
pipeline into a clinical-grade AI system — powered by **Gemma 4 via Ollama**, grounded by 
**ChromaDB**, and orchestrated by **n8n**.

---

## The Problem

TP53 mutations drive **>50% of all human cancers**, yet expert genomic interpretation is 
inaccessible outside well-funded institutions. A researcher in Nairobi or a clinician in a 
rural hospital cannot access the same AI-powered analysis tools as a cancer centre in New York.

## The Solution

A **100% local, privacy-preserving, offline-capable** multi-agent AI platform that:
- Runs Gemma 4 entirely on local hardware via Ollama
- Grounds every response in verified clinical knowledge (RAG)
- Serves **6 specialised analysis agents** from one model
- Exposes a REST API for n8n workflow automation
- Works without internet after initial setup

---

## Architecture

```
TP53 Pipeline Output
        │
        ▼
┌───────────────────────────────────┐
│     AgentDispatcher               │
│  ┌──────────┐  ┌───────────────┐  │
│  │ Mutation │  │ ORF Analysis  │  │
│  │  Agent   │  │    Agent      │  │
│  ├──────────┤  ├───────────────┤  │
│  │ Phylo    │  │ Domain        │  │
│  │  Agent   │  │   Agent       │  │
│  ├──────────┤  ├───────────────┤  │
│  │ Clinical │  │ Report        │  │
│  │  Agent   │  │   Agent       │  │
│  └──────────┘  └───────────────┘  │
└──────────────┬────────────────────┘
               │
        ┌──────▼──────┐
        │  RAG Chain  │
        │  ┌────────┐ │
        │  │ChromaDB│ │  ← TP53 knowledge base
        │  │(local) │ │    (curated + NCBI + papers)
        │  └────────┘ │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Gemma 4    │  ← Local inference via Ollama
        │  (Ollama)   │    No cloud. No API keys.
        └─────────────┘
               │
        ┌──────▼──────┐
        │  FastAPI    │  ← n8n integration endpoint
        │  REST API   │
        └─────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed (Windows/Mac/Linux)
- 8GB+ RAM recommended

### 1. Pull Gemma 4 and embedding model
```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### 2. Clone and install
git clone https://github.com/mbote-droid/tp53_analysis.git
cd tp53_analysis
mkdir tp53_rag
cd tp53_rag
pip install -r requirements.txt

### 3. Configure
```bash
cp .env.example .env
# Edit .env — set ENTREZ_EMAIL at minimum
```

### 4. Build the knowledge base
```bash
# With internet (fetches NCBI + UniProt too)
python main.py build

# Offline mode (curated knowledge only — works anywhere)
python main.py build --offline
```

### 5. Run a demo
```bash
python main.py demo
```

### 6. Interactive queries
```bash
python main.py query
# Then type: "What is the clinical significance of R248W?"
```

### 7. Start API server (for n8n)
```bash
python main.py serve
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## The Six Agents

| Agent | Function | Key Questions Answered |
|-------|----------|------------------------|
| **Mutation Analysis** | Classifies & interprets detected mutations | Hotspot? Contact or conformational? GOF? Prognosis? |
| **ORF Analysis** | Interprets open reading frames | Which isoform? Biological significance? |
| **Phylogenetics** | Cross-species conservation analysis | Conserved? Functionally critical? Evolutionary context? |
| **Domain Annotation** | Protein domain interpretation | Domain function? Structural implications? |
| **Clinical Interpretation** | Clinical significance assessment | Pathogenic? Cancer associations? Therapy options? |
| **Report Generation** | Synthesis of all findings | Complete structured clinical report |

---

## Knowledge Base

The RAG knowledge base contains curated, verified TP53 information from:

- **Curated embedded knowledge** (offline-first): hotspot mutations, domain architecture, 
  clinical syndromes, pathway biology, therapeutics, phylogenetics, codon usage
- **NCBI Gene** (online, optional): official TP53 gene summaries
- **UniProt P04637** (online, optional): protein functional annotation
- **User PDFs** (optional): drop research papers in `data/documents/`

All knowledge is chunked, embedded via `nomic-embed-text`, and stored in ChromaDB 
with cosine similarity search.

---

## n8n Integration

The FastAPI server exposes endpoints that n8n HTTP Request nodes call:

```
POST /analyse          → Full 6-agent analysis of pipeline output
POST /query            → Single free-form question
POST /agent/{type}     → Specific agent query
GET  /health           → Health check for n8n monitoring
```

Example n8n workflow:
```
[TP53 Pipeline trigger]
    → [HTTP: POST /analyse with pipeline JSON]
    → [Parse response]
    → [Email/Slack report]
    → [Save to database]
```

---

## Why This Wins

### Health & Sciences Track
- Direct clinical utility: TP53 mutations in >50% of cancers
- Democratises expert-level genomic interpretation
- Works in resource-limited clinical settings

### Ollama Special Track
- Gemma 4 runs 100% locally via Ollama
- No cloud dependency, no API keys, no data leaves the machine
- HIPAA-friendly by design

### Safety & Trust Track
- Every response grounded in verified clinical knowledge (RAG)
- Source attribution with relevance scores
- Offline curated fallback prevents hallucination on unsupported queries
- Temperature=0.1 for consistent, reproducible clinical outputs

### Main Track
- Novel multi-agent platform architecture
- Real engineering: working pipeline + RAG + multi-agent + REST API
- Genuine global health impact

---

## Hackathon Details

- **Competition**: Google DeepMind Gemma 4 Good Hackathon
- **Track**: Health & Sciences | Ollama Special | Safety & Trust
- **Model**: Gemma 4 (gemma4) via Ollama
- **Deadline**: May 18, 2026

---

## Authors
- Samuel Mbote

## License
MIT License
