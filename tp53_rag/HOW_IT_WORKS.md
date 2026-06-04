# How It Works — Execution Flow (under the hood)

This document explains **how the codebase runs**, step by step, from the
moment someone opens the app to when an answer is returned — described as a
*cascade of events* (think glycolysis), in plain language rather than code.
It is a companion to `README.md` (which explains *what* the app does and *why*).

> Audience: any engineer (human or AI-assisted) who needs to understand the
> runtime behaviour of this platform without reading every line of code.

---

## Stage 0 — Startup (cold boot)

1. The host runs `streamlit run tp53_rag/app.py`. Streamlit starts a web
   server and executes `app.py` top-to-bottom.
2. **Theme + page config** are applied first (`st.set_page_config`,
   `inject_theme()` — pure CSS/fonts).
3. **RAG modules import** inside a `try/except` (`TP53RAGChain`, dispatcher,
   vector store). If imports fail, the app still loads and shows the real
   traceback — it never hard-crashes.
4. **Knowledge base is prepared once** (`init_rag_system`, cached for the
   process): the ChromaDB vector store is opened; if it is empty, it is
   **auto-built** from the in-code curated TP53 knowledge. Embeddings are
   chosen by environment — Ollama locally, a local ONNX model on cloud (no
   external embedding API).
5. **Session state** is initialised (chat history, pipeline data, etc.) and
   the **sidebar + 12 tabs** render. The app is now idle, waiting for input.

## Stage 1 — Input

6. The user provides input through a tab — a **typed question**, a **voice
   clip** (transcribed locally by Whisper → text), a **mutation + cancer**
   form, or an uploaded **VCF file** (parsed locally → TP53 variants).
7. Input is normalised into a question string and optional `pipeline_data`
   (mutation, cancer type, VAF, …).

## Stage 2 — Retrieval-augmented answer (the RAG core)

When a question needs the LLM, it flows through `safe_query → TP53RAGChain.query`:

8. **Concurrency guard** — a semaphore caps simultaneous inferences so
   parallel users can't exhaust the RAM budget.
9. **Rate-limit** check, then **intent routing** picks the agent type
   (mutation analysis, drug discovery, …) if not forced.
10. **PII scrubbing** — patient identifiers are SHA-256 hashed *before*
    anything leaves the function (HIPAA-safe).
11. **Semantic cache** — the query is embedded; if a past query is ≥0.92
    cosine-similar, the cached answer is returned immediately (no LLM call).
12. **Hybrid search** — otherwise, the knowledge base is searched two ways:
    BM25 (exact terms) + vector similarity, fused. If nothing is found, the
    search is broadened (zero-result handling guarantees a non-empty result).
13. **Reranking** — the top candidates are reranked; the best few become the
    grounding context, trimmed to fit the model's token budget.
14. **LLM generation** — the prompt (system + context + question) is sent to
    the active backend chosen by `INFERENCE_MODE`: Ollama / llama.cpp locally,
    or Google AI Studio (Gemma) on cloud.
15. **Self-correction** — the response is validated (JSON/format guardrails);
    on failure it retries up to 3× with a tightened prompt.
16. The result is **cached** and returned: `{answer, sources, agent_used,
    cache_hit, retries}`.

## Stage 3 — Safety & enrichment

17. **ClinVar hallucination guard** scans the answer: it extracts any TP53
    mutations + the classification the AI claimed and flags conflicts against
    a curated ClinVar reference (e.g. AI says "benign", ClinVar says
    "pathogenic"). Shown inline.
18. Optional **PubMed citations** can be fetched on demand (live Entrez, with
    a search-pointer fallback).

## Stage 4 — Structured agents (no LLM needed)

19. Many panels are driven by **rule-based agents** that return structured
    JSON directly (variant curator, immunogenicity, gene expression, TNM
    staging, African atlas, synthetic-lethality, molecular docking, structural
    analyzer, IND generator, …). These are deterministic, fast, and offline.
20. **Live-data agents** (ChEMBL drugs, ClinicalTrials.gov, PubMed) call
    external APIs **offline-first**: a real call when reachable (cached), a
    curated/search-pointer fallback otherwise — so they never break or block.

## Stage 5 — Visualisation & output

21. Every agent result is rendered by a **pure visualization helper**
    (`utils/viz.py`) — animated charts, gauges, networks, the 3D structure
    viewer — all of which return a non-empty figure even on bad input.
22. Outputs can be exported: **FHIR R4** clinical resources, **markdown/PDF**
    reports, **JSON**, and downloadable drafts (e.g. the IND skeleton).

---

## Cross-cutting principles (true at every stage)

- **Offline-first + graceful degradation** — every external dependency has a
  fallback; the app never hard-fails on a missing service.
- **Never-empty outputs** — every function returns something explainable.
- **Honest data labelling** — illustrative/estimated values (e.g. docking
  affinities, ΔΔG) are marked as such, distinct from real/live data.
- **Logging, not prints** — all diagnostics go through `loguru`.
- **Modular monolith + thin service layer** — agents run in-process; a
  FastAPI/n8n layer exposes the platform for integration (see README).

---

### One-line summary

> Open app → (KB auto-builds once) → input (text/voice/form/VCF) → guarded,
> PII-scrubbed, cache-checked, hybrid-retrieved, reranked context → Gemma LLM
> with self-correction → ClinVar safety check → rule-based agents + live data
> (offline-first) → pure-function visualisations → FHIR/PDF/JSON output.
