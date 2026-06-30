# Architecture

This document describes how the platform is built today and the heterogeneous
AMD compute architecture it is designed to grow into. The two are kept clearly
separate: **what runs now** is distinguished from **what is a roadmap target**,
so nothing here overstates the current state.

---

## 1. Current architecture — a modular monolith with a service layer

The platform runs today as a **modular monolith**: one Streamlit process in
which ~26 agents are cleanly-separated Python modules sharing in-process state.
A thin service layer (FastAPI, n8n, docker-compose) sits on top for integration.

```
            User input  (text · voice · VCF upload)
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  Security gate  (utils/security.py)            │
        │  upload validation · prompt sanitisation       │
        └───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  Intent router + token-efficient router        │
        │  cache hit / deterministic agent / LLM         │
        └───────────────────────────────────────────────┘
              │                         │
              ▼                         ▼
   ┌────────────────────┐   ┌──────────────────────────────┐
   │ Deterministic core │   │  RAG + LLM generation         │
   │ tumour board ·     │   │  hybrid retrieval (vector+BM25)│
   │ explainability ·   │   │  → inference backend          │
   │ variant curation · │   │  (Ollama/llama.cpp/Google/     │
   │ TNM · atlas        │   │   Fireworks-on-AMD-Instinct)   │
   └────────────────────┘   └──────────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Dual guardrails (form + ClinVar fact) →       │
        │  confidence → FHIR / PDF / JSON (RUO-stamped)  │
        └───────────────────────────────────────────────┘
```

### Compute that runs today

| Workload | Where it runs today |
|---|---|
| LLM generation (hosted) | **AMD Instinct GPU via Fireworks** (`INFERENCE_MODE=fireworks`) |
| LLM generation (local) | CPU via Ollama / llama.cpp (offline) |
| Heavy bio precompute (ESM-2) | **AMD Developer Cloud GPU via ROCm** (`tools/benchmark_amd.py`, precompute-and-serve) |
| Embeddings | Local ONNX (all-MiniLM) — no network, no key |
| Deterministic agents | CPU, in-process — no accelerator needed |

The live compute backend is probed and logged honestly at startup
(`utils/hardware_probe.py`): it reports ROCm when actually present and CPU-only
otherwise.

---

## 2. Target architecture — heterogeneous AMD silicon (roadmap)

> **Status: roadmap, not yet running.** The design below maps each workload to
> the AMD component best suited to it. The NPU path requires a physical Ryzen AI
> (XDNA) device, which the current build does not target at runtime — it is
> documented here as the intended deployment, consistent with the "Future
> deployment" column in the in-app deployment map.

The principle is **heterogeneous compute**: match each workload's mathematical
profile to the right silicon, so a continuous light stream (audio, embeddings,
a safety gate) never contends with heavy LLM token generation.

```
        ┌─────────────────────────────────────────────┐
        │  Ryzen AI NPU (XDNA) — ROADMAP               │
        │  via ONNX Runtime / VitisAIExecutionProvider │
        │  • voice tokenisation (quantised Whisper)    │
        │  • query embeddings                          │
        │  • PII safety gate (lightweight classifier)  │
        └─────────────────────────────────────────────┘
                            │  (offloads light/continuous work)
                            ▼
        ┌─────────────────────────────────────────────┐
        │  AMD ROCm GPU — AVAILABLE (cloud)            │
        │  via PyTorch / vLLM                          │
        │  • LLM token generation (heavy, bursty)      │
        │  • long-context VCF / EHR processing         │
        └─────────────────────────────────────────────┘
```

**Why it matters (biocomputational rationale):** running BM25 lexical search,
vector embeddings, and LLM generation on the *same* unit causes the token rate
to drop under load ("genomic stutter"). Offloading the continuous embedding and
audio work to the NPU would preserve the full GPU budget for generation,
keeping a steady tokens-per-second rate. This is the engineering value behind
the split — not novelty for its own sake.

### Honest current-vs-target mapping

| Workload | Today | Target (roadmap) |
|---|---|---|
| Voice transcription (Whisper) | CPU (optional) | Ryzen AI NPU (quantised, ONNX) |
| Query embeddings | Local ONNX (CPU) | Ryzen AI NPU |
| PII safety gate | CPU regex + SHA-256 | NPU lightweight classifier |
| LLM generation | AMD Instinct (Fireworks) / CPU local | AMD ROCm GPU (vLLM) |

---

## 3. Why this separation is the right design

- **Determinism where it matters.** Variant curation, TNM staging, the tumour
  board and explainability are rule-based and run without the LLM — the model
  is isolated to language generation only, which is where the dual guardrails
  then check it.
- **Edge-ready.** The deterministic core, ESM-2 lookups, and local inference all
  run offline on commodity hardware; the accelerated paths are additive, not
  required.
- **Swappable backends.** A single `INFERENCE_MODE` switch selects the backend;
  three of four backends are native (no framework lock-in on the hot path).

See [METHODS.md](METHODS.md) for the quantitative methods, [SECURITY.md](SECURITY.md)
for the threat model, and the in-app Debug tab for the live deployment map and
compute probe.
