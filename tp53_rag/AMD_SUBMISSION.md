# AMD Developer Hackathon: ACT II — Track 3 (Unicorn) Submission

**Project:** Precision Onco Africa — a transparent, offline-first clinical
copilot for TP53 cancer genomics, built for low-resource settings.

**One line:** Six AI specialists debate a cancer case and vote toward a
consensus you can trust — running on AMD infrastructure, working even when the
internet doesn't.

This document is the judge's quick map: it shows how the project meets each
Track 3 criterion, how AMD platforms are used, and why this is a startup, not a
science project.

---

## How this maps to the Track 3 criteria

| Criterion | Where we deliver |
|---|---|
| **Creativity & Originality** | The **Live AI Tumour Board** — specialists that debate, cite evidence, and vote with *earned* confidence — plus an honesty framework that refuses to fabricate. Nothing on the market debates a case this way. |
| **Product/Market Potential** | A concrete user (Dr. Amara, below), a real market (precision oncology for the ~2.7B people in low-oncologist-density regions), and a credible revenue model. |
| **Completeness** | Fully functional: 26 agents, 419 automated tests, real benchmarks vs ClinVar/IARC, containerized, deployed. Not a mock-up. |
| **Use of AMD Platforms** | Fireworks AI (AMD-hardware-hosted models) as a first-class inference mode, AMD Developer Cloud + ROCm for heavy compute, and a ROCm/vLLM benchmark harness. Details below. |

---

## Use of AMD platforms

Precision Onco Africa uses AMD infrastructure in three concrete ways:

1. **Fireworks AI API (AMD-hardware-hosted models).** A first-class, selectable
   inference mode (`INFERENCE_MODE=fireworks`). The multi-agent reasoning and
   answer generation run on **open models served on AMD Instinct GPUs** via
   Fireworks — this is the platform's primary hosted-inference path.
   - Model used: `__FIREWORKS_MODEL__`  *(to be finalised at submission)*
   - Code: `agents/rag_chain.py` → `FireworksBackend` (streaming + non-streaming)

2. **AMD Developer Cloud + ROCm.** Heavy compute (protein-language-model
   variant-effect precompute, and the inference benchmark) runs on AMD Instinct
   GPUs via ROCm/PyTorch. A dedicated harness (`tools/benchmark_amd.py`) probes
   the device, measures matmul throughput, LLM latency, and **vLLM throughput on
   ROCm**, and writes `data/amd_benchmark.json`, which the app displays.

3. **Honest compute reporting.** At startup the app probes and logs the real
   backend (`utils/hardware_probe.py`) — it reports ROCm when genuinely present
   and CPU-only otherwise. No fabricated hardware claims.

> **On the "heterogeneous NPU + GPU" design in [ARCHITECTURE.md](ARCHITECTURE.md):**
> the Ryzen AI NPU path is documented as a **roadmap target**, not a running
> feature — consistent with the hackathon's cloud-based format and this
> project's honesty principle. We do not simulate hardware we do not have.

### Benchmark: local CPU vs AMD cloud

> ⚠️ **Placeholder — to be filled with real numbers from `data/amd_benchmark.json`
> after the AMD Developer Cloud run. No figures are invented.**

| Metric | Local (8 GB CPU, Ollama) | AMD Instinct (Fireworks / ROCm) |
|---|---|---|
| Single-answer latency | `__LOCAL_LATENCY__` | `__AMD_LATENCY__` |
| Tumour-board full run | `__LOCAL_BOARD__` | `__AMD_BOARD__` |
| vLLM throughput (tokens/s) | n/a | `__AMD_TPS__` |
| fp16 matmul (TFLOP/s) | `__LOCAL_TFLOPS__` | `__AMD_TFLOPS__` |

*Once populated, the headline reads, e.g.:* “Moving inference from local CPU to
AMD Instinct via Fireworks cut answer latency from **X s to Y s**, making the
multi-agent debate feel real-time.”

### AMD credit utilisation

- **$100 AMD Developer Cloud credits** → running the ESM-2 variant-effect
  precompute and the ROCm/vLLM inference benchmark on AMD Instinct GPUs.
- **$50 Fireworks AI credits** → serving the hosted inference mode (multi-agent
  reasoning + answer generation) on AMD-hardware-hosted models during
  development and the demo.

---

## Product / Market Potential

### The user: Dr. Amara

> **Dr. Amara is a medical officer at a district hospital in western Kenya.**
> There is no oncologist and no clinical geneticist on staff. A patient's biopsy
> comes back with a TP53 mutation. She has minutes, not hours, no
> multidisciplinary team to convene, and an unreliable internet connection. She
> needs to understand what the mutation means, what the evidence says, what
> treatment is *realistically available near her*, and how to explain it to the
> patient — before she refers them hundreds of kilometres away.

Every feature answers a gap Amara has:

| Amara's gap | The platform's answer |
|---|---|
| No tumour board to convene | **Live AI Tumour Board** (the MDT she lacks) |
| No time to read the literature | **Explainability** trace with citations |
| No geneticist to trust the call | **Confidence + ClinVar guardrails** |
| Drugs she cannot actually obtain | **Equity agent** — locally-available alternatives |
| Patient doesn't speak English | **Kiswahili** patient report |
| No reliable internet | **Offline-first** operation |

**Honest layering (research-use-only):** the tool is decision-support, not a
diagnostic device. Realistic early adopters *today* are oncology trainees,
researchers, and NGO-run clinics using it for decision-support and education;
the frontline-clinician use is the vision, pending clinical validation and
regulatory work. We pitch it as a **copilot**, never as something that diagnoses.

### Why this couldn't exist without AI

Before this generation of models, Amara's only options were *refer-and-wait* or
*guess*. There was no way to convene six specialist perspectives, synthesise the
evidence, **and** flag uncertainty in under a minute — offline, on a laptop.
The multi-agent consensus is the thing that only became possible now. That is
both the originality and the "why AI" answer.

### Why TP53 first (and why that's a strength, not a limit)

TP53 is **mutated in roughly half of all human cancers** — the single most
frequently altered gene in oncology. Going deep on the highest-impact gene,
rather than shallow across many, maximises real clinical value and lets us prove
the architecture on the hardest, best-characterised target.

**The platform is gene-agnostic by design.** The agent-consensus framework,
retrieval layer, and guardrails are not TP53-specific — the same architecture
extends to **BRCA1/2, KRAS, EGFR, BRAF, APC** and beyond by adding curated
knowledge and reference data. TP53 is the first supported gene, not the ceiling:
this is a precision-oncology *platform*, demonstrated on the gene that matters
most.

### Market

- **Target segment:** oncology care in low-specialist-density regions —
  sub-Saharan Africa first, then South/Southeast Asia and rural settings
  globally. The WHO reports vast gaps in oncology and genetics staffing across
  these regions.
- **TAM framing (top-down):** precision-oncology decision-support is a
  multi-billion-dollar and fast-growing category; our beachhead is the
  underserved slice global vendors ignore because it demands offline,
  low-cost, equity-aware operation.  *(Cite specific figures from WHO / market
  reports in the final slide — kept honest, not inflated.)*
- **Beachhead → expansion:** TP53 decision-support for African oncology centres
  and NGOs → multi-gene panel → a precision-oncology operating system.

### Revenue model

- **Per-site SaaS** — annual licence per hospital/clinic, priced for
  low-resource budgets; volume tiers for hospital networks.
- **API / integration** — usage-based access for EHR/LIS vendors and research
  groups (FHIR R4 output already supported).
- **Grant- and NGO-funded deployments** — global-health funders underwrite
  installs in public hospitals (aligns with the equity mission).
- **Research/education tier** — universities and training programmes.

Local-first design keeps per-deployment cost low (no per-query cloud bill in
offline mode), which is what makes the low-resource market economically viable.

---

## Completeness at a glance

- **26 agents**; **25 run without any LLM call** (deterministic curation,
  staging, tumour board, explainability) — the model touches only language
  generation. This is the opposite of a fragile "5-LLM-calls-chained" design.
- **419 automated tests** (2 skip off-GPU), CI on every push.
- Real evaluation vs **ClinVar / IARC** ground truth.
- Containerized (`docker compose up`), public repo, offline-capable.
- Security-hardened (see [SECURITY.md](SECURITY.md)); methods documented (see
  [METHODS.md](METHODS.md)); architecture in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick links

- **Demo app:** `__DEMO_URL__`  *(Streamlit — to be finalised)*
- **Repo:** https://github.com/mbote-droid/precision-onco-africa
- **Guided demo:** [DEMO.md](DEMO.md) · **Methods:** [METHODS.md](METHODS.md) ·
  **Security:** [SECURITY.md](SECURITY.md) · **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
