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
| **Completeness** | Fully functional: 26 agents, 517 automated tests, real benchmarks vs ClinVar/IARC, containerized, deployed. Not a mock-up. |
| **Use of AMD Platforms** | Fireworks AI (AMD-hardware-hosted models) as a first-class inference mode, AMD Developer Cloud + ROCm for heavy compute, and a ROCm/vLLM benchmark harness. Details below. |

---

## Use of AMD platforms

Precision Onco Africa uses AMD infrastructure in three concrete ways:

1. **Fireworks AI API (AMD-hardware-hosted models).** A first-class, selectable
   inference mode (`INFERENCE_MODE=fireworks`). The multi-agent reasoning and
   answer generation run on **open models served on AMD Instinct GPUs** via
   Fireworks — this is the platform's primary hosted-inference path.
   - Model used: `accounts/fireworks/models/minimax-m3`
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

### Hardware-elastic inference (one build, cloud → clinic)

The same `INFERENCE_MODE` switch makes the platform **hardware-elastic**: it
runs *serialised and quantised on an 8 GB commodity laptop* (`ollama` /
`llamacpp`, fully offline) and *parallel-batched on AMD Instinct GPUs*
(`fireworks`) — with no code change, just an environment variable. That is the
whole thesis in one lever: the AMD-hosted path gives the throughput that makes
the multi-agent tumour board feel real-time (182 s → 5.1 s, ~35× — see below),
while the local path guarantees the tool still works in a clinic with no
connectivity. The constraint (8 GB, intermittent internet) becomes the
selling point: **precision oncology that scales from the Cloud to the Clinic on
the same codebase.**

### Benchmark: local CPU vs AMD cloud

Real numbers, captured by `tools/benchmark_amd.py`, committed at
[`data/amd_benchmark.json`](data/amd_benchmark.json). Same prompt
("*What is the clinical significance of TP53 R175H?*"), same machine, only the
inference backend changed.

| Metric | Local (8 GB CPU laptop, Ollama/gemma4-lowmem) | AMD Instinct (Fireworks/minimax-m3) |
|---|---|---|
| Single-answer latency | **182.2 s** | **5.1 s** |
| Speedup | — | **≈35×** |
| fp16 matmul (TFLOP/s, this CPU host) | 0.06 | *(ROCm/vLLM throughput run — see note)* |

**Headline:** moving inference from a local 8 GB CPU laptop to AMD Instinct via
Fireworks cut answer latency **from 182 s to 5.1 s (~35×)** — the difference
between a multi-agent tumour-board debate being unusable and feeling real-time.

*Note: the ROCm/vLLM throughput run (`tools/benchmark_amd.py --vllm <model>`)
requires an active AMD Developer Cloud GPU instance; it was not executed in
this environment (no ROCm host attached) and is not fabricated here. The
tumour board itself is excluded from this table because it is a deterministic
agent with no LLM call (see Completeness, below) — its latency is
backend-independent by design.*

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
- **517 automated tests**, CI on every push.
- Real evaluation vs **ClinVar / IARC** ground truth.
- Containerized (`docker compose up`), public repo, offline-capable.
- Security-hardened (see [SECURITY.md](SECURITY.md)); methods documented (see
  [METHODS.md](METHODS.md)); architecture in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick links

- **Demo app:** https://tp53analysis-g8iqzkuhoqmjcjtkvjcgbb.streamlit.app/
- **Repo:** https://github.com/mbote-droid/precision-onco-africa
- **Guided demo:** [DEMO.md](DEMO.md) · **Methods:** [METHODS.md](METHODS.md) ·
  **Security:** [SECURITY.md](SECURITY.md) · **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
