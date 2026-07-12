# lablab.ai / AMD Hackathon — Submission Form Copy (paste-ready)

Fill the form fields with the blocks below. Replace `__VIDEO_URL__` after you upload
the demo. Keep it honest — every number here is real and defensible.

---

## Project name
Precision Onco Africa

## Tagline (one line)
Six AI specialists debate a cancer case and vote toward a consensus you can trust —
offline-first, and accelerated on AMD Instinct.

## Track / Category
Track 3 — Unicorn (Startup Potential) · submitting for **Best Use of Gemma** bonus

---

## Short description (≈50 words)
An offline-first, honesty-first clinical copilot for TP53 cancer genomics. Six Gemma-
powered specialist agents reason over a case, cast mathematical votes, and graph a
consensus with uncertainty shown. Runs on an 8 GB laptop with no internet — or
parallel-batched on AMD Instinct GPUs — from the same codebase.

---

## Full description (paste into the long field)

**The problem.** Half of all human cancers carry a TP53 mutation. In low-specialist-
density regions, there is often no oncologist and no geneticist to interpret it — and
no reliable internet to look it up. Clinicians face *refer-and-wait* or *guess*.

**What we built.** Precision Onco Africa convenes the multidisciplinary tumour board
the frontline lacks. Six specialist personas — pathologist, geneticist, oncologist,
surgeon, pharmacologist, and an equity officer — reason over the case. Each returns a
probability distribution over the options; we graph the consensus **and** the
disagreement. A dedicated adversarial-skeptic agent hunts for contradicting evidence,
ClinVar/IARC guardrails flag uncertainty, and the tool is research-use-only by design.
It never claims to diagnose.

**Creative use of Gemma 4 — a multimodal reasoning core, not a chatbot.** Gemma
*sees* a rendered p53 structure and reasons about the mutation's warp; *reads*
photographed lab reports and H&E slides with no OCR layer; *casts a mathematical
vote* as one of six personas; *argues against itself*; and *speaks Kiswahili* for the
patient-facing report. Five distinct jobs from one model.

**Use of AMD.** A single `INFERENCE_MODE` switch makes the platform hardware-elastic:
serialized and quantized on a commodity 8 GB laptop (fully offline), or parallel-
batched on **AMD Instinct GPUs via Fireworks/vLLM** — no code change. Moving inference
to AMD Instinct cut answer latency from **182 s to 5.1 s (~35×)**, the difference
between the tumour board being unusable and feeling real-time. A ROCm/vLLM benchmark
harness and an autonomic self-healing GPU-ops layer read **real** `rocm-smi`/`psutil`
telemetry — we never fabricate hardware numbers.

**Completeness.** 26 agents (25 run with no LLM call — deterministic curation,
staging, and board logic), 517 automated tests with CI on every push, real evaluation
against ClinVar/IARC ground truth, containerized with `docker compose`, and deployed
live. Built solo, on an 8 GB laptop.

**Why it matters.** Precision oncology that scales from the Cloud to the Clinic on the
same codebase — built for the places the internet forgets.

---

## Technologies used (tags)
Gemma 4 · AMD Instinct · ROCm · vLLM · Fireworks AI · Python · Streamlit · ChromaDB ·
LangChain · BM25 hybrid retrieval · cross-encoder reranking · FastAPI · Docker ·
Biopython · FHIR R4 · py3Dmol · multi-agent systems · RAG

## Keywords / themes
precision oncology · TP53 · clinical decision support · offline-first · global health ·
multi-agent · multimodal · responsible AI · health equity

---

## Links
- **Live demo:** https://tp53analysis-g8iqzkuhoqmjcjtkvjcgbb.streamlit.app/
- **Repo:** https://github.com/mbote-droid/precision-onco-africa
- **Demo video:** __VIDEO_URL__
- **Guided demo / methods:** DEMO.md · METHODS.md · ARCHITECTURE.md · AMD_SUBMISSION.md

---

## Pre-submit checklist
- [ ] Demo video uploaded (unlisted YouTube/Vimeo is fine) → paste URL above and in README
- [ ] Live-demo link opens in **incognito** with no login wall
- [ ] Repo is public; README badges render; CI green
- [ ] `AMD_SUBMISSION.md` present (judge quick-map)
- [ ] Track 3 + "Best Use of Gemma" both selected on the form
- [ ] Video shows the **local** run (voice, offline inference, rocm-smi/autonomic ops) — not just the cloud app
