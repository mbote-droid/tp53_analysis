# Precision Onco Africa — Pitch Deck (paste-ready)

**AMD Developer Hackathon ACT II · Track 3 (Unicorn) + Best Use of Gemma**

Each slide below has: **[TITLE]**, on-slide bullets (keep them short — the slide is
not the script), and **Say:** (what you narrate). Aim for ~1 slide / 20–30 s → a
tight 4-minute pitch. Palette to match the app: deep indigo `#0b0e1a`, amethyst
`#8b7cf6`, gold `#f0a830`, coral `#ff5d8f`.

---

## Slide 1 — Title

**Precision Onco Africa**
- Six AI specialists debate a cancer case and vote toward a consensus you can trust
- Offline-first · runs on AMD Instinct · works when the internet doesn't
- *Dr. Samuel Ngigi Mbote — surgeon + AI developer, Nairobi*

**Say:** "Half of all human cancers carry a TP53 mutation. In most of the world,
there's no oncologist in the room to explain what that means. Precision Onco Africa
puts a six-specialist tumour board on a laptop — and it runs offline."

---

## Slide 2 — The Problem (Dr. Amara)

- District hospital, western Kenya. No oncologist. No geneticist.
- A biopsy comes back: **TP53 mutation.** She has minutes, not hours.
- Unreliable internet. Patient doesn't speak English. Drugs on the guideline aren't stocked.

**Say:** "Meet Dr. Amara — a real composite of the users we build for. When a TP53
result lands, she has no team to convene and no reliable connection to look it up.
Every design decision in this project answers a gap she actually has."

---

## Slide 3 — The Solution

- **Live AI Tumour Board** — six specialist personas (pathologist, geneticist,
  oncologist, surgeon, pharmacologist, equity officer)
- Each casts a **mathematical vote** → a graphed consensus, with uncertainty shown
- Citations, ClinVar/IARC guardrails, Kiswahili patient report — **fully offline-capable**

**Say:** "The platform convenes the multidisciplinary team she doesn't have. Six
specialists reason over the case, each returns a probability distribution, and we
graph the consensus — including how much they disagree. Nothing is hidden."

---

## Slide 4 — Honesty by Design (the differentiator)

- **Refuses to fabricate** — RUO-labelled, decision-support, never "diagnosis"
- An **adversarial skeptic agent** hunts for *contradicting* evidence
- Confidence + ClinVar guardrails flag uncertainty instead of bluffing

**Say:** "The thing that makes this trustworthy is that it's built to say 'I'm not
sure.' A dedicated skeptic agent argues *against* the consensus, and every claim is
research-use-only. In clinical AI, honesty is the feature."

---

## Slide 5 — Creative Use of Gemma 4 (multimodal core, not a chatbot)

- **Sees** the rendered p53 structure and reasons about the mutation's warp
- **Reads** photographed lab reports + H&E slides — no OCR
- **Votes** with math (six personas → probability distributions)
- **Argues** against itself · **Speaks** Kiswahili

**Say:** "Gemma 4 isn't a chatbot here — it's a multimodal reasoning core. It sees a
rendered protein structure, reads a photographed lab report with no OCR layer, casts
a numerical vote, argues against itself, and speaks the patient's language. That's
five distinct jobs from one model."

---

## Slide 6 — Use of AMD (hardware-elastic: Cloud → Clinic)

- **One `INFERENCE_MODE` switch:** 8 GB laptop (offline) ↔ AMD Instinct (Fireworks/vLLM)
- **182 s → 5.1 s (~35×)** moving inference to AMD Instinct — real numbers
- **ROCm benchmark harness** + **autonomic self-healing GPU ops** on *real*
  `rocm-smi`/`psutil` telemetry (never faked)

**Say:** "The same codebase runs serialized on a commodity laptop and parallel-batched
on AMD Instinct — one environment variable, no code change. On AMD hardware the
tumour board goes from 182 seconds to 5 — unusable to real-time. And the GPU-ops
layer reports *real* telemetry; we never fabricate hardware numbers."

---

## Slide 7 — Completeness

- **26 agents** — 25 run with **no LLM call** (deterministic curation, staging, board)
- **517 automated tests**, CI on every push
- Real evaluation vs **ClinVar / IARC** · Dockerized · public repo · deployed live

**Say:** "This isn't a demo held together with prompt-chaining. Twenty-five of the
twenty-six agents are deterministic — the model only touches language. Five hundred
seventeen tests run on every push. It's containerized, evaluated against ground
truth, and live right now."

---

## Slide 8 — Market & Model

- **Beachhead:** oncology care in low-specialist-density regions (sub-Saharan Africa first)
- **Revenue:** per-site SaaS · API/EHR integration (FHIR R4) · grant/NGO-funded installs · research tier
- **Gene-agnostic platform** — TP53 first (the highest-impact gene), then BRCA/KRAS/EGFR…

**Say:** "TP53 is the beachhead because it's mutated in half of all cancers — deepest
value on the hardest target. The architecture is gene-agnostic, so this is a
precision-oncology operating system, priced for low-resource budgets and funded by
the global-health grants that already exist."

---

## Slide 9 — Close

- **Precision oncology that scales from the Cloud to the Clinic — on the same codebase**
- Live demo · public repo · MIT
- *Built solo, on an 8 GB laptop, for the places the internet forgets*

**Say:** "Precision Onco Africa scales from an AMD cloud GPU down to a laptop with no
signal — honestly, and on one codebase. It's live, it's open, and it was built for
the clinics the rest of the industry ignores. Thank you."

---

### Backup slides (only if asked)
- **Architecture diagram** (ARCHITECTURE.md) — agent graph + inference modes
- **Benchmark table** (data/amd_benchmark.json) — the 35× number in context
- **Roadmap** — Ryzen AI NPU path (documented as roadmap, not simulated); ESMFold
  generative structural rescue (v2)
