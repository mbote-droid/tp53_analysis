# Demo recording script — Precision Onco Africa

A ~4-minute screen recording that sells **both** the AMD Track 3 (Unicorn) and
the **Best Use of Gemma** tracks. This is the *shot-list* (what to click, when)
plus the *narration* (what to say). Record it yourself — you narrate in your own
voice.

> This is a companion to [PITCH.md](PITCH.md) (slide deck + high-level script)
> and [DEMO.md](DEMO.md) (judge self-guided walkthrough). This file is the
> **recording choreography**.

---

## Before you hit record

- **Screen recorder:** Windows **Game Bar** (`Win + G` → record) — built in, no
  third-party tool.
- **Inference mode:** set `INFERENCE_MODE=api` in `.env` for the snappiest,
  most reliable demo (Google Gemma, no cold starts). `fireworks` also works.
  *(The AMD-hardware beats — live `rocm-smi`, the self-heal VRAM spike,
  speculative decoding — are recorded separately on the AMD Developer Cloud
  droplet per [AMD_VLLM_SETUP.md](AMD_VLLM_SETUP.md); don't fake them on the
  laptop.)*
- **Pre-load:** open the app, convene the board once on `R175H` so caches are
  warm, then refresh for the take.
- Silence notifications; use a clean browser window; 1080p.

---

## The 4-minute shot-list

| Time | Action (what you click) | Narration (what you say) |
|---|---|---|
| **0:00–0:20** | Title slide, then the app's top banner. | "Half of all cancers carry a TP53 mutation. But Dr. Amara — a district clinician in western Kenya — has no oncologist, no geneticist, and no reliable internet. Precision Onco Africa is the tumour board she never had." |
| **0:20–0:50** | **⭐ Tumour Board** → enter `R175H`, cancer Breast → **Convene the board**. Let the six specialists render. | "Six AI specialists — each with a distinct, orthogonal reasoning posture — form an evidence-grounded opinion and vote toward a consensus. Confidence is *earned*, not asserted." |
| **0:50–1:15** | Click **🧮 Run confidence vote**. Show the probability bar chart + per-specialist table. | "Instead of a wall of text, each specialist returns a probability distribution — six independent votes, run concurrently, aggregated into a mathematical consensus you can *see*." |
| **1:15–1:45** | Click **🔴 Run skeptic cross-examination**. Show the counterfactual viability + the skeptic→proposer exchange. | "Then the honesty layer: an adversarial skeptic actively hunts for evidence that would *contradict* the plan — ClinVar conflicts, trials that were stopped early — and challenges it in a bounded exchange. Most tools only look for evidence that agrees." |
| **1:45–2:20** | **🧬 Molecular → 🔬 Structure** → "Gemma sees the structure": enter `R175H` → **Render & let Gemma see it**. Show the rendered backbone + Gemma's narration side-by-side. | "This is the Gemma track. We render the protein backbone with the mutation highlighted, and hand the *image* to Gemma 4. It doesn't read a description — it **looks** at the structure and reasons about it. One open model for sight and language." |
| **2:20–2:45** | **🔬 Pathology** → **Use sample report** → **Read report with Gemma Vision**. Show extracted fields + ClinVar cross-check. | "Same model, more modalities: Gemma reads a photographed paper lab report — no OCR engine — extracts the variant, and cross-checks it against ClinVar. It also reads H&E slides directly." |
| **2:45–3:10** | **🔍 Query** → Kiswahili expander: type `mgonjwa ana kohoa`. Show HPO/ICD-10 mapping. Then ask a question with **🔊 Read answer aloud**. | "For the people it's built for: a clinician types a symptom in Kiswahili and it's mapped to standard clinical codes. And Jarvis reads the answer aloud — hands-free." |
| **3:10–3:35** | **🎤 Voice & Tools → 🛠 Debug** → Autonomic panel: show live RAM + the honest GPU line; click **Run self-heal now**. | "It's also hardware-aware. One environment variable moves inference from an 8 GB laptop to AMD Instinct — same code, cloud to clinic. And an autonomic manager watches real memory and heals itself. On our AMD droplet, this same panel shows live `rocm-smi` GPU telemetry." |
| **3:35–4:00** | Back to the consensus / "Why?" trace. Close on the title. | "Transparent, offline-capable, honest about what it doesn't know — Gemma on AMD, bringing the tumour board to the clinics that never had one. Thank you." |

---

## Both-tracks callouts (say at least one of each)

- **Track 3 (Unicorn):** the Amara user + market; **Gemma served on AMD
  Instinct** (Fireworks/vLLM); ROCm benchmark (182 s → 5.1 s, ~35×); the honest
  self-healing GPU-ops layer.
- **Best Use of Gemma:** Gemma is the **multimodal core** — sees slides, reads
  lab-report photos (no OCR), looks at rendered structures — plus offline local
  Gemma via Ollama.

## Honesty guardrails for the recording

- Don't show a "VRAM spike" on the laptop — record the GPU/self-heal beat on the
  real AMD droplet.
- Everything else in the shot-list is genuinely live; nothing is scripted or
  faked.
- Keep the RUO framing visible (the banner) — it's a copilot, not a diagnosis.

## Optional B-roll
- The DNA-helix codebase graph (Debug/■ visual) for a 3–5 s establishing shot.
- The README trailer GIF as an intro card.
