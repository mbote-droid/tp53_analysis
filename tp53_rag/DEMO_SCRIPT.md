# Demo recording script — Precision Onco Africa

A ~4-minute screen recording that sells **both** the AMD Track 3 (Unicorn) and the
**Best Use of Gemma** tracks. Record it yourself, in your own voice, on your **local**
run (that's where every feature lights up). Companion to [PITCH.md](PITCH.md) /
[SLIDE_DECK.md](SLIDE_DECK.md) (slides) and [DEMO.md](DEMO.md) (self-guided walkthrough).

> **You only touch 5 tabs.** Tab 1 (Tumour Board) is the meat — three button-clicks on
> one tab. The other four are ~30 s each. That's the whole video.

---

## Before you hit record (2-minute setup)

1. **Run locally** (not the cloud app — the cloud app hides voice, offline, and the
   AMD backend badge). Start Streamlit on your machine.
2. **`.env` — set these three:**
   - `INFERENCE_MODE=fireworks` → so the sidebar badge reads **"Fireworks / AMD
     Instinct"** (your on-screen AMD proof). *If Fireworks stalls mid-take, fall back
     to `INFERENCE_MODE=api` and lean on the benchmark for the AMD story.*
   - `GOOGLE_API_KEY=…` → **required** for the two Gemma-vision beats (structure +
     pathology). Without it those buttons warn instead of read.
   - `FIREWORKS_API_KEY=…`
3. **Warm the caches:** open the app, convene the board once on `R175H` / Breast, let
   it finish, then refresh for the clean take.
4. **Screen recorder:** Windows **Game Bar** (`Win + G` → record). Silence
   notifications, clean browser window, 1080p.
5. **Mic on** and grant browser mic permission (for the Jarvis beat).

---

## The 5-tab flow

Each beat below has: **the tab**, **exact clicks/inputs**, **Say** (narration), and
**⚡ Why no chatbot can do this** (say a compressed version out loud — this is what
protects your creativity score).

### ⏱ 0:00–0:15 — Cold open (title card)
**Show:** the `SLIDE_DECK.md` title slide (or the README trailer GIF), then the app's
top banner with the RUO notice visible.
**Say:** "Half of all cancers carry a TP53 mutation. But Dr. Amara — a district
clinician in western Kenya — has no oncologist, no geneticist, and no reliable
internet. Precision Onco Africa is the tumour board she never had."

---

### ⏱ 0:15–1:30 — TAB 1: ⭐ Tumour Board  *(the meat — 3 clicks, one tab)*

**1a. Convene (0:15–0:45)** — Enter `R175H`, cancer = **Breast** → **Convene the
board**. Let the six specialists render.
**Say:** "Six AI specialists — each with a distinct, *orthogonal* reasoning posture —
form an evidence-grounded opinion and vote toward a consensus."
**⚡** Six deliberately different personas, not one voice wearing six hats.

**1b. 🧮 Run confidence vote (0:45–1:05)** — click it; show the probability bar chart +
per-specialist table.
**Say:** "Instead of a wall of text, each specialist returns a *probability
distribution* over the management options — six independent votes, run concurrently,
aggregated into a consensus you can **see**, including how much they disagree."
**⚡** A chatbot gives you one confident paragraph. This **graphs the disagreement.**

**1c. 🔴 Run skeptic cross-examination (1:05–1:30)** — click it; show the
counterfactual viability + the skeptic→proposer exchange.
**Say:** "Then the honesty layer: an adversarial skeptic actively hunts for evidence
that would **contradict** the plan — ClinVar conflicts, trials stopped early — and
challenges it in a bounded exchange."
**⚡** Every chatbot looks for evidence that *agrees*. This one argues *against itself.*

---

### ⏱ 1:30–2:05 — TAB 2: 🧬 Molecular → 🔬 Structure  *(Gemma track)*
**Clicks:** "Gemma sees the structure" → enter `R175H` → **Render & let Gemma see it**.
Show the rendered backbone + Gemma's narration side-by-side.
**Say:** "This is the Gemma track. We render the protein backbone with the mutation
highlighted and hand the **image** to Gemma 4. It doesn't read a description — it
**looks** at the structure and reasons about it."
**⚡** One open model doing *sight* and *language* — not an OCR pipeline bolted on.

---

### ⏱ 2:05–2:35 — TAB 3: 🔬 Pathology  *(Gemma track, more modalities)*
**Clicks:** **✨ Use sample report (synthetic demo)** → **📑 Read report with Gemma
Vision**. Show extracted fields + ClinVar cross-check. *(Sample is built in — one
click, no upload.)*
**Say:** "Same model, more modalities: Gemma reads a **photographed** paper lab report —
no OCR engine — extracts the variant, and cross-checks it against ClinVar. It reads
H&E slides the same way."
**⚡** No OCR, no template. The model *reads the photo* like a clinician would.

---

### ⏱ 2:35–3:05 — TAB 4: 🔍 Query  *(equity + Jarvis — skip generic Q&A)*
**Clicks:** open the **Kiswahili** expander → type `mgonjwa ana kohoa` → show the
HPO/ICD-10 mapping. Then trigger **🔊 Read answer aloud** on an existing answer.
> **Don't** do a plain "type a question, read the answer" — that's the one beat that
> looks like an ordinary chatbot. Show only the Kiswahili mapping + the voice read.
**Say:** "For the people it's built for: a clinician types a symptom in **Kiswahili**
and it maps to standard clinical codes. And Jarvis reads the answer aloud — hands-free."
**⚡** Localized clinical-code mapping + on-device voice, for low-literacy, hands-busy
frontline settings.
**Optional wow:** while Jarvis is speaking, **talk over it** — the barge-in stops it
and it says "Go ahead." (Needs mic permission.)

---

### ⏱ 3:05–3:35 — TAB 5: 🎤 Voice & Tools → 🛠 Debug  *(hardware honesty)*
**Clicks:** Autonomic panel → show live **RAM** + the honest **GPU line** → click
**Run self-heal now** (watch RAM before/after).
**Say:** "It's hardware-aware. One environment variable moves inference from an 8 GB
laptop to AMD Instinct — same code, cloud to clinic. An autonomic manager watches
**real** memory and heals itself."
**⚡** Real telemetry, not a fake dashboard.
> **HONESTY:** on the laptop the GPU line honestly reads *"no AMD GPU present."* Do
> **not** stage a VRAM spike here. The live `rocm-smi` / self-heal-on-VRAM beat is
> recorded separately on the AMD droplet ([AMD_VLLM_SETUP.md](AMD_VLLM_SETUP.md)) if
> you get it up — otherwise narrate it as the documented AMD path, don't fake it.

---

### ⏱ 3:35–4:00 — Close
**Show:** back to the consensus / "Why?" trace, then the title card.
**Say:** "Transparent, offline-capable, and honest about what it doesn't know — Gemma
on AMD, bringing the tumour board to the clinics that never had one. Thank you."

---

## Both-tracks callouts (say at least one of each)
- **Track 3 (Unicorn):** the Amara user + market; **Gemma served on AMD Instinct**
  (Fireworks/vLLM); ROCm benchmark (182 s → 5.1 s, ~35×); the honest self-healing
  GPU-ops layer.
- **Best Use of Gemma:** Gemma is the **multimodal core** — sees rendered structures,
  reads lab-report photos (no OCR), reads H&E slides — plus offline local Gemma via
  Ollama, and the six-persona probability vote.

## Honesty guardrails
- No faked VRAM spike on the laptop — record the GPU beat on the real AMD droplet or
  narrate it as documented.
- Everything else in the flow is genuinely live; nothing is scripted or faked.
- Keep the RUO banner visible — it's a copilot, not a diagnosis.

## If you record in segments (instead of one take)
Cut between **tabs** (natural breaks at each ⏱ boundary above). Start each segment by
clicking the next tab so the joins are clean. You'll stitch them yourself in
**Clipchamp** (built into Windows 11) — I can't assemble video files. One continuous
take is simpler if your nerves allow it.

## Optional B-roll
- The DNA-helix codebase graph (Debug → visual) — 3–5 s establishing shot.
- The README trailer GIF as the intro card.
