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

### ⏱ 1:30–2:05 — TAB 2a: 🧬 Molecular → 🔬 Structure  *(Gemma track)*
**Clicks:** "Gemma sees the structure" → enter `R175H` → **Render & let Gemma see it**.
Show the rendered backbone + Gemma's narration side-by-side.
**Say:** "This is the Gemma track. We render the protein backbone with the mutation
highlighted and hand the **image** to Gemma 4. It doesn't read a description — it
**looks** at the structure and reasons about it."
**⚡** One open model doing *sight* and *language* — not an OCR pipeline bolted on.

### ⏱ 2:05–2:45 — TAB 2b: 🧬 In-Silico Structural Rescue  *(THE CROWN JEWEL — Gemma × AMD)*
Scroll to **In-Silico Structural Rescue**. The **rotating overlay** is already on
screen: wild-type p53 as a green ghost, the **R175H mutant solid in amethyst**, the
mutation site in gold. Click **🧬 Run the rescue reasoning (Gemma ↔ AMD fold)**.
**Say:** "Here's the virtual wet-lab. Gemma *proposes* a second-site suppressor —
N239Y — to rescue the mutation. That candidate sequence was folded for real by
**ESMFold on an AMD Instinct MI300X**: full-length p53 in **2.8 seconds**, and while
it ran the GPU hit **100% utilisation at 749 watts**. Then Gemma reads the measured
geometry back — a 0.71 Å structural warp — and interprets it. This is Gemma reasoning
*about 3D space*, fact-checked by real AMD compute."
**⚡** No chatbot runs a real physics model on a data-center GPU mid-conversation and
reasons over the result. And it's **honest** — pLDDT is framed as confidence, not
stability; it's labelled an in-silico hypothesis, not a cure.
> This one beat carries **both** tracks at once: Gemma (proposes + interprets) and AMD
> (the MI300X does the folding). It's your pattern-interrupt "BOOM" — the moment a
> judge stops thinking "RAG wrapper."

---

### ⏱ 2:45–3:05 — TAB 3: 🔬 Pathology  *(Gemma track, more modalities)*
**Clicks:** **✨ Use sample report (synthetic demo)** → **📑 Read report with Gemma
Vision**. Show extracted fields + ClinVar cross-check. *(Sample is built in — one
click, no upload.)*
**Say:** "Same model, more modalities: Gemma reads a **photographed** paper lab report —
no OCR engine — extracts the variant, and cross-checks it against ClinVar. It reads
H&E slides the same way."
**⚡** No OCR, no template. The model *reads the photo* like a clinician would.

---

### ⏱ 3:05–3:20 — TAB 4: 🔍 Query  *(equity — Kiswahili)*
**Clicks:** open the **Kiswahili** expander → type `mgonjwa ana kohoa` → show the
HPO/ICD-10 mapping. *(Skip plain Q&A — the conversation beat below carries voice.)*
**Say:** "For the people it's built for: a clinician types a symptom in **Kiswahili**
and it maps to standard clinical codes."
**⚡** Localized clinical-code mapping most tools ignore.

---

### ⏱ 3:20–3:50 — TAB 4b: 🎤 Talk to Gemma  *(the human-conversation wow — record LOCAL)*
**Clicks:** Voice tab → **🎙️ Record** a spoken question (e.g. *"What does the R175H
mutation mean for my patient?"*) → Gemma answers, **Jarvis reads it back**, the running
transcript grows. Ask a **follow-up by voice** (Gemma remembers the thread). Then say
**"thank you Gemma, that will be all"** — it recognises the sign-off and closes
gracefully (*"Anytime, doctor."*).
**Say:** "And you can just *talk* to it. Speech in, Gemma answers, Jarvis speaks back —
a real back-and-forth that remembers the conversation, and understands when I'm done."
*(Optional: talk over Jarvis mid-answer — the barge-in stops it.)*
**⚡** A multi-turn spoken conversation with memory and a polite sign-off — fully
on-device (faster-whisper in, browser TTS out). Almost nobody ships this.
> Record on your **local** run; confirm the Voice tab shows **"✅ Speech-to-text
> ready"** first.

---

### ⏱ 3:50–4:10 — TAB 5: 🎤 Voice & Tools → 🛠 Debug  *(hardware honesty)*
**Clicks:** Autonomic panel → show live **RAM** + the honest **GPU line** → click
**Run self-heal now** (watch RAM before/after).
**Say:** "It's hardware-aware. One environment variable moves inference from an 8 GB
laptop to AMD Instinct — same code, cloud to clinic. An autonomic manager watches
**real** memory and heals itself."
**⚡** Real telemetry, not a fake dashboard.
> **HONESTY:** on the laptop the GPU line honestly reads *"no AMD GPU present"* — that's
> correct and fine. Your real AMD proof is the **MI300X capture** from the rescue beat
> (100% GPU, 749 W, 2.8 s fold). Do **not** stage a fake VRAM spike on the laptop.

---

### ⏱ 4:10–4:30 — Close
**Show:** back to the consensus / "Why?" trace, then the title card.
**Say:** "Transparent, offline-capable, and honest about what it doesn't know — Gemma
on AMD, bringing the tumour board to the clinics that never had one. Thank you."

---

## Both-tracks callouts (say at least one of each)
- **Track 3 (Unicorn):** the Amara user + market; **Gemma served on AMD Instinct**
  (Fireworks); **real ESMFold folding on a live MI300X** — full p53 in **2.8 s**, GPU
  at **100% / 749 W** (captured in [`data/amd_mi300x_rocm_smi.txt`](data/amd_mi300x_rocm_smi.txt));
  the honest self-healing GPU-ops layer.
- **Best Use of Gemma:** Gemma is the **multimodal core** — sees rendered structures,
  reads lab-report photos (no OCR), reads H&E slides, **proposes a structural rescue
  then interprets the folded geometry**, holds a spoken conversation — plus offline
  local Gemma via Ollama and the six-persona probability vote.

## Honesty guardrails
- The MI300X numbers (2.8 s fold, 100% GPU, 749 W) are **real captures** — show/narrate
  those. Do **not** fake a VRAM spike on the laptop; the live laptop autonomic panel
  honestly reads *"no AMD GPU present"* and that's fine.
- The rescue beat: pLDDT is **confidence, not stability**; it's an *in-silico
  hypothesis*, never a proven cure. Say so.
- Everything in the flow is genuinely live; nothing is scripted or faked.
- Keep the RUO banner visible — it's a copilot, not a diagnosis.

## If you record in segments (instead of one take)
Cut between **tabs** (natural breaks at each ⏱ boundary above). Start each segment by
clicking the next tab so the joins are clean. You'll stitch them yourself in
**Clipchamp** (built into Windows 11) — I can't assemble video files. One continuous
take is simpler if your nerves allow it.

## Optional B-roll
- The DNA-helix codebase graph (Debug → visual) — 3–5 s establishing shot.
- The README trailer GIF as the intro card.
