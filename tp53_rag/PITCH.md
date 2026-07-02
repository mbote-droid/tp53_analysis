# Pitch Kit — Slides + Video Script

Everything you need to record the **Video Presentation** and build the **Slide
Presentation** required by the AMD ACT II submission. Written to the Track 3
rubric: creativity, product/market, completeness, use of AMD platforms.

Golden rule: **lead with Amara and the tumour board. Show it working. Quantify
the AMD benefit. Keep it honest** (copilot, not a diagnostic device).

---

## A. Slide deck outline (10 slides)

Keep to ~10 slides, one idea each. Judges reward clarity over polish.

1. **Title / cover.** "Precision Onco Africa — a transparent clinical copilot
   for TP53 cancer genomics." Tagline: *"Six AI specialists debate your case —
   offline, on AMD."* Cover image = the tumour-board consensus screen.

2. **The problem (Amara).** One human: a medical officer at a Kenyan district
   hospital, no oncologist, no geneticist, no internet, a TP53 mutation on the
   bench. Stat: **TP53 is mutated in ~50% of all cancers.** Stat: the oncologist/
   geneticist shortage across sub-Saharan Africa.

3. **The idea.** A precision-oncology *operating environment* that convenes a
   virtual tumour board and explains itself — built to run where specialists
   and connectivity don't exist.

4. **Demo screenshot — Tumour Board.** The six specialists + consensus +
   confidence. Caption: *"debate → vote → consensus, with earned confidence."*

5. **Demo screenshot — Explainability ("Why?").** Evidence trace: classification
   → ClinVar → ESM-2 → pathways → citations → "what we don't know."

6. **Trust / honesty.** Dual guardrails, ClinVar cross-check, RUO framing, "never
   fabricates." One line: *"the only medical AI that tells you when it's unsure."*

7. **Use of AMD platforms.** Fireworks (AMD-hosted models) + AMD Developer Cloud
   + ROCm/vLLM benchmark. Show the **local-vs-AMD latency number** once you have
   it. One line: *"the debate runs in real time on AMD Instinct."*

8. **Product / market.** Target user (Amara), TAM framing, revenue model
   (per-site SaaS / API / NGO-funded). "Gene-agnostic — TP53 first, not last."

9. **Completeness.** 26 agents (25 deterministic), 419 tests, benchmarked vs
   ClinVar/IARC, containerized, offline-first, deployed. "Not a mock-up."

10. **Roadmap + ask.** Multi-gene expansion (BRCA/KRAS/EGFR), pilot with an
    African oncology centre, heterogeneous edge deployment. Close on the mission.

**Design notes:** dark theme to match the app; big screenshots, few words;
every AMD-related slide gets the AMD logo/wordmark; put the demo URL + repo on
the last slide.

---

## B. Video script (~4 minutes)

> Structure mirrors the winning format: 0:30 problem → 2:00 live demo →
> 1:30 market + AMD + impact → close. Record the demo section **first** and
> re-record if anything stutters. Run the demo in **`api`/`fireworks` mode** so
> it's fast, and use the streaming answer so it feels real-time.

### 0:00–0:30 — The problem (Amara)
> "Meet Amara. She's a doctor at a district hospital in western Kenya. There's
> no oncologist on staff, no geneticist, and the internet comes and goes. Today
> a patient's biopsy came back with a mutation in TP53 — the gene that's broken
> in nearly half of all cancers. Amara has minutes, not hours, and no specialist
> team to ask. This is the problem we built for."

### 0:30–2:30 — Live demo (the heart of it)
> "This is Precision Onco Africa. I'll enter the patient's mutation — R175H."
>
> *(Analysis tab: show the variant land on the protein needle plot.)*
> "Instantly it's mapped onto the p53 protein, with the real evidence — ClinVar,
> the functional scores — pulled in."
>
> *(Tumour Board tab: click Convene.)*
> "Now the part I love. Six AI specialists — a pathologist, geneticist,
> oncologist, surgeon, pharmacologist, and an equity officer — each give their
> read, challenge each other, and **vote toward a consensus.** And notice the
> confidence: it's *earned*. For a well-known hotspot like this, it's high. Give
> it an uncertain variant, and it honestly tells you it's not sure."
>
> *(Scroll to Explainability.)*
> "Every recommendation answers *why* — the classification, ClinVar, the ESM-2
> language-model score, the pathways it disrupts, the citations, and crucially,
> what we *don't* know."
>
> *(Point to the equity note + Kiswahili button.)*
> "And because this is built for Amara, it flags what's realistically available
> locally, and generates the patient explanation in Kiswahili. All of this runs
> **offline** — no internet required."

### 2:30–3:15 — Use of AMD
> "Under the hood, the heavy reasoning runs on **AMD** — open models served on
> **AMD Instinct GPUs through Fireworks**, with the protein-language-model
> compute benchmarked on **AMD Developer Cloud with ROCm.** Moving inference
> from a local CPU to AMD Instinct cut our answer latency from **[X] to [Y]
> seconds** — which is what makes the multi-agent debate feel real-time."
> *(Show the AMD benchmark panel.)*

### 3:15–4:00 — Market, honesty, and the close
> "TP53 is first because it's in half of all cancers — but the architecture is
> gene-agnostic: BRCA, KRAS, EGFR are next. Our user is every clinician who
> faces cancer without a specialist down the hall — a market the big vendors
> ignore because it demands offline, low-cost, equity-aware AI.
>
> One last thing that matters in medicine: this is a **copilot, not a diagnosis**.
> It's transparent, it cites its evidence, and it tells you when it's unsure.
> That's how you earn a clinician's trust.
>
> Precision Onco Africa — bringing the tumour board to the places that never had
> one. Thank you."

---

## C. Recording checklist
- [ ] Run the app in `fireworks` (or `api`) mode — fast, no stutter.
- [ ] Pre-load the R175H case so the demo path is flawless (see [DEMO.md](DEMO.md)).
- [ ] Fill the `[X]/[Y]` latency numbers from the real AMD benchmark.
- [ ] Cover image = the consensus screen; keep total video ≤ 4–5 min.
- [ ] Upload: video + slides + cover image + demo URL + repo link to lablab.
