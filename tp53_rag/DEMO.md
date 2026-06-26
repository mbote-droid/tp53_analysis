# Guided Demo — The Patient Journey (≈60 seconds)

> A single illustrative case walked end-to-end, so a first-time viewer
> understands the platform in under a minute.
>
> **The patient is fictional. All outputs are research-use-only and not a
> clinical determination.**

---

## The story we tell

> *"Amara, 34, presents with an early-stage breast tumour at a regional clinic
> with no on-site oncologist or genetic counsellor. Her biopsy is sequenced.
> Watch the platform take her from a raw variant to a defensible, equity-aware
> plan — in under a minute."*

This is the spine of the demo. Every click below advances Amara's story.

---

## The 60-second run

| # | Time | Tab | Action | What the audience sees |
|---|------|-----|--------|------------------------|
| 1 | 0:00 | **Analysis** | Upload the sample VCF (or type `R175H`) | TP53 variant auto-extracted; the **needle/lollipop plot** drops the mutation onto the protein over its domains — instantly legible |
| 2 | 0:12 | **Analysis** | Open the annotation panel | Real curated annotation: ClinVar, SIFT/PolyPhen, CADD, gnomAD — the evidence base, no jargon needed to feel it |
| 3 | 0:22 | **⭐ Tumour Board** | Click **Convene the board** | Six AI specialists appear one by one, each with a stance and a **confidence meter**; they cross-examine, then **vote to a consensus** — the memorable moment |
| 4 | 0:42 | **⭐ Tumour Board** | Scroll to **Why?** | The Explainability trace: every claim backed by classification, ClinVar, ESM-2, pathways, citations — and an honest *"what we don't know"* |
| 5 | 0:54 | **Tumour Board** | Note the **Equity Officer** card + export | The differentiator: a plan aware of what's actually available locally, exportable as an RUO-stamped report |

End on the consensus banner. That single screen — six specialists, a recommendation, a confidence, and a "why" — is the whole pitch.

---

## Why this order works

- **Starts concrete** (a real variant on a real protein) — no abstract setup.
- **The wow lands by 0:30** (the board debating) — within the first-minute window judges remember.
- **Trust is earned, not claimed** — the "Why?" panel pre-empts the black-box objection.
- **Ends on the differentiator** — equity-aware care for low-resource settings, the thing no generic oncology bot does.

---

## Presenter notes

- Run in **`INFERENCE_MODE=api`** for snappy narration if any live LLM text is shown; the board and explainability traces are deterministic and need no model.
- If offline, everything in the 60-second path still works — that *is* part of the story (offline cancer copilot).
- Keep talking over the board reveal; let the consensus banner land in silence.
- Say "think of this as a digital twin of the case" if you like — but never claim it predicts *this patient's* survival. It explores evidence; it does not prognosticate an individual.
