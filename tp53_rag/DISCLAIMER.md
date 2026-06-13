# Disclaimer — Research Use Only (RUO)

**TP53 RAG Platform · Daktari Genomed Labs**

## Intended use

This software is a **research and decision-support tool for qualified
researchers and clinicians**. It is intended to assist with the exploration and
interpretation of TP53 cancer-genomics information.

## It is NOT a medical device

- It is **not FDA-cleared, not CE-marked, and not certified** for clinical
  diagnostic use.
- It **must not** be used as the sole basis for any diagnosis, prognosis, or
  treatment decision.
- All outputs — including variant classifications, drug-discovery results,
  docking affinities, structural estimates (ΔΔG, cavity druggability), staging,
  and generated reports — are **informational** and may be incomplete,
  approximate, or incorrect.

## Confirm before acting

Any clinically actionable finding must be **independently confirmed** by a
CLIA-certified / accredited laboratory and reviewed by a qualified clinician
before it informs patient care. Drug-availability notes (e.g. Kenya Essential
Medicines List context) are guidance only and must be verified locally.

## Data handling

- Do **not** enter real, identifiable patient data into any public/hosted
  instance of this application. In cloud (`INFERENCE_MODE=api`) deployments,
  queries are sent to an external model provider and **do not stay on-device**.
- The platform includes PII-scrubbing safeguards, but these are
  **risk-reducing, not guarantees**. Institutions remain responsible for their
  own privacy/compliance review (HIPAA/GDPR/local law). The platform is
  **HIPAA-aligned by design but not independently certified**.

## Illustrative vs. real data

Some computations are **modelled approximations** (e.g. heuristic docking
affinities, ΔΔG estimates) and are labelled as such in the interface. They are
not laboratory measurements.

## No warranty

This software is provided "as is", without warranty of any kind, express or
implied. The authors and Daktari Genomed Labs accept no liability for any use
of, or reliance on, its outputs.

---

*By using this software you acknowledge and accept the terms of this
disclaimer.*
