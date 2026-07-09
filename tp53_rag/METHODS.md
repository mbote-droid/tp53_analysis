# Methods — Mathematical & Biophysical Formulations

This document sets out the quantitative methods the platform actually uses.
Every formula below corresponds to code in the repository; none are decorative.
Where a method is a heuristic rather than a first-principles model, it is said
so plainly.

---

## 1. Protein language-model variant effect (ESM-2)

Each missense substitution is scored by the masked-marginal log-likelihood
ratio under ESM-2 — the model's relative preference for the mutant versus the
wild-type residue in its sequence context:

$$
\text{LLR}(wt \rightarrow mut) = \log P\big(x_i = mut \mid x_{\setminus i}\big) - \log P\big(x_i = wt \mid x_{\setminus i}\big)
$$

where position $i$ is masked and $x_{\setminus i}$ is the surrounding sequence.
More negative ⇒ more deleterious. The matrix is precomputed once on a GPU host
and served offline (`utils/variant_effect.py`, `tools/precompute_esm2.py`).
Labelling thresholds are configurable (`ESM2_THRESH_*`).

## 2. Hybrid retrieval score

Retrieval combines dense semantic similarity with sparse exact-term matching
(important for tokens like `R175H`, `MDM2`, `APR-246`):

$$
S(q, d) = \alpha\, S_{\text{vector}}(q, d) + (1-\alpha)\, S_{\text{BM25}}(q, d),
\qquad \alpha = 0.7
$$

The BM25 component (`agents/rag_chain.py`, `class BM25`) is the standard
Okapi form:

$$
\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t)\,
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\big(1 - b + b\,\frac{|d|}{\text{avgdl}}\big)}
$$

with $k_1 = 1.5$, $b = 0.75$, and
$\text{IDF}(t) = \ln\!\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1$.

## 3. Semantic cache hit

A new query reuses a cached answer when its embedding is sufficiently close, by
cosine similarity:

$$
\cos(\mathbf{q}, \mathbf{q}') = \frac{\mathbf{q}\cdot\mathbf{q}'}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{q}'\rVert} \;\geq\; 0.92
$$

(`agents/rag_chain.py`, `SemanticCache`). Hits cost zero LLM tokens — see §7.

## 4. Tumour-board consensus confidence

Each board member casts a confidence-weighted vote; the consensus confidence is
the mean confidence of the winning bloc, scaled by how unified the panel is:

$$
c_{\text{consensus}} = \bar{c}_{\text{backers}} \cdot \big(0.5 + 0.5\, r_{\text{agree}}\big),
\qquad r_{\text{agree}} = \frac{|\text{backers}|}{|\text{members}|}
$$

so a split panel is reported as *less* certain than a unanimous one
(`agents/tumor_board.py`). This is a calibration heuristic, not a probability.

## 5. Structural destabilisation (ΔΔG)

Mutation impact on fold stability is expressed as the change in folding free
energy — genuine biophysics (thermodynamics of protein stability):

$$
\Delta\Delta G = \Delta G_{\text{mut}} - \Delta G_{\text{wt}} \quad [\text{kcal·mol}^{-1}]
$$

Positive $\Delta\Delta G$ ⇒ destabilising. Values are curated from the structural
literature (`agents/structural_analyzer.py`); they are not computed from a
force field on-device and are labelled accordingly.

## 6. AlphaFold per-residue confidence (pLDDT)

Structures carry a per-residue confidence $\text{pLDDT}\in[0,100]$, banded as:

$$
\text{band} = \begin{cases}
\text{very high} & \text{pLDDT} > 90\\
\text{confident} & 70 < \text{pLDDT} \leq 90\\
\text{low} & 50 < \text{pLDDT} \leq 70\\
\text{very low} & \text{pLDDT} \leq 50
\end{cases}
$$

(`utils/alphafold_client.py`). Mean pLDDT over a region summarises model trust.

## 7. Token-efficient routing — measured savings

The router avoids an LLM call when a cache hit or deterministic agent can answer
(`utils/token_router.py`). Over a session the saving is *measured*, not claimed:

$$
T_{\text{saved}} = \sum_{q \in \text{avoided}} \big(\tau_{\text{prompt}} + \tau_{\text{response}}\big),
\qquad
\text{cost}_{\text{saved}} = \frac{T_{\text{saved}}}{1000}\cdot \rho
$$

with token estimate $\tau(\text{text}) \approx \lceil |\text{text}|/4 \rceil$ and
price $\rho$ (USD per 1K tokens). Conservative and transparent by design.

## 8. Microfluidic QC — compute saved by early abort

When the fluidics policy aborts a run at frame $k$ of $N$ planned, the
sequencing compute avoided is

$$
\text{compute saved} = (N - k_{\text{processed}}) \cdot t_{\text{frame}} \;\;[\text{s}]
$$

(`utils/microfluidic.py`). The intelligence demonstrated is the abort decision,
on simulated telemetry — not image recognition.

## 9. Epistemic Uncertainty Index (multi-sample agreement)

For a hosted model whose internals we cannot access, honest uncertainty is the
*disagreement across repeated samples*. Sampling the same prompt $N$ times at
temperature $>0$ and embedding each answer $e_i$, the index is one minus the
mean pairwise cosine similarity:

$$
U = 1 - \frac{2}{N(N-1)} \sum_{i<j} \cos(e_i, e_j)
$$

Bands: green $U<0.15$, amber $0.15\le U<0.35$, red $U\ge0.35$
(`agents/uncertainty.py`). **Computed on real model outputs.** This measures
model-output *agreement* — an epistemic-uncertainty proxy — **not** a clinical
probability.

## 10. Counterfactual trial viability (illustrative heuristic)

Beyond positive trial matches, the adversarial layer counts negative signals
$h_k$ (stopped trials, ClinVar conflicts, resistance) and folds them into a
viability score:

$$
V = S_{\text{match}} \cdot \left(1 - \min\!\Big(1, \sum_k w_k\, h_k\Big)\right),
\qquad h_k = \min\!\Big(1, \tfrac{\text{count}_k}{3}\Big)
$$

where $S_{\text{match}}$ scales with the number of matching recruiting trials
and $\sum_k w_k \le 1$ (`agents/adversarial_evidence.py`). **Illustrative
composite heuristic (RUO)** — a triage/attention signal, **not** a validated
efficacy or survival probability.

## 11. Trust-adjusted retrieval rerank

A transparent multiplicative trust prior is applied to each candidate's
relevance score *before* ranking, so a retracted or superseded source is
down-weighted before it can reach the prompt:

$$
\text{score}' = \text{score}_{\text{rel}} \cdot \tau, \qquad
\tau = \tau_{\text{retraction}} \cdot \tau_{\text{recency}} \cdot \tau_{\text{authority}} \in (0, 1]
$$

$\tau=1$ for ordinary documents (no perturbation to normal retrieval);
$\tau_{\text{retraction}}\!\approx\!0.05$ for retracted content
(`compute_trust_weight`, `agents/rag_chain.py`). The honest, retrieval-layer
version of "epistemic reranking" — no model internals.

## 12. Kiswahili → HPO/ICD-10 mapping with confidence gate

An exact-substring lookup maps known Kiswahili phrases to codes; an unmatched
phrase is mapped to the nearest curated term only if the embedding similarity
clears a threshold, otherwise it is passed through as "no confident mapping":

$$
\text{code} = \operatorname*{arg\,max}_{p}\; \cos\!\big(\text{emb}(\text{phrase}), \text{emb}(\text{term}_p)\big)
\quad\text{accepted iff}\quad \max_p \cos \ge \theta\;(\approx 0.60)
$$

(`agents/kiswahili_hpo.py`). The threshold is a safety gate against a
wrong-but-confident clinical-code mapping; the table is a curated,
clinically-reviewable starter set.

---

### A note on honesty

Some items above are curated or simulated inputs (ΔΔG, microfluidic telemetry),
and one is an explicitly illustrative heuristic (§10 viability), and they say
so. The rest operate on real model outputs (§1, §9) or real retrieval
(§2, §11). Keeping that line clear — computed vs curated vs simulated vs
illustrative-heuristic — is itself part of the methodology.
