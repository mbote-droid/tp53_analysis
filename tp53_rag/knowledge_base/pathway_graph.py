"""
============================================================
Precision Onco Africa - TP53 Pathway Graph (GraphRAG-lite)
knowledge_base/pathway_graph.py
============================================================
Standard RAG retrieves prose; the biological *relationships* between p53 and
its targets get lost in it. This module encodes well-established TP53 pathway
relationships as explicit (subject → predicate → object) triples and renders
them as retrievable chunks, so a single vector lookup returns both the
semantic text AND the relational structure — GraphRAG-lite, no separate graph
database, no custom kernel.

Honest: every triple is a textbook-level, well-established relationship;
nothing here is inferred or invented. Rendered as normal documents that flow
through the same embedding + retrieval path as the rest of the knowledge base.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (subject, predicate, object, mechanism/effect) — established TP53 biology.
TP53_PATHWAY_TRIPLES: List[Tuple[str, str, str, str]] = [
    ("TP53", "transactivates", "CDKN1A (p21)", "G1/S cell-cycle arrest"),
    ("TP53", "transactivates", "BAX", "intrinsic apoptosis"),
    ("TP53", "transactivates", "PUMA (BBC3)", "intrinsic apoptosis"),
    ("TP53", "transactivates", "NOXA (PMAIP1)", "intrinsic apoptosis"),
    ("TP53", "transactivates", "MDM2", "negative-feedback autoregulation"),
    ("TP53", "transactivates", "GADD45A", "DNA repair / growth arrest"),
    ("TP53", "transactivates", "SFN (14-3-3σ)", "G2/M arrest"),
    ("TP53", "transactivates", "TIGAR", "metabolic regulation, lowers ROS"),
    ("TP53", "represses", "BCL2", "tips balance toward apoptosis"),
    ("MDM2", "ubiquitinates / degrades", "TP53", "keeps p53 low in unstressed cells"),
    ("CDKN2A (p14ARF)", "inhibits", "MDM2", "stabilises p53 under oncogenic stress"),
    ("ATM", "phosphorylates", "TP53", "DNA double-strand-break response"),
    ("CHK2", "phosphorylates", "TP53", "stabilises p53 after DNA damage"),
    ("MDM2 inhibitors (nutlin, idasanutlin)", "block", "MDM2–TP53 interaction",
     "restore wild-type p53 activity"),
    ("APR-246 / PRIMA-1", "refolds / reactivates", "mutant TP53",
     "restores transcriptional function in some conformational mutants"),
]


def triple_text(t: Tuple[str, str, str, str]) -> str:
    s, p, o, note = t
    return f"{s} —{p}→ {o} ({note})."


def pathway_enrichment_text() -> str:
    """One combined, retrievable block of the p53 relationship graph."""
    lines = ["TP53 pathway relationship graph (subject → relation → target):"]
    lines += [f"- {triple_text(t)}" for t in TP53_PATHWAY_TRIPLES]
    return "\n".join(lines)


def pathway_documents() -> List[Dict]:
    """Return the pathway graph as document dicts ({content, metadata}) ready to
    become LangChain Documents. One combined graph doc + per-target-theme docs
    so both broad and specific queries surface the relations."""
    docs: List[Dict] = [{
        "content": pathway_enrichment_text(),
        "metadata": {"source": "pathway_graph", "category": "pathway_graph",
                     "gene": "TP53", "priority": "high",
                     "offline_available": True,
                     "ingestion_source": "curated_embedded"},
    }]
    # Group triples by effect so an apoptosis / arrest / repair query hits a
    # focused chunk that still carries the relations.
    by_effect: Dict[str, List[Tuple]] = {}
    for t in TP53_PATHWAY_TRIPLES:
        key = t[3].split(",")[0].strip()
        by_effect.setdefault(key, []).append(t)
    for effect, ts in by_effect.items():
        if len(ts) < 2:
            continue
        body = "\n".join(f"- {triple_text(t)}" for t in ts)
        docs.append({
            "content": f"TP53 relationships involving {effect}:\n{body}",
            "metadata": {"source": "pathway_graph", "category": "pathway_graph",
                         "gene": "TP53", "effect": effect,
                         "offline_available": True,
                         "ingestion_source": "curated_embedded"},
        })
    return docs
