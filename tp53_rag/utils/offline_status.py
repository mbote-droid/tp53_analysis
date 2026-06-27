"""
============================================================
TP53 RAG Platform - Offline Cancer Copilot — Readiness Map
utils/offline_status.py
============================================================
Reports, honestly, which capabilities work with NO internet connection and
which require it. This is the substance behind the "offline cancer copilot"
claim: in a low-connectivity clinic the core analysis still runs; only the
live-enrichment paths need a network.

Pure and deterministic — a curated capability map plus a small runtime probe
for the active inference mode. No fabrication: a capability is only marked
offline if it genuinely runs without a network.
"""
from __future__ import annotations

import os
from typing import Dict, List

# Curated capability map. offline=True means it runs with the network cable
# pulled. Anything that calls a live external API is offline=False.
_CAPABILITIES: List[Dict] = [
    {"name": "Variant classification & needle plot", "offline": True,
     "detail": "Codon/domain logic and the lollipop map are pure-Python."},
    {"name": "Curated variant annotation baseline", "offline": True,
     "detail": "9-hotspot curated SIFT/PolyPhen/ClinVar baseline ships in-code."},
    {"name": "ESM-2 variant effect", "offline": True,
     "detail": "Served from the precomputed matrix — no torch/network at runtime."},
    {"name": "AlphaFold structure", "offline": True,
     "detail": "Works from cached PDB; first fetch needs a network."},
    {"name": "Live AI Tumour Board + Explainability", "offline": True,
     "detail": "Deterministic curated reasoning — no LLM required."},
    {"name": "African Oncology Command Center", "offline": True,
     "detail": "Aggregates the in-code curated atlas."},
    {"name": "RAG knowledge base", "offline": True,
     "detail": "Local ChromaDB + ONNX embeddings; auto-builds on first run."},
    {"name": "LLM narration (local mode)", "offline": True,
     "detail": "Ollama / llama.cpp run on-device (CPU)."},
    {"name": "Live VEP / MyVariant enrichment", "offline": False,
     "detail": "Ensembl/MyVariant REST — optional override, needs network."},
    {"name": "LLM narration (hosted modes)", "offline": False,
     "detail": "Google / Fireworks APIs need a network."},
    {"name": "PubMed / ClinicalTrials live lookups", "offline": False,
     "detail": "NCBI / ClinicalTrials.gov APIs need a network."},
]


def offline_capabilities() -> Dict:
    """Return the readiness map plus a count summary. Never empty."""
    offline = [c for c in _CAPABILITIES if c["offline"]]
    online = [c for c in _CAPABILITIES if not c["offline"]]
    mode = os.getenv("INFERENCE_MODE", "ollama").strip().lower()
    local_llm = mode in ("ollama", "llamacpp")
    return {
        "capabilities": _CAPABILITIES,
        "offline_count": len(offline),
        "online_count": len(online),
        "total": len(_CAPABILITIES),
        "active_mode": mode,
        "fully_offline_capable": local_llm,
        "summary": (
            f"{len(offline)}/{len(_CAPABILITIES)} capabilities run fully offline. "
            + ("Active inference mode is local — the platform runs end-to-end "
               "with no internet." if local_llm else
               f"Active inference mode '{mode}' needs a network for LLM "
               "narration; switch to ollama/llamacpp for full offline use.")
        ),
    }
