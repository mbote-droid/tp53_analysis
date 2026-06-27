"""
============================================================
Precision Onco Africa - AMD Benchmark Loader & Deployment Map
utils/amd_benchmark.py
============================================================
Serves the results of a real benchmark run on AMD hardware (produced once by
`tools/benchmark_amd.py` on the AMD Developer Cloud) and the honest deployment
map shown in the UI.

Design mirrors the precompute-and-serve pattern used elsewhere: the heavy run
happens on capable hardware and writes a small JSON; the app loads and displays
it with **no torch/ROCm dependency at runtime**. When the JSON is absent the
loader reports an honest "not yet run" state — numbers are never fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from utils.logger import log

BENCHMARK_PATH = Path("data/amd_benchmark.json")

# Honest deployment map. "current" = actually runnable today; "future" =
# credible roadmap targets, NEVER claimed as functioning. This is the
# measure-real / diagram-aspirational rule made concrete.
DEPLOYMENT_TIERS = {
    "current": [
        {"target": "AMD Developer Cloud (Instinct GPU)", "via": "ROCm + PyTorch",
         "use": "ESM-2 / structure precompute, batch inference"},
        {"target": "Fireworks AI (AMD Instinct)", "via": "OpenAI-compatible API",
         "use": "Hosted LLM generation (INFERENCE_MODE=fireworks)"},
        {"target": "Local CPU", "via": "Ollama / llama.cpp",
         "use": "Fully offline operation in low-connectivity settings"},
        {"target": "Cloud API", "via": "Google AI Studio",
         "use": "Latency-sensitive demos when local compute is thin"},
    ],
    "future": [
        {"target": "Ryzen AI laptops (NPU)", "via": "ONNX Runtime / Vitis AI",
         "use": "On-device clinician copilot at the point of care"},
        {"target": "AMD Kria K26 (FPGA)", "via": "Vitis AI",
         "use": "Genomic preprocessing / variant-calling acceleration"},
        {"target": "Hospital edge servers", "via": "ROCm on-prem",
         "use": "Private, in-building inference for PHI-sensitive data"},
    ],
}


def load_benchmark(path: Optional[Path] = None) -> Dict:
    """Load the AMD benchmark JSON. Never raises; returns an honest
    'available: False' state when the file is absent or malformed."""
    p = Path(path) if path is not None else BENCHMARK_PATH
    if not p.exists():
        return {"available": False,
                "reason": "Benchmark not yet run on AMD hardware.",
                "path": str(p)}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("runs"):
            return {"available": False,
                    "reason": "Benchmark file present but empty/invalid.",
                    "path": str(p)}
        data["available"] = True
        return data
    except Exception as e:  # pragma: no cover
        log.warning(f"AMD benchmark load failed: {e}")
        return {"available": False, "reason": f"Could not read benchmark: {e}",
                "path": str(p)}
