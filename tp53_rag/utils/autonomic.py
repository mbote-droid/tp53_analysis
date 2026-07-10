"""
============================================================
Precision Onco Africa - Autonomic Resource Manager
utils/autonomic.py
============================================================
A self-healing ops layer: it watches real system memory (psutil, everywhere)
and real AMD GPU telemetry (rocm-smi / amd-smi, when an AMD host is present),
and when memory crosses a threshold it takes a GENUINE reclaim action (garbage
collection + any registered cache/model reclaimers) and logs what it did.

Honesty is the whole point:
  * `system_stats()` is real on any machine.
  * `gpu_stats()` returns REAL rocm-smi/amd-smi numbers only when those tools
    exist (i.e. on an AMD ROCm host such as the AMD Developer Cloud). On a
    non-AMD box it says so plainly — it NEVER fabricates GPU numbers.
  * `self_heal()` performs real reclaim work and reports the before/after
    memory, so the "the system saved itself" demo is genuine, not scripted.

Pure/injectable → unit-testable offline. Never raises.
"""
from __future__ import annotations

import gc
import json
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import log


def system_stats() -> Dict:
    """Real host memory + CPU via psutil. Never raises."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "available": True,
            "ram_used_pct": round(vm.percent, 1),
            "ram_used_gb": round((vm.total - vm.available) / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2),
            "cpu_pct": round(psutil.cpu_percent(interval=0.0), 1),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "note": f"psutil unavailable: {e}"}


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ""
    except Exception as e:  # pragma: no cover
        log.warning(f"{cmd[0]} failed: {e}")
        return None


def gpu_stats() -> Dict:
    """Real AMD GPU telemetry when rocm-smi/amd-smi exist; an honest 'not
    present' otherwise. NEVER fabricates numbers. Never raises."""
    # amd-smi (newer) — JSON.
    raw = _run(["amd-smi", "metric", "--json"])
    src = "amd-smi"
    if raw is None:
        raw = _run(["rocm-smi", "--showuse", "--showmemuse", "--json"])
        src = "rocm-smi"
    if raw is None:
        return {"available": False, "source": None,
                "note": "No AMD GPU / ROCm tools on this host — running "
                        "CPU/API mode. (Real GPU telemetry appears when run on "
                        "an AMD Instinct host, e.g. the AMD Developer Cloud.)"}
    gpus: List[Dict] = []
    try:
        data = json.loads(raw)
        # Both tools nest per-card dicts; extract best-effort util/VRAM.
        cards = data.values() if isinstance(data, dict) else data
        for c in (cards or []):
            if not isinstance(c, dict):
                continue
            blob = json.dumps(c).lower()
            gpus.append({"raw": c,
                         "has_vram": "vram" in blob or "memory" in blob,
                         "has_util": "use" in blob or "util" in blob})
    except Exception:
        # Not JSON (older rocm-smi text) — still real; pass the text through.
        return {"available": True, "source": src, "raw_text": raw[:4000],
                "gpus": [], "note": "Live AMD telemetry (unparsed text form)."}
    return {"available": True, "source": src, "gpus": gpus,
            "note": "Live AMD GPU telemetry."}


class AutonomicManager:
    """Threshold-driven self-healing. Reclaimers are injected (name, callable)
    so the app wires real ones (clear semantic cache, unload pathology model,
    etc.) and tests wire fakes."""

    def __init__(self, ram_threshold_pct: float = 90.0):
        self.threshold = float(ram_threshold_pct)
        self.log: List[Dict] = []

    def status(self) -> Dict:
        sysm = system_stats()
        over = bool(sysm.get("available") and
                    sysm.get("ram_used_pct", 0) >= self.threshold)
        return {"system": sysm, "gpu": gpu_stats(),
                "threshold_pct": self.threshold, "over_threshold": over}

    def self_heal(self,
                  reclaimers: Optional[List[Tuple[str, Callable]]] = None,
                  force: bool = False) -> Dict:
        """Reclaim memory if over threshold (or force=True). Runs gc + any
        injected reclaimers, reports before/after RAM. Never raises."""
        before = system_stats().get("ram_used_pct", 0.0)
        if not force and before < self.threshold:
            return {"triggered": False, "reason": "below threshold",
                    "ram_pct": before, "threshold_pct": self.threshold}
        actions: List[str] = []
        try:
            freed = gc.collect()
            actions.append(f"gc.collect() reclaimed {freed} objects")
        except Exception as e:  # pragma: no cover
            actions.append(f"gc failed: {e}")
        for name, fn in (reclaimers or []):
            try:
                fn()
                actions.append(f"ran reclaimer: {name}")
            except Exception as e:
                actions.append(f"reclaimer {name} failed: {e}")
        after = system_stats().get("ram_used_pct", 0.0)
        entry = {"triggered": True, "ram_before_pct": before,
                 "ram_after_pct": after,
                 "reclaimed_pct": round(before - after, 1), "actions": actions}
        self.log.append(entry)
        return entry
