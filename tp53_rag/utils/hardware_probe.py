"""
============================================================
Precision Onco Africa - Compute Hardware Probe
utils/hardware_probe.py
============================================================
Reports, honestly, what compute backend the platform is actually running on:
AMD ROCm GPU, an NVIDIA/CUDA GPU, or CPU-only. It probes the real environment
(ROCm install, torch HIP/CUDA build, env flags) and never claims acceleration
that is not present.

This backs the AMD story with truth rather than decoration: on the AMD
Developer Cloud it will detect and log ROCm; on a local laptop it will honestly
report CPU-only. No fabricated "NPU found" output — the Ryzen AI NPU is a
roadmap target (see ARCHITECTURE.md), and a capability is only reported when it
is genuinely detected.

Pure and dependency-light: torch is optional and probed defensively, so this
runs in the lean cloud image too.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def detect_compute() -> Dict:
    """Probe the host for the real compute backend. Never raises.

    Returns a dict with: accelerator ('amd_rocm'|'nvidia_cuda'|'cpu'),
    rocm (bool), hip_version, cuda (bool), device_name, cpu_only (bool),
    inference_mode, and a human-readable summary.
    """
    info: Dict = {
        "accelerator": "cpu",
        "rocm": False,
        "hip_version": None,
        "cuda": False,
        "device_name": None,
        "cpu_only": True,
        "inference_mode": os.getenv("INFERENCE_MODE", "ollama"),
        "torch": None,
    }

    # 1. ROCm install hints from the environment (work even without torch).
    rocm_env = bool(
        os.getenv("ROCM_PATH")
        or os.getenv("HIP_VISIBLE_DEVICES")
        or os.getenv("HSA_OVERRIDE_GFX_VERSION")
        or Path("/opt/rocm").exists()
    )

    # 2. torch is the authoritative probe — but it is optional (lean image).
    try:
        import torch
        info["torch"] = torch.__version__
        hip = getattr(torch.version, "hip", None)
        if hip:
            info["hip_version"] = hip
        if torch.cuda.is_available():
            # On a ROCm build, torch.cuda is the ROCm device interface.
            info["device_name"] = torch.cuda.get_device_name(0)
            if hip:
                info["accelerator"] = "amd_rocm"
                info["rocm"] = True
            else:
                info["accelerator"] = "nvidia_cuda"
                info["cuda"] = True
            info["cpu_only"] = False
    except Exception:
        # torch absent or failed — fall back to env hints only.
        if rocm_env:
            info["rocm"] = True  # ROCm present in env even if torch isn't

    # ROCm env without torch: report ROCm available but note CPU execution.
    if rocm_env and not info["rocm"] and info["cpu_only"]:
        info["rocm"] = True

    info["summary"] = _summarise(info)
    return info


def _summarise(info: Dict) -> str:
    mode = info.get("inference_mode", "ollama")
    if info.get("accelerator") == "amd_rocm":
        dev = info.get("device_name") or "AMD GPU"
        return (f"AMD ROCm acceleration detected ({dev}, HIP "
                f"{info.get('hip_version')}). Inference mode: {mode}.")
    if info.get("accelerator") == "nvidia_cuda":
        return (f"NVIDIA CUDA GPU detected ({info.get('device_name')}). "
                f"Inference mode: {mode}.")
    if info.get("rocm"):
        return ("ROCm runtime present but no GPU device active in torch — "
                f"running CPU-side. Inference mode: {mode}.")
    return (f"No GPU accelerator detected — running CPU-only "
            f"(offline-capable). Inference mode: {mode}.")


def log_compute_banner(logger) -> Dict:
    """Log the honest compute banner at startup and return the probe result.

    Uses the given logger (loguru or stdlib). The Ryzen AI NPU is intentionally
    NOT probed/claimed here — it is a roadmap target, not a runtime capability.
    """
    info = detect_compute()
    try:
        if info.get("accelerator", "cpu") != "cpu":
            logger.info(f"[compute] {info['summary']}")
        else:
            logger.info(f"[compute] {info['summary']}")
    except Exception:
        pass
    return info
