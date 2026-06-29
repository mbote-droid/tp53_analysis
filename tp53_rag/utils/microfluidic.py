"""
============================================================
Precision Onco Africa - Microfluidic QC Decision Engine
utils/microfluidic.py
============================================================
An intelligent fluidics quality-control policy for a liquid-biopsy workflow:
it watches per-frame quality telemetry from the microfluidic channel and, the
moment it detects an unrecoverable fault (a bubble, an occlusion, collapsing
droplet uniformity), it ABORTS the run early — instead of sequencing a sample
that will only yield garbage — and requests recollection.

The intelligence being demonstrated is the *decision policy*, not image
recognition. The quality telemetry is a simulated input (in production it would
come from on-chip sensors / imaging); this module is honest about that and does
NOT pretend to do computer vision or to diagnose patients. What it shows is a
lab that knows when to stop — saving expensive sequencing compute.

Pure, deterministic and unit-tested. Every result is labelled a simulated QC
workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

DISCLAIMER = ("Simulated liquid-biopsy QC workflow — illustrates the abort/"
              "recollect decision policy, not real imaging or diagnosis.")

# Quality thresholds (0–1 where higher is better, except faults which are bool).
_MIN_FLOW = 0.45            # below this, flow is too weak to sequence
_MIN_DROPLET_UNIFORMITY = 0.6
# Hard faults abort immediately; soft dips lower quality but may continue.


@dataclass
class FrameVerdict:
    index: int
    quality: float
    fault: Optional[str]
    action: str            # continue | abort


def _frame_quality(frame: Dict) -> FrameVerdict:
    """Score one telemetry frame and decide continue/abort. Pure."""
    idx = int(frame.get("index", 0))
    bubble = bool(frame.get("bubble"))
    occluded = bool(frame.get("occlusion"))
    try:
        flow = float(frame.get("flow_rate", 1.0))
    except (TypeError, ValueError):
        flow = 1.0
    try:
        uniform = float(frame.get("droplet_uniformity", 1.0))
    except (TypeError, ValueError):
        uniform = 1.0

    # Hard faults → immediate abort.
    if bubble:
        return FrameVerdict(idx, 0.0, "air bubble in channel", "abort")
    if occluded:
        return FrameVerdict(idx, 0.0, "channel occlusion", "abort")
    if flow < _MIN_FLOW:
        return FrameVerdict(idx, round(flow, 2), "insufficient flow rate", "abort")
    if uniform < _MIN_DROPLET_UNIFORMITY:
        return FrameVerdict(idx, round(uniform, 2),
                            "droplet uniformity collapse", "abort")

    quality = round(min(flow, uniform), 2)
    return FrameVerdict(idx, quality, None, "continue")


def analyze_run(frames: List[Dict], total_planned: Optional[int] = None,
                compute_per_frame_s: float = 4.0) -> Dict:
    """Run the QC policy over a sequence of telemetry frames.

    Aborts at the first hard fault and reports the sequencing compute saved by
    not processing the remaining planned frames. Never empty, never raises.
    """
    frames = frames or []
    planned = total_planned if total_planned is not None else len(frames)
    verdicts: List[FrameVerdict] = []
    decision = "completed"
    abort_at = None

    for i, f in enumerate(frames):
        f = dict(f or {})
        f.setdefault("index", i)
        v = _frame_quality(f)
        verdicts.append(v)
        if v.action == "abort":
            decision = "aborted"
            abort_at = i
            break

    processed = len(verdicts)
    if decision == "aborted":
        frames_saved = max(0, planned - processed)
        compute_saved_s = round(frames_saved * compute_per_frame_s, 1)
        fault = verdicts[-1].fault
        message = (f"Aborted at frame {abort_at} ({fault}) — recollection "
                   f"requested; {frames_saved} frames of sequencing skipped.")
    else:
        frames_saved = 0
        compute_saved_s = 0.0
        message = f"Run passed QC across {processed} frame(s)."

    quals = [v.quality for v in verdicts if v.fault is None]
    mean_q = round(sum(quals) / len(quals), 2) if quals else 0.0
    return {
        "decision": decision,
        "abort_at": abort_at,
        "frames_processed": processed,
        "frames_planned": planned,
        "frames_saved": frames_saved,
        "compute_saved_s": compute_saved_s,
        "mean_quality": mean_q,
        "verdicts": [asdict(v) for v in verdicts],
        "message": message,
        "mock": True,
        "disclaimer": DISCLAIMER,
    }


def demo_scenarios() -> Dict[str, Dict]:
    """Two illustrative runs: a clean pass and a bubble-fault early-abort."""
    good = [{"flow_rate": 0.9, "droplet_uniformity": 0.92} for _ in range(8)]
    faulty = (
        [{"flow_rate": 0.88, "droplet_uniformity": 0.9} for _ in range(2)]
        + [{"bubble": True}]                       # fault at frame 2
        + [{"flow_rate": 0.9, "droplet_uniformity": 0.9} for _ in range(5)]
    )
    return {
        "clean_run": analyze_run(good, total_planned=8),
        "fluidics_fault": analyze_run(faulty, total_planned=8),
    }
