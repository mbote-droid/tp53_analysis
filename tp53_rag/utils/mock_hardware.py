"""
============================================================
Precision Onco Africa - Mock Hardware Abstraction Layer
utils/mock_hardware.py
============================================================
A software mock of a portable genomic-sequencer's device API — the way real
firmware teams develop *before* the hardware arrives. It does NOT pretend the
hardware exists; every response is explicitly flagged ``mock: True`` and the
UI labels it a simulated device interface.

What it models honestly: the device *state machine* and its control endpoints
(door lock, sample insertion, barcode scan, optical focus, run progress). This
demonstrates that the platform is built to integrate with edge hardware
through a clean, swappable abstraction — drop in a real driver later and the
rest of the app is unchanged.

Deterministic and pure: state advances through explicit transitions; nothing
is random or time-dependent, so it is fully unit-testable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

# Ordered lifecycle stages of a sequencing run.
STAGES = ["idle", "door_open", "sample_loaded", "barcode_verified",
          "focus_locked", "ready", "sequencing", "complete"]

_STAGE_LABEL = {
    "idle": "Idle — awaiting sample",
    "door_open": "Door open",
    "sample_loaded": "Sample cartridge loaded",
    "barcode_verified": "Barcode verified",
    "focus_locked": "Optical focus locked",
    "ready": "Ready to sequence",
    "sequencing": "Sequencing in progress",
    "complete": "Run complete",
}


@dataclass
class DeviceState:
    stage: str = "idle"
    door_locked: bool = True
    sample_present: bool = False
    barcode: str = ""
    focus_score: float = 0.0      # 0–1, optical focus quality
    temperature_c: float = 24.0
    run_progress: float = 0.0     # 0–1
    errors: List[str] = field(default_factory=list)


class MockSequencer:
    """Deterministic mock of a portable sequencer's control API.

    Every method returns a structured response carrying ``mock: True`` so no
    caller can mistake it for real hardware telemetry.
    """

    def __init__(self) -> None:
        self._s = DeviceState()

    # ── helpers ───────────────────────────────────────────────────
    def _resp(self, ok: bool, message: str) -> Dict:
        return {
            "ok": ok, "message": message, "mock": True,
            "note": "Simulated device interface — no physical hardware attached.",
            "state": self.snapshot(),
        }

    def snapshot(self) -> Dict:
        s = asdict(self._s)
        s["stage_label"] = _STAGE_LABEL.get(self._s.stage, self._s.stage)
        s["stage_index"] = STAGES.index(self._s.stage) if self._s.stage in STAGES else 0
        s["mock"] = True
        return s

    # ── control endpoints (the "device API") ──────────────────────
    def open_door(self) -> Dict:
        self._s.door_locked = False
        self._s.stage = "door_open"
        return self._resp(True, "Door unlocked and opened.")

    def insert_sample(self, barcode: str = "") -> Dict:
        if self._s.door_locked:
            return self._resp(False, "Cannot insert sample: door is locked.")
        self._s.sample_present = True
        self._s.barcode = str(barcode or "").strip()
        self._s.stage = "sample_loaded"
        return self._resp(True, "Sample cartridge detected.")

    def scan_barcode(self) -> Dict:
        if not self._s.sample_present:
            return self._resp(False, "No sample to scan.")
        if not self._s.barcode:
            self._s.errors.append("Unreadable barcode")
            return self._resp(False, "Barcode unreadable — re-seat the cartridge.")
        self._s.stage = "barcode_verified"
        return self._resp(True, f"Barcode {self._s.barcode} verified.")

    def lock_and_focus(self) -> Dict:
        if self._s.stage != "barcode_verified":
            return self._resp(False, "Verify the barcode before focusing.")
        self._s.door_locked = True
        self._s.focus_score = 0.97
        self._s.temperature_c = 37.0
        self._s.stage = "focus_locked"
        return self._resp(True, "Door locked; optical focus locked at 0.97.")

    def arm(self) -> Dict:
        if self._s.stage != "focus_locked":
            return self._resp(False, "Device not focused — cannot arm.")
        self._s.stage = "ready"
        return self._resp(True, "Device armed and ready to sequence.")

    def advance_run(self, fraction: float = 0.25) -> Dict:
        if self._s.stage not in ("ready", "sequencing"):
            return self._resp(False, "Device is not ready to sequence.")
        self._s.stage = "sequencing"
        try:
            step = float(fraction)
        except (TypeError, ValueError):
            step = 0.25
        self._s.run_progress = min(1.0, round(self._s.run_progress + step, 3))
        if self._s.run_progress >= 1.0:
            self._s.stage = "complete"
            return self._resp(True, "Sequencing run complete.")
        return self._resp(True, f"Sequencing… {self._s.run_progress:.0%}.")

    def reset(self) -> Dict:
        self._s = DeviceState()
        return self._resp(True, "Device reset to idle.")

    def pipeline(self) -> List[Dict]:
        """Stage list with reached/active flags — for the control-panel viz."""
        current = STAGES.index(self._s.stage) if self._s.stage in STAGES else 0
        return [{"stage": s, "label": _STAGE_LABEL.get(s, s),
                 "reached": i <= current, "active": i == current}
                for i, s in enumerate(STAGES)]


def run_mock_demo_sequence() -> Dict:
    """Drive the device through a full clean lifecycle (for tests/demo). Returns
    the final telemetry after a complete run."""
    dev = MockSequencer()
    dev.open_door()
    dev.insert_sample(barcode="TP53-DEMO-0001")
    dev.scan_barcode()
    dev.lock_and_focus()
    dev.arm()
    for _ in range(4):
        dev.advance_run(0.25)
    return {"final": dev.snapshot(), "pipeline": dev.pipeline(), "mock": True}
