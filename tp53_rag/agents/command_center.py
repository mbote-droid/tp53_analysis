"""
============================================================
TP53 RAG Platform - African Oncology Command Center
agents/command_center.py
============================================================
Aggregates the regional African TP53 atlas into a single decision-support
snapshot: continental KPIs, per-region analytics (dominant cancers, key
mutations, environmental drivers) and an honest resource/access layer for
planners working in low-resource settings.

Builds on the curated AfricanTP53Atlas — no new epidemiological claims are
invented here; this is an aggregation and presentation layer. Pure, offline,
never-empty.

DISCLAIMER: curated regional epidemiology for research and planning — not
individual patient data and not a substitute for national cancer registries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from utils.logger import log
from agents.african_atlas import atlas_profile, _atlas

AGENT_ID = "command_center"
DISCLAIMER = ("Curated regional epidemiology for research and health-planning "
              "use — not individual patient data, not a national registry.")

# Curated, honest resource/access context per region (planning-level).
_ACCESS_NOTES = {
    "West Africa": "Limited radiotherapy capacity; aflatoxin-driven HCC needs "
                   "HBV vaccination + grain-storage public-health measures.",
    "East Africa": "Growing oncology centres (Nairobi, Kampala); pathology and "
                   "molecular testing remain referral-dependent.",
    "Southern Africa": "Stronger registry coverage; HIV-associated malignancy "
                       "burden shapes oncology demand.",
    "North Africa": "Comparatively higher imaging/radiotherapy access; "
                    "bladder/breast burden prominent.",
    "Central Africa": "Sparse oncology infrastructure; cross-border referral "
                      "and tele-oncology are critical.",
}


@dataclass
class CommandCenterSnapshot:
    kpis: Dict[str, int]
    regions: List[Dict] = field(default_factory=list)
    top_drivers: List[str] = field(default_factory=list)
    top_cancers: List[str] = field(default_factory=list)
    country_burden: Dict[str, int] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class OncologyCommandCenter:
    """Aggregate the African atlas into a command-center snapshot."""

    def __init__(self) -> None:
        self._audit_log = Path("logs/command_center.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Command-center audit dir unavailable: {e}")

    def snapshot(self) -> Dict:
        """Continental decision-support snapshot. Never empty."""
        atlas = atlas_profile()  # continental overview (all profiles)
        a = atlas.get("atlas", {})
        profiles = a.get("matched_profiles", [])

        # Per-region aggregation.
        region_map: Dict[str, Dict] = {}
        for p in profiles:
            for r in p.get("regions", []):
                rm = region_map.setdefault(r, {
                    "region": r, "countries": [], "cancers": [],
                    "key_mutations": [], "drivers": []})
                for c in p.get("countries", []):
                    if c not in rm["countries"]:
                        rm["countries"].append(c)
                dc = p.get("dominant_cancer")
                if dc and dc not in rm["cancers"]:
                    rm["cancers"].append(dc)
                for km in p.get("key_mutations", []):
                    if km not in rm["key_mutations"]:
                        rm["key_mutations"].append(km)
                drv = p.get("environmental_driver")
                if drv and drv not in rm["drivers"]:
                    rm["drivers"].append(drv)
        for r, rm in region_map.items():
            rm["access_note"] = _ACCESS_NOTES.get(r, "Resource data not curated "
                                                   "for this region.")

        burden = _atlas.country_burden()
        kpis = {
            "countries": len(a.get("countries", [])),
            "regions": len(region_map),
            "cancers": len(a.get("cancers", [])),
            "key_mutations": len(a.get("key_mutations", [])),
            "drivers": len(a.get("environmental_drivers", [])),
        }

        snap = CommandCenterSnapshot(
            kpis=kpis,
            regions=sorted(region_map.values(), key=lambda x: x["region"]),
            top_drivers=a.get("environmental_drivers", [])[:6],
            top_cancers=a.get("cancers", [])[:6],
            country_burden=burden,
            sources=a.get("sources", []),
        )
        self._audit(f"snapshot: {kpis['regions']} regions, "
                    f"{kpis['countries']} countries")
        return {
            **asdict(snap),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": (f"Command-center snapshot: {kpis['regions']} regions, "
                        f"{kpis['countries']} countries, "
                        f"{kpis['key_mutations']} key mutations"),
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Command-center audit failed: {e}")


_center = OncologyCommandCenter()


def command_center_snapshot() -> Dict:
    """Module-level convenience wrapper."""
    return _center.snapshot()
