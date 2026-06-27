"""
============================================================
TP53 RAG Platform - Agent Evaluation Harness
benchmarks/agent_eval.py
============================================================
Continuously benchmarks the deterministic decision agents (tumour board,
explainability) on metrics that can be measured honestly without a live LLM:

  • latency        — mean wall-clock per case
  • success rate   — fraction returning a valid, non-empty result
  • calibration    — does confidence rank hotspots above VUS? (a real quality
                     signal, not a vanity number)
  • citation rate  — fraction of explanations carrying ≥1 citation
  • uncertainty    — fraction of VUS cases that are correctly flagged uncertain

These are deterministic and offline, so the harness runs in CI and in the app
to demonstrate engineering rigor. Nothing is inflated: every metric is derived
from the agents' actual outputs over a fixed case set.
"""
from __future__ import annotations

import time
from typing import Dict, List

# Fixed evaluation cases with known expected character.
EVAL_CASES: List[Dict] = [
    {"mutation": "R175H", "kind": "hotspot"},
    {"mutation": "R248Q", "kind": "hotspot"},
    {"mutation": "R273H", "kind": "hotspot"},
    {"mutation": "R213*", "kind": "truncating"},
    {"mutation": "A159V", "kind": "vus"},
    {"mutation": "T125T", "kind": "vus"},
    {"mutation": "???",    "kind": "unknown"},
]


def _time_call(fn, *args) -> tuple:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1000.0


def evaluate_tumor_board(cases: List[Dict] = None) -> Dict:
    """Evaluate the tumour board over the case set."""
    from agents.tumor_board import convene_tumor_board
    cases = cases or EVAL_CASES
    latencies, successes = [], 0
    hotspot_conf, vus_conf = [], []
    for c in cases:
        out, ms = _time_call(convene_tumor_board, c["mutation"],
                             {"cancer": "Breast", "stage": "II"})
        latencies.append(ms)
        ok = bool(out.get("members")) and bool(out.get("consensus", {}).get(
            "recommendation"))
        successes += int(ok)
        conf = out.get("consensus", {}).get("confidence", 0.0)
        if c["kind"] == "hotspot":
            hotspot_conf.append(conf)
        elif c["kind"] == "vus":
            vus_conf.append(conf)
    calibrated = (bool(hotspot_conf) and bool(vus_conf)
                  and (sum(hotspot_conf) / len(hotspot_conf)
                       > sum(vus_conf) / len(vus_conf)))
    return {
        "agent": "tumor_board",
        "cases": len(cases),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "success_rate": round(successes / len(cases), 3),
        "calibrated": calibrated,
        "mean_hotspot_confidence": round(
            sum(hotspot_conf) / len(hotspot_conf), 3) if hotspot_conf else None,
        "mean_vus_confidence": round(
            sum(vus_conf) / len(vus_conf), 3) if vus_conf else None,
    }


def evaluate_explainability(cases: List[Dict] = None) -> Dict:
    """Evaluate the explainability engine over the case set."""
    from agents.explainability import explain_variant
    cases = cases or EVAL_CASES
    latencies, successes, with_citations = [], 0, 0
    vus_total, vus_flagged = 0, 0
    for c in cases:
        out, ms = _time_call(explain_variant, c["mutation"])
        latencies.append(ms)
        successes += int(bool(out.get("evidence")))
        with_citations += int(bool(out.get("citations")))
        if c["kind"] == "vus":
            vus_total += 1
            flagged = any("uncertain" in u.lower() or "not established" in u.lower()
                          for u in out.get("uncertainty", []))
            vus_flagged += int(flagged)
    return {
        "agent": "explainability",
        "cases": len(cases),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "success_rate": round(successes / len(cases), 3),
        "citation_rate": round(with_citations / len(cases), 3),
        "vus_uncertainty_flag_rate": (round(vus_flagged / vus_total, 3)
                                      if vus_total else None),
    }


def run_agent_eval() -> Dict:
    """Run the full agent-evaluation suite. Never empty."""
    results = [evaluate_tumor_board(), evaluate_explainability()]
    return {
        "agents": results,
        "agent_count": len(results),
        "cases": len(EVAL_CASES),
        "all_passing": all(r["success_rate"] >= 0.99 for r in results),
    }
