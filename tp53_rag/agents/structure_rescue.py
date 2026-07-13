"""
============================================================
Precision Onco Africa - In-Silico Structural Rescue
agents/structure_rescue.py
============================================================
The "virtual wet-lab" feature. Gemma (the architect) proposes a second-site
suppressor for an oncogenic p53 mutation; a real generative fold is run on an
AMD Instinct MI300X (ESMFold, facebook/esmfold_v1); and the geometry is measured
so Gemma can reason about the result.

This module is the *local-first* consumer of that heavy compute: the folds are
precomputed once on the GPU and committed under ``data/esmfold/`` (mirroring the
ESM-2 precompute pattern), so the app loads real structures without needing a GPU
at runtime and degrades gracefully when they are absent.

HONESTY (non-negotiable):
  * pLDDT is ESMFold's per-residue CONFIDENCE in its own prediction — NOT a
    thermodynamic stability score. We never present it as stability.
  * RMSD is a Kabsch-aligned Cα deviation — the structural "warp".
  * All numbers are whatever ESMFold actually produced. This is an in-silico
    research hypothesis, not therapeutic evidence (RUO).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import log

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "esmfold"
VARIANTS = ("wt", "r175h", "r175h_n239y")
MUTATION = "R175H"
RESCUE_CANDIDATE = "N239Y"

DISCLAIMER = (
    "In-silico structural hypothesis — ESMFold (facebook/esmfold_v1) prediction "
    "run on AMD Instinct. pLDDT is model CONFIDENCE, not thermodynamic stability. "
    "Research use only; not therapeutic evidence."
)

RESCUE_SYSTEM = (
    "You are a structural biologist. The p53 DNA-binding domain carries the "
    "oncogenic {mutation} mutation, which destabilises the core. Propose ONE "
    "plausible second-site 'suppressor' mutation that might re-stabilise the fold. "
    "Reply with the single mutation code (e.g. N239Y) followed by one sentence of "
    "structural mechanism. State clearly this is a research hypothesis."
)


def _pdb_path(variant: str) -> Path:
    return DATA_DIR / f"p53_{variant}.pdb"


def structures_available() -> bool:
    """True only when all three precomputed ESMFold structures are present."""
    return all(_pdb_path(v).exists() for v in VARIANTS)


def load_pdb(variant: str) -> Optional[str]:
    p = _pdb_path(variant)
    if not p.exists():
        return None
    try:
        return p.read_text()
    except OSError as e:
        log.warning(f"structure_rescue: cannot read {p}: {e}")
        return None


def parse_ca(pdb_text: str) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Return (Cα coordinates, per-residue pLDDT) parsed from PDB ATOM records.
    pLDDT is read from the B-factor column; ESMFold writes it on a 0–1 scale."""
    coords: List[Tuple[float, float, float]] = []
    plddt: List[float] = []
    for line in (pdb_text or "").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                coords.append((float(line[30:38]), float(line[38:46]),
                               float(line[46:54])))
                plddt.append(float(line[60:66]))
            except ValueError:
                continue
    return coords, plddt


def mean_plddt(pdb_text: str) -> float:
    """Mean pLDDT on a 0–100 scale (normalising ESMFold's 0–1 B-factors)."""
    _, p = parse_ca(pdb_text)
    if not p:
        return 0.0
    m = sum(p) / len(p)
    return round(m * 100 if m <= 1.5 else m, 1)


def kabsch_rmsd(P: List, Q: List) -> float:
    """Kabsch-aligned RMSD over Cα coordinates (the structural 'warp')."""
    import numpy as np
    A, B = np.asarray(P, float), np.asarray(Q, float)
    n = min(len(A), len(B))
    if n == 0:
        return 0.0
    A, B = A[:n], B[:n]
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return round(float(np.sqrt(((Ac @ R.T - Bc) ** 2).sum(1).mean())), 3)


def _honest_verdict(warp: float, plddt: Dict[str, float]) -> str:
    delta = round(plddt.get("r175h_n239y", 0) - plddt.get("r175h", 0), 1)
    trend = ("essentially unchanged" if abs(delta) < 2
             else "modestly higher" if delta > 0 else "modestly lower")
    return (
        f"ESMFold folds all three variants with comparable confidence "
        f"(mean pLDDT ~{plddt.get('wt', 0):.0f}); the {MUTATION} substitution "
        f"produces a measurable {warp:.2f} Å backbone deviation in the "
        f"DNA-binding core. With the {RESCUE_CANDIDATE} second-site candidate, "
        f"model confidence is {trend} (Δ pLDDT {delta:+.1f}). This is an in-silico "
        f"hypothesis about structure — not evidence of therapeutic rescue."
    )


def structural_rescue_analysis() -> Dict:
    """Assemble the honest analysis from the precomputed real structures.
    Never raises; returns ``available: False`` when structures are missing."""
    if not structures_available():
        return {"available": False,
                "reason": "ESMFold structures not precomputed (needs the GPU run).",
                "disclaimer": DISCLAIMER}
    wt_ca, _ = parse_ca(load_pdb("wt") or "")
    mut_ca, _ = parse_ca(load_pdb("r175h") or "")
    resc_ca, _ = parse_ca(load_pdb("r175h_n239y") or "")
    plddt = {v: mean_plddt(load_pdb(v) or "") for v in VARIANTS}
    warp = kabsch_rmsd(wt_ca, mut_ca)
    rescue_shift = kabsch_rmsd(mut_ca, resc_ca)
    return {
        "available": True,
        "device": "AMD Instinct MI300X (ROCm)",
        "model": "facebook/esmfold_v1",
        "mutation": MUTATION,
        "rescue_candidate": RESCUE_CANDIDATE,
        "mean_plddt": plddt,
        "warp_rmsd_wt_vs_r175h": warp,
        "rescue_rmsd_r175h_vs_candidate": rescue_shift,
        "plddt_delta_rescue": round(plddt["r175h_n239y"] - plddt["r175h"], 1),
        "verdict": _honest_verdict(warp, plddt),
        "disclaimer": DISCLAIMER,
    }


def hypothesis_prompt(mutation: str = MUTATION) -> Tuple[str, str]:
    """(system, user) prompt for Gemma to propose a suppressor — the architect step."""
    system = RESCUE_SYSTEM.format(mutation=mutation)
    user = (f"Oncogenic mutation: {mutation}. Propose one second-site suppressor "
            f"and its structural rationale.")
    return system, user


def gemma_interpret(analysis: Dict,
                    generate_fn: Callable[[str, str], str]) -> Optional[str]:
    """Let Gemma read the measured geometry and explain it. Injected generate_fn;
    returns None on any failure (graceful)."""
    if not analysis.get("available"):
        return None
    system = ("You are a structural biologist explaining an in-silico result to a "
              "clinician. Be precise, cautious, and note this is RUO, not therapeutic "
              "evidence. 2–3 sentences.")
    user = (f"ESMFold on AMD Instinct measured: {MUTATION} causes a "
            f"{analysis['warp_rmsd_wt_vs_r175h']} Å Cα warp vs wild-type; the "
            f"{RESCUE_CANDIDATE} candidate shifts the backbone by "
            f"{analysis['rescue_rmsd_r175h_vs_candidate']} Å with Δ pLDDT "
            f"{analysis['plddt_delta_rescue']:+.1f}. Interpret honestly.")
    try:
        out = generate_fn(system, user)
        return (out or "").strip() or None
    except Exception as e:  # graceful degradation
        log.warning(f"structure_rescue: gemma_interpret failed: {e}")
        return None


def rescue_overlay_html(wt_pdb: str, mutant_pdb: str,
                        mutation_resi: int = 175, height: int = 460) -> str:
    """Self-contained 3Dmol.js overlay: wild-type as a green ghost, mutant solid
    in amethyst, with the mutated residue highlighted. Returns an HTML string."""
    import json as _json
    wt_js = _json.dumps(wt_pdb or "")
    mut_js = _json.dumps(mutant_pdb or "")
    return f"""
<div id="rescue3d" style="width:100%;height:{height}px;position:relative;
     background:#0b0e1a;border-radius:10px;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function(){{
  var el = document.getElementById('rescue3d');
  var v = $3Dmol.createViewer(el, {{backgroundColor:'#0b0e1a'}});
  v.addModel({wt_js}, 'pdb');
  v.setStyle({{model:0}}, {{cartoon:{{color:'#4ade80', opacity:0.6}}}});
  v.addModel({mut_js}, 'pdb');
  v.setStyle({{model:1}}, {{cartoon:{{color:'#8b7cf6'}}}});
  v.addStyle({{model:1, resi:{mutation_resi}}},
             {{stick:{{color:'#f0a830', radius:0.4}}}});
  v.zoomTo();
  v.render();
  v.spin('y', 0.4);
}})();
</script>
"""
