"""
============================================================
Precision Onco Africa - Visual Protein Snapshots (Gemma multimodal)
agents/structure_snapshot.py
============================================================
Renders a 2D snapshot of the p53 alpha-carbon backbone — with the patient's
mutated residue highlighted — entirely SERVER-SIDE (matplotlib/Agg, no WebGL,
no GPU), then hands that PNG to Gemma 4 vision. Instead of describing the
structure in text and asking the model to imagine it, we let Gemma *see* it.
That is "encoder-free multimodal RAG" using the model's strongest native
capability, honestly.

Coarse by design: this is a Cα trace, not an all-atom model, and every
downstream narration says so (research use only).

Pure/deterministic render (never raises → returns None on failure); the Gemma
call is injected/optional so this is unit-testable offline.
"""
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

from utils.logger import log

_MUT_CODON = re.compile(r"[A-Za-z](\d{1,4})")


def mutation_codon(mutation: str) -> Optional[int]:
    """Extract the residue number from a mutation like 'R175H'. None if absent."""
    m = _MUT_CODON.search(mutation or "")
    return int(m.group(1)) if m else None


def parse_ca_coords(pdb_text: str) -> Dict[int, Tuple[float, float, float]]:
    """Parse alpha-carbon (CA) coordinates keyed by residue number from PDB
    text. Fixed-column parse; tolerant; never raises."""
    coords: Dict[int, Tuple[float, float, float]] = {}
    for line in (pdb_text or "").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resi = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        coords.setdefault(resi, (x, y, z))
    return coords


def render_snapshot(pdb_text: str, highlight_resi: Optional[int] = None,
                    mutation: str = "", title: str = "") -> Optional[bytes]:
    """Render the Cα backbone as a PNG (bytes), highlighting `highlight_resi`.
    Returns None if there is nothing to draw. Never raises."""
    coords = parse_ca_coords(pdb_text)
    if len(coords) < 3:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:  # pragma: no cover
        log.warning(f"matplotlib unavailable for snapshot: {e}")
        return None
    try:
        order = sorted(coords)
        xs = [coords[r][0] for r in order]
        ys = [coords[r][1] for r in order]
        zs = [coords[r][2] for r in order]

        fig = plt.figure(figsize=(6, 6), facecolor="#0b0e1a")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0b0e1a")
        # backbone trace, colour-graded N→C so the fold reads
        n = len(order)
        for i in range(n - 1):
            frac = i / max(n - 1, 1)
            ax.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2],
                    color=(0.35 + 0.5 * frac, 0.45, 0.95 - 0.4 * frac),
                    linewidth=1.6)
        ax.scatter(xs, ys, zs, s=6, c="#8b7cf6", alpha=0.5)

        if highlight_resi in coords:
            hx, hy, hz = coords[highlight_resi]
            ax.scatter([hx], [hy], [hz], s=260, c="#f0a830",
                       edgecolors="#ff5d8f", linewidths=2, depthshade=False)
            ax.text(hx, hy, hz, f"  {mutation or highlight_resi}",
                    color="#f0a830", fontsize=12, weight="bold")

        ax.set_axis_off()
        ttl = title or (f"p53 Cα backbone — residue {highlight_resi} highlighted"
                        if highlight_resi else "p53 Cα backbone")
        ax.set_title(ttl, color="#e8eaf2", fontsize=11)
        ax.view_init(elev=18, azim=35)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="#0b0e1a",
                    bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        log.error(f"Structure snapshot render failed: {e}")
        return None


def analyze_structure(mutation: str, pdb_text: Optional[str] = None,
                      gemma_agent: object = None,
                      use_live: bool = True) -> Dict:
    """Full pipeline: fetch/accept a p53 structure → render the snapshot with
    the mutated residue highlighted → let Gemma vision comment on it.
    Returns the PNG (for display) + the narration. Never raises."""
    if pdb_text is None:
        try:
            from utils.alphafold_client import get_tp53_structure
            struct = get_tp53_structure(use_live=use_live)
            pdb_text = getattr(struct, "pdb_text", None)
        except Exception as e:
            return {"success": False, "error": f"Structure fetch failed: {e}"}
    if not pdb_text:
        return {"success": False,
                "error": "No AlphaFold structure available (offline?)."}

    resi = mutation_codon(mutation)
    png = render_snapshot(pdb_text, highlight_resi=resi, mutation=mutation)
    if not png:
        return {"success": False, "error": "Could not render the structure."}

    result = {"success": True, "image_png": png, "residue": resi,
              "mutation": mutation, "narration": None}

    if gemma_agent is None:
        try:
            from agents.gemma_vision import GemmaVisionAgent
            gemma_agent = GemmaVisionAgent()
        except Exception:
            gemma_agent = None
    if gemma_agent is not None and getattr(gemma_agent, "health", lambda: False)():
        vis = gemma_agent.read_structure_snapshot(png, "image/png", mutation)
        if vis.get("success"):
            result["narration"] = vis["narration"]
            result["caution"] = vis.get("caution")
        else:
            result["narration_error"] = vis.get("error")
    else:
        result["narration_error"] = ("Gemma vision unavailable (no "
                                      "GOOGLE_API_KEY) — snapshot rendered only.")
    return result
