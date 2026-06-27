"""
============================================================
Precision Onco Africa - Molecular Docking Agent (AutoDock Vina)
agents/molecular_docking.py
============================================================
Estimates drug–protein binding for a TP53 mutation + candidate drug.

Two honest modes (the active mode is ALWAYS reported in `method`):
  * "autodock_vina"      — a real AutoDock Vina run (subprocess) when Vina is
                           installed AND prepared PDBQT receptor/ligand are
                           supplied. The best binding affinity is parsed from
                           Vina's output.
  * "heuristic_estimate" — fallback when Vina (or its inputs) is unavailable,
                           e.g. on Streamlit Cloud. A deterministic, clearly
                           labelled ESTIMATE — NOT a real docking result.

Offline-first, graceful (never crashes if Vina is missing), never empty.
Vina-output parsing is a pure function (unit-testable without Vina).

DISCLAIMER: binding values are research/educational estimates unless
`method == "autodock_vina"`; verify experimentally before any conclusion.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import log

AGENT_ID = "molecular_docking"
DISCLAIMER = ("Binding values are estimates unless method == 'autodock_vina'; "
              "research/educational use only — verify experimentally.")

# Curated interaction context per binding strategy (illustrative).
_INTERACTIONS = {
    "stabiliser": ["H-bond to the mutation-induced surface pocket",
                   "hydrophobic packing in the destabilised cleft"],
    "reactivator": ["covalent/thiol engagement of a cysteine",
                    "stabilisation of the DNA-binding domain fold"],
    "mdm2_inhibitor": ["occupies the MDM2 p53-binding cleft (Phe19/Trp23/Leu26 mimic)"],
    "chemotherapy": ["DNA adduct formation (p53-independent)"],
    "generic": ["van der Waals contacts in the p53 DNA-binding domain"],
}


def _strategy(mechanism: str) -> str:
    m = str(mechanism or "").lower()
    if "mdm2" in m or "mdmx" in m or "mdm4" in m:
        return "mdm2_inhibitor"
    if "stabilis" in m or "stabiliz" in m or "pocket" in m or "y220c" in m:
        return "stabiliser"
    if "reactivat" in m or "refold" in m or "metallochaperone" in m or "thiol" in m:
        return "reactivator"
    if "dna" in m or "platin" in m or "cross" in m:
        return "chemotherapy"
    return "generic"


def vina_available() -> bool:
    """True if an AutoDock Vina executable is on PATH."""
    return shutil.which("vina") is not None or shutil.which("vina.exe") is not None


def parse_vina_output(text: str) -> Optional[float]:
    """Parse the best (mode 1) binding affinity (kcal/mol) from Vina stdout.

    Pure & defensive — returns None if no affinity table is found.
    Vina prints rows like:  '   1       -7.5      0.000      0.000'
    """
    best: Optional[float] = None
    for line in str(text or "").splitlines():
        m = re.match(r"\s*(\d+)\s+(-?\d+\.\d+)\s", line)
        if m:
            try:
                aff = float(m.group(2))
            except ValueError:
                continue
            if best is None or aff < best:  # most negative = strongest
                best = aff
    return best


@dataclass
class DockingResult:
    mutation: str
    drug: str
    pdb_id: str
    binding_affinity: Optional[float]   # kcal/mol
    method: str                          # autodock_vina | heuristic_estimate
    vina_available: bool
    pocket_residues: List[int]
    interactions: List[str]
    disclaimer: str = DISCLAIMER


class MolecularDockingAgent:
    """Dock a candidate drug against TP53 — real Vina or honest estimate."""

    def __init__(self, timeout: float = 120.0) -> None:
        self._timeout = timeout
        self._audit_log = Path("logs/molecular_docking.log")
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning(f"Docking audit dir unavailable: {e}")

    # ── pocket + estimate helpers (lazy viz import; never crash) ──
    @staticmethod
    def _pocket(mutation: str) -> List[int]:
        try:
            from utils.viz import parse_residues
            return parse_residues(mutation)
        except Exception:
            return [175, 248, 273]

    @staticmethod
    def _estimate_affinity(mutation: str, drug: str) -> float:
        try:
            from utils.viz import dock_candidates
            for c in dock_candidates(mutation):
                if str(drug).lower() in c["name"].lower() or c["name"].lower() in str(drug).lower():
                    return float(c["affinity"])
            return float(dock_candidates(mutation)[0]["affinity"])  # best generic
        except Exception:
            return -6.0  # neutral default

    def _run_vina(self, receptor_pdbqt: str, ligand_pdbqt: str,
                  config: Optional[str]) -> Optional[float]:
        cmd = ["vina", "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt]
        if config:
            cmd += ["--config", config]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self._timeout)
            return parse_vina_output(proc.stdout)
        except Exception as e:
            log.warning(f"Vina run failed: {e}")
            return None

    def dock(self, mutation: str, drug: str, pdb_id: str = "2OCJ",
             mechanism: str = "", receptor_pdbqt: Optional[str] = None,
             ligand_pdbqt: Optional[str] = None,
             config: Optional[str] = None) -> Dict:
        """Dock `drug` against TP53 `mutation`. Never crashes; never empty."""
        mut = str(mutation or "TP53").strip()
        drg = str(drug or "candidate").strip()
        pdb = "".join(c for c in str(pdb_id) if c.isalnum())[:8] or "2OCJ"
        have_vina = vina_available()

        affinity: Optional[float] = None
        method = "heuristic_estimate"
        if have_vina and receptor_pdbqt and ligand_pdbqt:
            affinity = self._run_vina(receptor_pdbqt, ligand_pdbqt, config)
            if affinity is not None:
                method = "autodock_vina"
        if affinity is None:  # fallback — honest estimate
            affinity = self._estimate_affinity(mut, drg)
            method = "heuristic_estimate"

        result = DockingResult(
            mutation=mut, drug=drg, pdb_id=pdb,
            binding_affinity=round(affinity, 2) if affinity is not None else None,
            method=method, vina_available=have_vina,
            pocket_residues=self._pocket(mut),
            interactions=_INTERACTIONS.get(_strategy(mechanism), _INTERACTIONS["generic"]),
        )
        self._audit(f"dock:{mut}/{drg} -> {result.binding_affinity} ({method})")
        return {
            **asdict(result),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": (f"{drg} vs TP53 {mut}: {result.binding_affinity} kcal/mol "
                        f"({'AutoDock Vina' if method == 'autodock_vina' else 'estimate'})"),
        }

    def _audit(self, msg: str) -> None:
        try:
            entry = json.dumps({"ts": datetime.now().isoformat(), "event": msg}) + "\n"
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:  # pragma: no cover
            log.warning(f"Docking audit failed: {e}")


_agent = MolecularDockingAgent()


def dock_drug(mutation: str, drug: str, mechanism: str = "") -> Dict:
    return _agent.dock(mutation, drug, mechanism=mechanism)
