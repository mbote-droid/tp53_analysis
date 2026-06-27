"""
============================================================
TP53 RAG Platform - Codebase Knowledge Graph (DNA helix)
utils/codegraph.py
============================================================
Builds a real dependency graph of the platform's own source — nodes are
modules, edges are internal imports — and lays the nodes out along a DNA
double helix. The result is both a genuine architecture map and the project's
signature visual: the dots, if joined, trace a double helix (fitting, for a
genomics platform whose code literally forms DNA).

Everything here is pure and deterministic: imports are parsed with `ast`, so
no module is executed, and the layout coordinates are computed analytically.
Offline, never-empty, safe to run anywhere.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Dict, List, Optional, Set

# Internal top-level packages whose imports count as code-graph edges.
_INTERNAL_ROOTS = {
    "agents", "utils", "config", "knowledge_base", "api", "benchmarks", "tools",
}
_SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules",
              "tp53_rag", ".pytest_cache"}

# Group colours by package (year-2100 neon palette).
_GROUP_COLOR = {
    "agents": "#00d4ff", "utils": "#2ecc71", "config": "#f1c40f",
    "knowledge_base": "#ff6b9d", "api": "#a29bfe", "benchmarks": "#f39c12",
    "tools": "#ff8c6b", "root": "#9aa7b4",
}


def _module_id(path: Path, root: Path) -> str:
    """Dotted module id for a .py file relative to root (e.g. agents.rag_chain)."""
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _iter_py_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        files.append(p)
    return files


def _imports(path: Path) -> Set[str]:
    """Top-level imported module names parsed from a file (ast — no execution)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.add(node.module)
    return out


def build_codegraph(root: Optional[Path] = None) -> Dict:
    """Return {nodes, links} for the internal import graph. Never empty."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    files = _iter_py_files(root)
    id_by_module: Dict[str, Path] = {}
    for f in files:
        mid = _module_id(f, root)
        if mid.split(".")[0] in _INTERNAL_ROOTS or "." not in mid:
            id_by_module[mid] = f

    internal = set(id_by_module)
    nodes: List[Dict] = []
    links: List[Dict] = []
    for mid, f in sorted(id_by_module.items()):
        group = mid.split(".")[0] if mid.split(".")[0] in _GROUP_COLOR else "root"
        try:
            loc = len(f.read_text(encoding="utf-8").splitlines())
        except Exception:
            loc = 0
        nodes.append({"id": mid, "label": mid.split(".")[-1], "group": group,
                      "color": _GROUP_COLOR.get(group, "#9aa7b4"), "loc": loc})
        for imp in _imports(f):
            # Match the longest internal module prefix of the import.
            target = None
            if imp in internal:
                target = imp
            else:
                parts = imp.split(".")
                for k in range(len(parts), 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in internal:
                        target = cand
                        break
            if target and target != mid:
                links.append({"source": mid, "target": target})

    if not nodes:  # never empty
        nodes = [{"id": "tp53_rag", "label": "tp53_rag", "group": "root",
                  "color": "#9aa7b4", "loc": 0}]
    return {"nodes": nodes, "links": links,
            "module_count": len(nodes), "edge_count": len(links)}


def helix_layout(graph: Dict, radius: float = 28.0, rise: float = 7.0,
                 turn: float = 0.55) -> Dict:
    """Attach (x, y, z) coordinates placing nodes on a DNA double helix.

    Even-indexed nodes go on strand A, odd on strand B (offset by π), so the
    two strands spiral around a common axis — joining the dots traces a double
    helix. Returns the same graph with coordinates + base-pair rungs added.
    """
    nodes = graph.get("nodes", [])
    n = len(nodes)
    for i, node in enumerate(nodes):
        strand = i % 2
        t = (i // 2) * turn
        phase = t + (math.pi if strand else 0.0)
        node["x"] = round(radius * math.cos(phase), 3)
        node["z"] = round(radius * math.sin(phase), 3)
        node["y"] = round((i // 2) * rise - (n // 4) * rise, 3)
        node["strand"] = strand
    # Base-pair rungs connect the two strands at the same vertical level.
    rungs = [{"source": nodes[i]["id"], "target": nodes[i + 1]["id"]}
             for i in range(0, n - 1, 2)]
    graph["rungs"] = rungs
    graph["layout"] = "double_helix"
    return graph


def build_helix_codegraph(root: Optional[Path] = None) -> Dict:
    """Convenience: build the import graph and apply the helix layout."""
    return helix_layout(build_codegraph(root))
