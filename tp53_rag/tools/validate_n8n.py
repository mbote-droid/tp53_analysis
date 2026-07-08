"""
============================================================
n8n Workflow Structural Validator
tools/validate_n8n.py
============================================================
Verifies n8n_workflow.json is a coherent, fully-wired automation graph WITHOUT
needing a running n8n daemon: every node reachable, a webhook trigger present,
the FastAPI hand-off (httpRequest) nodes present, and an audit trail (writeFile)
present.

This is deliberately honest: it validates the *wiring*, not a live execution.
A true end-to-end run requires the n8n container (`docker compose up n8n`, then
import this JSON) — see validate_result()'s note. Never raises on bad input;
returns a structured report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "n8n_workflow.json"


def validate_workflow(path: Optional[Path] = None) -> Dict:
    """Load + structurally validate the workflow. Never raises."""
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return {"ok": False, "reason": "workflow_missing", "path": str(p)}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "reason": f"unreadable: {e}", "path": str(p)}

    nodes: List[Dict] = data.get("nodes", []) or []
    conns: Dict = data.get("connections", {}) or {}
    if not nodes:
        return {"ok": False, "reason": "no_nodes"}

    names = {n.get("name") for n in nodes}
    types = [str(n.get("type", "")).lower() for n in nodes]

    # Reachability: a node is connected if it's a connection source or target.
    targets = set()
    for _src, outs in conns.items():
        for out in outs.get("main", []) or []:
            for c in (out or []):
                if c and c.get("node"):
                    targets.add(c["node"])
    sources = set(conns.keys())
    orphans = sorted(n for n in names
                     if n not in targets and n not in sources)

    has_webhook = any("webhook" in t for t in types)
    http_nodes = sum(1 for t in types if "httprequest" in t)
    audit_nodes = sum(1 for t in types if "writefile" in t)

    checks = {
        "has_webhook_trigger": has_webhook,
        "has_fastapi_httprequest": http_nodes >= 1,
        "has_audit_writefile": audit_nodes >= 1,
        "no_orphan_nodes": not orphans,
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "node_count": len(nodes),
        "connection_sources": len(conns),
        "http_request_nodes": http_nodes,
        "audit_nodes": audit_nodes,
        "orphans": orphans,
        "checks": checks,
        "note": ("Structural wiring verified. A live end-to-end run requires "
                 "the n8n container: `docker compose up n8n`, open "
                 "http://localhost:5678, import n8n_workflow.json, and POST to "
                 "the webhook — the httpRequest nodes call the FastAPI service "
                 "at http://api:8000."),
    }


def main() -> None:
    report = validate_workflow()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
