"""
============================================================
TP53 RAG Platform - Export Disclaimer Stamping
============================================================
Stamps a Research-Use-Only (RUO) disclaimer onto every artifact
that leaves the application (Markdown reports, JSON payloads,
FHIR R4 resources).

Rationale: the on-screen banner only protects the live view. Once a
user downloads a report it can be emailed, printed, or filed with no
surrounding context. Embedding the disclaimer in the file itself keeps
the Research-Use-Only framing attached to the data wherever it travels.

Pure functions only — no Streamlit, no I/O, no global state. Every
function is defensive (never raises on odd input) and never returns an
empty payload.
============================================================
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

# Canonical RUO text — kept in lock-step with the on-screen banner in app.py.
RUO_DISCLAIMER = (
    "Research Use Only. Not a diagnostic device and not for clinical "
    "decisions. All outputs are informational — confirm with a CLIA-certified "
    "laboratory and a qualified clinician. Do not enter real, identifiable "
    "patient data."
)

# Short machine-readable tag used as the key/marker inside structured exports.
_DISCLAIMER_KEY = "_disclaimer"
_GENERATED_KEY = "_generated_utc"
_SOURCE_KEY = "_source"
_SOURCE_VALUE = "TP53 RAG Platform — Research Use Only"


def _utc_stamp() -> str:
    """ISO-8601 UTC timestamp (timezone-aware). Isolated for testability."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stamp_markdown(content: Any, title: str = "") -> str:
    """Return *content* with a RUO header and footer block.

    Always returns a non-empty string, even if *content* is None/empty —
    the disclaimer alone is still a valid (and safe) document.
    """
    body = "" if content is None else str(content)
    header_lines = ["> **⚠️ Research Use Only**", ">", f"> {RUO_DISCLAIMER}"]
    header = "\n".join(header_lines)
    footer = (
        f"---\n_{_SOURCE_VALUE}. Generated {_utc_stamp()}._\n\n"
        f"_{RUO_DISCLAIMER}_"
    )
    parts = [header]
    if title:
        parts.append(f"# {title}")
    if body.strip():
        parts.append(body)
    parts.append(footer)
    return "\n\n".join(parts) + "\n"


def stamp_dict(payload: Any) -> dict:
    """Return a deep copy of *payload* with RUO metadata keys injected.

    Never mutates the caller's object. If *payload* is not a dict (e.g. a
    list or scalar) it is wrapped under a ``data`` key so the disclaimer
    still has a home. Reserved keys already present are preserved under a
    ``_user_*`` alias rather than silently overwritten.
    """
    if isinstance(payload, dict):
        stamped = copy.deepcopy(payload)
    else:
        stamped = {"data": copy.deepcopy(payload)}

    for key, value in (
        (_DISCLAIMER_KEY, RUO_DISCLAIMER),
        (_SOURCE_KEY, _SOURCE_VALUE),
        (_GENERATED_KEY, _utc_stamp()),
    ):
        if key in stamped:
            stamped[f"_user{key}"] = stamped[key]
        stamped[key] = value
    return stamped


def stamp_json(payload: Any, indent: int = 2) -> str:
    """Serialise *payload* to JSON with RUO metadata embedded.

    Falls back to a string repr for non-serialisable values so a download
    never crashes the UI. Always returns valid, non-empty JSON text.
    """
    stamped = stamp_dict(payload)
    try:
        return json.dumps(stamped, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(stamped, indent=indent, ensure_ascii=False, default=str)


def stamp_fhir(resource: Any) -> dict:
    """Attach the RUO notice to a FHIR R4 resource without breaking schema.

    The disclaimer is added two ways, both schema-valid:
      * ``meta.security`` — a coded flag (HL7 ``ActReason`` style) marking
        the resource as research-use, machine-discoverable.
      * ``text.div`` is left untouched; instead a top-level ``note`` is added
        when the resource type supports it (best-effort, non-fatal).

    Returns a deep copy; never mutates input; never raises.
    """
    if not isinstance(resource, dict) or not resource:
        # Not a real resource — return a minimal Basic resource carrying the notice.
        return {
            "resourceType": "Basic",
            "meta": {"security": [_ruo_security_coding()]},
            _DISCLAIMER_KEY: RUO_DISCLAIMER,
        }

    stamped = copy.deepcopy(resource)
    meta = stamped.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        stamped["meta"] = meta
    security = meta.setdefault("security", [])
    if not isinstance(security, list):
        security = []
        meta["security"] = security
    if not any(
        isinstance(c, dict) and c.get("code") == "HRESCH" for c in security
    ):
        security.append(_ruo_security_coding())

    # Best-effort human-readable note (valid on many resource types).
    note = stamped.setdefault("note", [])
    if isinstance(note, list):
        note.append({"text": RUO_DISCLAIMER})

    return stamped


def _ruo_security_coding() -> dict:
    """HL7 v3 ActReason coding marking a resource as healthcare-research use."""
    return {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
        "code": "HRESCH",
        "display": "healthcare research",
    }
