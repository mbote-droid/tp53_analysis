"""
============================================================
Precision Onco Africa - Sanger .ab1 Chromatogram Reader + Variant Caller
agents/sanger_ab1.py
============================================================
Reads a real Sanger sequencing trace (.ab1 / ABIF format) with Biopython and
turns it into: QC metrics, the base-called sequence, heterozygous (double-peak)
site detection from the raw trace, and — if a reference segment is supplied —
variant calls against that reference with per-base quality gating.

Why this is honest:
  * The .ab1 parse is real (Biopython's ABIF reader), not a mock.
  * Heterozygous detection uses the actual secondary-peak intensity ratio at
    each called base — a genuine Sanger signal — not a guess.
  * Variant calling is a transparent pairwise alignment + mismatch report with
    the real PHRED quality attached; low-quality calls are flagged, not hidden.
  * There is NO claim of clinical-grade variant calling — this is a
    research/triage aid (RUO). Confirm every call on the raw trace.

`synthesize_ab1()` writes a minimal but valid ABIF file so a clinician without
a sequencer can still exercise the feature on a clearly-synthetic trace; real
uploads go through the same `parse_ab1()` path.

Everything degrades gracefully — missing Biopython or a corrupt file returns
{"success": False, ...}, never an unhandled exception.
"""
from __future__ import annotations

import io
import struct
from typing import Dict, List, Optional

from utils.logger import log

# ABIF base order for the four DATA channels we write (DATA9..DATA12).
_BASE_ORDER = "GATC"


def _require_biopython():
    from Bio import SeqIO  # noqa: F401
    return SeqIO


def parse_ab1(data: bytes) -> Dict:
    """Parse ABIF bytes into sequence, quality, trace channels and peaks.
    Never raises — returns {"success": False, "error": ...} on failure."""
    try:
        SeqIO = _require_biopython()
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Biopython not installed: {e}"}
    try:
        rec = SeqIO.read(io.BytesIO(data), "abi")
    except Exception as e:
        return {"success": False, "error": f"Not a readable .ab1 file: {e}"}

    seq = str(rec.seq)
    quality = list(rec.letter_annotations.get("phred_quality", []))
    raw = rec.annotations.get("abif_raw", {})

    def _as_list(v):
        if v is None:
            return []
        return list(v) if isinstance(v, (tuple, list)) else [v]

    base_order = raw.get("FWO_1")
    if isinstance(base_order, bytes):
        base_order = base_order.decode(errors="replace")
    base_order = base_order or _BASE_ORDER

    trace = {b: _as_list(raw.get(f"DATA{9 + i}"))
             for i, b in enumerate(base_order[:4])}
    peaks = _as_list(raw.get("PLOC2") or raw.get("PLOC1"))

    return {"success": True, "sequence": seq, "quality": quality,
            "trace": trace, "peaks": peaks, "base_order": base_order,
            "length": len(seq)}


def qc_metrics(quality: List[int]) -> Dict:
    """Read-quality summary. Pure; never raises."""
    q = [int(x) for x in (quality or [])]
    n = len(q)
    if n == 0:
        return {"length": 0, "mean_quality": 0.0, "q20_fraction": 0.0,
                "usable": False, "note": "No quality values in trace."}
    mean_q = sum(q) / n
    q20 = sum(1 for x in q if x >= 20) / n
    usable = mean_q >= 20 and q20 >= 0.6
    return {"length": n, "mean_quality": round(mean_q, 1),
            "q20_fraction": round(q20, 3), "usable": bool(usable),
            "note": ("Good-quality read." if usable
                     else "Low-quality read — interpret variants with caution.")}


def detect_heterozygous_sites(sequence: str, trace: Dict, peaks: List[int],
                              base_order: str = _BASE_ORDER,
                              min_ratio: float = 0.35) -> List[Dict]:
    """Flag positions where a secondary trace peak rivals the called base — the
    classic Sanger signature of a heterozygous variant. Returns a list of
    {position (1-based), called_base, secondary_base, ratio}. Pure; never
    raises. Requires per-base peak locations aligned to the trace."""
    out: List[Dict] = []
    if not sequence or not peaks or not trace:
        return out
    channels = base_order[:4]
    trace_len = max((len(trace.get(b, [])) for b in channels), default=0)
    if trace_len == 0:
        return out
    for i, called in enumerate(sequence):
        if i >= len(peaks):
            break
        loc = peaks[i]
        if loc < 0 or loc >= trace_len:
            continue
        intensities = {}
        for b in channels:
            ch = trace.get(b, [])
            intensities[b] = ch[loc] if loc < len(ch) else 0
        primary = intensities.get(called, max(intensities.values(), default=0))
        if primary <= 0:
            continue
        others = {b: v for b, v in intensities.items() if b != called}
        if not others:
            continue
        sec_base = max(others, key=others.get)
        ratio = others[sec_base] / primary
        if ratio >= min_ratio:
            out.append({"position": i + 1, "called_base": called,
                        "secondary_base": sec_base, "ratio": round(ratio, 2)})
    return out


def call_variants(read_seq: str, read_quality: List[int], reference: str,
                  min_quality: int = 20) -> List[Dict]:
    """Call variants of the read against a reference segment via global
    pairwise alignment. Reports substitutions, insertions and deletions with
    the read PHRED quality; sub-threshold calls are flagged low_confidence.
    Never raises — alignment failure returns []."""
    if not read_seq or not reference:
        return []
    try:
        from Bio.Align import PairwiseAligner
    except Exception as e:  # pragma: no cover
        log.warning(f"PairwiseAligner unavailable: {e}")
        return []
    try:
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -2.5
        aligner.extend_gap_score = -0.5
        aln = aligner.align(reference, read_seq)[0]
        idx = aln.indices  # shape (2, L): row0=reference, row1=read; -1 = gap
    except Exception as e:
        log.warning(f"Sanger alignment failed: {e}")
        return []

    variants: List[Dict] = []
    ncols = idx.shape[1]
    for c in range(ncols):
        i_ref = int(idx[0, c])
        i_read = int(idx[1, c])
        if i_ref >= 0 and i_read >= 0:
            r, a = reference[i_ref], read_seq[i_read]
            if r != a:
                q = read_quality[i_read] if i_read < len(read_quality) else None
                variants.append(_variant("substitution", i_ref + 1, r, a, q,
                                         min_quality))
        elif i_ref >= 0 and i_read < 0:
            variants.append(_variant("deletion", i_ref + 1, reference[i_ref],
                                     "-", None, min_quality))
        elif i_ref < 0 and i_read >= 0:
            q = read_quality[i_read] if i_read < len(read_quality) else None
            variants.append(_variant("insertion", i_ref + 1, "-",
                                     read_seq[i_read], q, min_quality))
    return variants


def _variant(kind: str, pos: int, ref: str, alt: str,
             quality: Optional[int], min_quality: int) -> Dict:
    low = quality is not None and quality < min_quality
    return {"type": kind, "ref_position": pos, "ref": ref, "alt": alt,
            "quality": quality,
            "confidence": "low" if (low or quality is None) else "high",
            "low_confidence": bool(low or quality is None)}


def analyze_ab1(data: bytes, reference: Optional[str] = None) -> Dict:
    """Full pipeline: parse → QC → heterozygous sites → (optional) variant
    calls vs a reference. Never raises."""
    parsed = parse_ab1(data)
    if not parsed.get("success"):
        return parsed
    qc = qc_metrics(parsed["quality"])
    het = detect_heterozygous_sites(
        parsed["sequence"], parsed["trace"], parsed["peaks"],
        parsed["base_order"])
    result = {"success": True, "sequence": parsed["sequence"],
              "length": parsed["length"], "qc": qc,
              "heterozygous_sites": het,
              "disclaimer": ("Research use only — Sanger basecalls and variant "
                             "calls must be confirmed on the raw trace by a "
                             "qualified analyst.")}
    if reference:
        result["variants"] = call_variants(
            parsed["sequence"], parsed["quality"], reference)
    return result


# ─────────────────────────────────────────────────────────────────────
# Minimal ABIF writer — produces a valid, Biopython-readable .ab1 so the
# feature can be demoed without a physical sequencer. Clearly synthetic.
# ─────────────────────────────────────────────────────────────────────

_PEAK_SPACING = 12
_PEAK_START = 12


def synthesize_ab1(sequence: str, quality: Optional[List[int]] = None,
                   het_sites: Optional[Dict[int, str]] = None) -> bytes:
    """Build a minimal but valid ABIF (.ab1) byte string for `sequence`.

    quality: optional per-base PHRED (default 40, tapering at the ends).
    het_sites: optional {1-based position: alt_base} to inject a secondary
    trace peak, so heterozygous detection has something to find.

    This is a SYNTHETIC trace generator for demos/tests — the peaks are
    idealised Gaussians, not real electrophoresis data.
    """
    seq = (sequence or "").upper()
    n = len(seq)
    if quality is None:
        quality = [40 if 3 <= i < n - 3 else 25 for i in range(n)]
    quality = [max(0, min(62, int(q))) for q in quality]
    het_sites = het_sites or {}

    # Peak locations, evenly spaced.
    peaks = [_PEAK_START + i * _PEAK_SPACING for i in range(n)]
    trace_len = (peaks[-1] + _PEAK_START) if peaks else _PEAK_START

    # Four channels in _BASE_ORDER ("GATC"); Gaussian bump per called base.
    channels = {b: [0] * trace_len for b in _BASE_ORDER}

    def _bump(chan: List[int], center: int, height: int, width: float = 2.2):
        for x in range(max(0, center - 6), min(trace_len, center + 7)):
            chan[x] += int(height * pow(2.718281828, -((x - center) ** 2) /
                                        (2 * width * width)))

    for i, base in enumerate(seq):
        center = peaks[i]
        prim = base if base in channels else "G"
        _bump(channels[prim], center, 1000)
        alt = het_sites.get(i + 1)
        if alt and alt in channels:
            _bump(channels[alt], center, 600)  # secondary peak ~0.6 ratio

    def _clip16(v):
        return max(-32768, min(32767, int(v)))

    # Assemble the tag table.
    tags = []

    def add(name: str, number: int, code: int, size: int, num: int, raw: bytes):
        tags.append({"name": name.encode(), "number": number, "code": code,
                     "size": size, "num": num, "raw": raw})

    seq_bytes = seq.encode()
    qual_bytes = bytes(quality)
    add("PBAS", 1, 2, 1, n, seq_bytes)
    add("PBAS", 2, 2, 1, n, seq_bytes)
    add("PCON", 1, 2, 1, n, qual_bytes)
    add("PCON", 2, 2, 1, n, qual_bytes)
    add("FWO_", 1, 2, 1, 4, _BASE_ORDER.encode())
    ploc_raw = struct.pack(">" + "h" * n, *[_clip16(p) for p in peaks]) if n else b""
    add("PLOC", 1, 4, 2, n, ploc_raw)
    add("PLOC", 2, 4, 2, n, ploc_raw)
    for i, b in enumerate(_BASE_ORDER):
        ch = channels[b]
        raw = struct.pack(">" + "h" * trace_len, *[_clip16(v) for v in ch])
        add("DATA", 9 + i, 4, 2, trace_len, raw)

    # Layout: "ABIF"(4) + header(26) + directory(N*28) + data blocks.
    head_offset = 4 + 26
    n_tags = len(tags)
    data_start = head_offset + n_tags * 28

    dir_bytes = bytearray()
    data_bytes = bytearray()
    cursor = data_start
    for t in tags:
        raw = t["raw"]
        datasize = len(raw)
        if datasize <= 4:
            # Inline: store raw (left-justified, zero-padded) in the offset field.
            offset_field = int.from_bytes(raw.ljust(4, b"\x00")[:4], "big")
        else:
            offset_field = cursor
            data_bytes += raw
            cursor += datasize
        dir_bytes += struct.pack(">4sI2H4I", t["name"], t["number"], t["code"],
                                 t["size"], t["num"], datasize, offset_field, 0)

    header = struct.pack(">H4sI2H3I", 101, b"tdir", 1, 1023, 28,
                         n_tags, n_tags * 28, head_offset)
    return b"ABIF" + header + bytes(dir_bytes) + bytes(data_bytes)
