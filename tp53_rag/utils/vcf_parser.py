"""
============================================================
TP53 RAG Platform - VCF Parser
utils/vcf_parser.py
============================================================
Accept real patient VCF files (v4.1 / v4.2) as input instead of manual
mutation entry. Filters to the TP53 locus (chr17p13.1) and returns a
structured mutation list compatible with the pipeline_data format.

Honest annotation policy:
  * The protein change (e.g. R175H) is taken from the VCF's OWN annotation
    (HGVS p./c. in the INFO field — what VEP/SnpEff/ClinVar write), using a
    standard 3-letter -> 1-letter amino-acid table.
  * Unannotated variants are reported with their genomic coordinates only
    (REF>ALT at position); no amino-acid change is invented.
  * Known TP53 hotspot codons are flagged once an AA change is available.

Pure functions throughout — unit-testable without any file I/O or network.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from utils.logger import log

# Standard 3-letter -> 1-letter amino-acid table (+ stop / synonymous).
AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U",
}

# TP53 genomic spans (generous windows) for both common reference builds.
# GRCh38 ~ chr17:7,668,421-7,687,550 ; GRCh37/hg19 ~ chr17:7,565,097-7,590,856.
TP53_REGIONS = [
    (7_660_000, 7_690_000),   # GRCh38
    (7_560_000, 7_596_000),   # GRCh37 / hg19
]

# Well-established TP53 hotspot codons (AA-level flagging).
HOTSPOT_CODONS = {175, 176, 179, 220, 238, 242, 245, 248, 249, 273, 282}

_HGVS_P3 = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|\*|=)")
_HGVS_P1 = re.compile(r"p\.([A-Z])(\d+)([A-Z*=])")
_HGVS_C = re.compile(r"c\.([\d_]+[ACGTacgt]+>[ACGTacgt]+|[\d_]+[a-zA-Z>+\-]*)")


def _norm_chrom(chrom: str) -> str:
    c = str(chrom or "").strip().lower().replace("chr", "")
    if c.startswith("nc_000017"):
        return "17"
    return c


def is_tp53_locus(chrom: str, pos: int) -> bool:
    """True if a genomic coordinate falls within the TP53 locus (either build)."""
    if _norm_chrom(chrom) != "17":
        return False
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return False
    return any(lo <= p <= hi for lo, hi in TP53_REGIONS)


def extract_protein_change(info: str) -> Optional[str]:
    """Pull a 1-letter protein change (e.g. R175H) from an INFO annotation."""
    text = str(info or "")
    m = _HGVS_P3.search(text)
    if m:
        ref = AA_3TO1.get(m.group(1))
        alt = AA_3TO1.get(m.group(3), m.group(3) if m.group(3) in ("*", "=") else None)
        if ref and alt is not None:
            return f"{ref}{m.group(2)}{alt}"
    m = _HGVS_P1.search(text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return None


def extract_cdna_change(info: str) -> Optional[str]:
    m = _HGVS_C.search(str(info or ""))
    return f"c.{m.group(1)}" if m else None


def _codon_of(aa_change: Optional[str]) -> Optional[int]:
    if not aa_change:
        return None
    digits = "".join(c for c in aa_change if c.isdigit())
    return int(digits) if digits else None


def parse_vcf_text(text: str, tp53_only: bool = True) -> Dict:
    """Parse VCF text into structured records. Never raises.

    Returns {variants, total_lines, tp53_count, skipped, columns_ok}.
    Each variant is pipeline_data-compatible:
      {gene, chrom, pos, ref, alt, qual, filter, amino_acid_change,
       hgvs_c, is_hotspot, annotated, source}
    """
    variants: List[Dict] = []
    total = skipped = 0
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        total += 1
        cols = line.split("\t")
        if len(cols) < 5:
            cols = re.split(r"\s+", line)  # tolerate space-delimited
        if len(cols) < 5:
            skipped += 1
            continue
        chrom, pos, _id, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
        qual = cols[5] if len(cols) > 5 else "."
        filt = cols[6] if len(cols) > 6 else "."
        info = cols[7] if len(cols) > 7 else ""
        try:
            ipos = int(pos)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if tp53_only and not is_tp53_locus(chrom, ipos):
            continue
        aa = extract_protein_change(info) or extract_protein_change(_id)
        codon = _codon_of(aa)
        variants.append({
            "gene": "TP53",
            "chrom": _norm_chrom(chrom),
            "pos": ipos,
            "ref": ref,
            "alt": alt,
            "qual": qual,
            "filter": filt,
            "amino_acid_change": aa,
            "hgvs_c": extract_cdna_change(info),
            "is_hotspot": bool(codon and codon in HOTSPOT_CODONS),
            "annotated": aa is not None,
            "source": "vcf",
        })
    return {
        "variants": variants,
        "total_lines": total,
        "tp53_count": len(variants),
        "skipped": skipped,
        "columns_ok": True,
    }


def parse_vcf_bytes(data: bytes, tp53_only: bool = True) -> Dict:
    """Parse VCF from raw bytes (e.g. an uploaded file). Decodes UTF-8 leniently."""
    try:
        text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
    except Exception as e:  # pragma: no cover
        log.warning(f"VCF decode failed: {e}")
        return {"variants": [], "total_lines": 0, "tp53_count": 0,
                "skipped": 0, "columns_ok": False}
    return parse_vcf_text(text, tp53_only=tp53_only)


SAMPLE_VCF = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
17\t7675088\t.\tC\tT\t250\tPASS\tANN=T|missense_variant|HIGH|TP53|HGVS.c=c.524G>A|HGVS.p=p.Arg175His
17\t7674220\t.\tC\tT\t180\tPASS\tHGVSp=p.Arg248Trp
17\t7670700\t.\tG\tA\t90\tLowQual\tTP53 intronic, unannotated
1\t12345\t.\tA\tG\t99\tPASS\tnot TP53
"""


def sample_vcf() -> str:
    """A small demo VCF (GRCh38) for the UI — one hotspot, one annotated, one
    unannotated TP53 variant, plus a non-TP53 line that must be filtered out."""
    return SAMPLE_VCF
