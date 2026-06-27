"""
Build a print-ready PDF from WHITEPAPER.md.

Pure-Python (markdown + xhtml2pdf) so it runs offline with no system
dependencies. Regenerate any time the whitepaper changes:

    python tools/whitepaper_to_pdf.py

Output: WHITEPAPER.pdf (next to WHITEPAPER.md).

Notes
-----
- Mermaid diagrams render on GitHub but not in a static PDF, so the mermaid
  block is converted to a captioned monospace figure that still reads clearly.
- Emoji are stripped (PDF base fonts have no glyphs for them); Greek/maths such
  as ΔΔG render via the Windows Arial TTF registered below.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "WHITEPAPER.md"
PDF_PATH = ROOT / "WHITEPAPER.pdf"

# Windows Arial covers Latin + Greek (ΔΔG) + symbols; fall back gracefully.
_FONT_DIR = Path("C:/Windows/Fonts")
_ARIAL = _FONT_DIR / "arial.ttf"
_ARIAL_BD = _FONT_DIR / "arialbd.ttf"
_ARIAL_IT = _FONT_DIR / "ariali.ttf"

# Emoji / pictographs / variation selectors — no glyphs in PDF base fonts.
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoji
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002B00-\U00002BFF"   # arrows/symbols block (keep simple arrows below)
    "]",
    flags=re.UNICODE,
)


def _preprocess(md_text: str) -> str:
    """Make the Markdown PDF-friendly before conversion."""
    # Mermaid → captioned plain code figure (PDF can't render mermaid JS).
    def _mermaid(m: re.Match) -> str:
        body = m.group(1).rstrip()
        return (
            "**Figure — System Architecture** "
            "*(interactive diagram in the GitHub / Markdown version)*\n\n"
            "```\n" + body + "\n```\n"
        )

    md_text = re.sub(r"```mermaid\n(.*?)```", _mermaid, md_text, flags=re.DOTALL)
    # Strip emoji (keep the rest of the text intact).
    md_text = _EMOJI.sub("", md_text)
    # Map non-Latin-1 symbols to ASCII so the PDF base font (Helvetica) has a
    # glyph for everything — avoids missing-glyph boxes on ΔΔG, arrows, etc.
    replacements = {
        "ΔΔG": "ddG", "Δ": "delta",
        "→": "->", "←": "<-", "↓": "v", "↑": "^",
        "≥": ">=", "≤": "<=", "×": "x", "·": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }
    for src, dst in replacements.items():
        md_text = md_text.replace(src, dst)
    # Final safety net: drop any remaining characters Helvetica/WinAnsi can't show.
    md_text = md_text.encode("latin-1", "ignore").decode("latin-1")
    # Collapse any leftover double spaces from removed characters.
    md_text = re.sub(r"[ \t]{2,}", " ", md_text)
    return md_text


# Sentinel URIs resolved to real font files by _link_callback (avoids
# xhtml2pdf's broken URL→temp-file copy for Windows font paths).
_FONT_URIS = {
    "docsans-regular.ttf": _ARIAL,
    "docsans-bold.ttf": _ARIAL_BD,
    "docsans-italic.ttf": _ARIAL_IT,
}


def _link_callback(uri: str, rel: str) -> str:
    """Resolve our font sentinels to absolute paths; pass through everything else."""
    if uri in _FONT_URIS:
        return str(_FONT_URIS[uri])
    return uri


def _font_face_css() -> str:
    """No custom font: use the built-in Helvetica for identical, dependency-free
    output everywhere. Non-Latin symbols are ASCII-mapped in _preprocess(), so
    Helvetica has a glyph for everything and nothing renders as a box."""
    return ""


_BODY_FONT = "Helvetica"

CSS = f"""
{_font_face_css()}
@page {{
    size: a4 portrait;
    margin: 2cm 1.8cm 2.2cm 1.8cm;
    @frame footer {{
        -pdf-frame-content: footerContent;
        bottom: 1cm; left: 1.8cm; right: 1.8cm; height: 1cm;
    }}
}}
body {{ font-family: "{_BODY_FONT}"; font-size: 10.5pt; color: #1b1b1b; line-height: 1.42; }}
h1 {{ font-size: 21pt; color: #0b3d63; margin: 0 0 2px 0; }}
h2 {{ font-size: 14pt; color: #0b3d63; border-bottom: 1px solid #cfd8e0;
      padding-bottom: 3px; margin-top: 18px; margin-bottom: 6px; -pdf-keep-with-next: true; }}
h3 {{ font-size: 11.5pt; color: #145a8a; margin-top: 12px; margin-bottom: 4px;
      -pdf-keep-with-next: true; }}
p {{ margin: 4px 0 6px 0; }}
a {{ color: #145a8a; text-decoration: none; }}
strong {{ color: #11324a; }}
hr {{ border: none; border-top: 1px solid #dfe5ea; margin: 10px 0; }}
ul, ol {{ margin: 4px 0 8px 18px; }}
li {{ margin: 1px 0; }}
code {{ font-family: "Courier"; font-size: 9pt; background-color: #f3f5f7; }}
pre {{ font-family: "Courier"; font-size: 8.5pt; background-color: #f3f5f7;
       border: 0.5pt solid #d7dee4; padding: 6px; }}
table {{ -pdf-keep-in-frame-mode: shrink; margin: 6px 0 10px 0; width: 100%; }}
th {{ background-color: #0b3d63; color: #ffffff; border: 0.5pt solid #6b7c8a;
      padding: 4px 5px; font-size: 9pt; text-align: left; }}
td {{ border: 0.5pt solid #b8c2cb; padding: 4px 5px; font-size: 9pt; vertical-align: top; }}
blockquote {{ background-color: #fff6e0; border-left: 4px solid #e0a200;
              padding: 7px 10px; margin: 8px 0; color: #4a3b00; }}
#footerContent {{ color: #8a97a3; font-size: 8pt; text-align: center; }}
"""


def build() -> int:
    if not MD_PATH.exists():
        print(f"ERROR: {MD_PATH} not found", file=sys.stderr)
        return 1

    md_text = _preprocess(MD_PATH.read_text(encoding="utf-8"))
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>
{html_body}
<div id="footerContent">Precision Onco Africa - Technical Whitepaper - Research Use Only - page <pdf:pagenumber>/<pdf:pagecount></div>
</body></html>"""

    with open(PDF_PATH, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8",
                                link_callback=_link_callback)

    if result.err:
        print(f"ERROR: PDF generation reported {result.err} error(s)", file=sys.stderr)
        return 2
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"OK: wrote {PDF_PATH.name} ({size_kb:.0f} KB) | font={_BODY_FONT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
