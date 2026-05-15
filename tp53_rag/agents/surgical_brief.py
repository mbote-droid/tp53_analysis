"""
============================================================
TP53 RAG Platform - Agent #10: Pre-Incision Surgical Brief
============================================================
Generates a strict, high-contrast surgical briefing designed
for display on an iPad or overhead screen inside an
operating room. 90 seconds to read. No fluff.

Answers exactly three critical surgical questions:
  1. Infiltration Risk — wider margins needed?
  2. Vascular Aggression — intraoperative bleeding risk?
  3. Adjuvant Timeline — optimal post-op chemo window?

This is the demo wow moment — a UI that looks like a
tactical military dashboard, not a tech demo.
============================================================
"""

from typing import Dict, Any
from agents.rag_chain import TP53RAGChain
from utils.logger import log


SURGICAL_BRIEF_PROMPT = """You are a precision oncology surgical intelligence system 
providing a pre-incision tactical briefing to a surgical team. You have 90 seconds of their 
attention before they enter the sterile field.

Output ONLY the following three sections. No introductions. No fluff. No summaries.
Use direct, imperative language. Every word must justify surgical action.

---

## ⚠ INFILTRATION RISK
Based on the detected TP53 mutations, state:
- Is this mutation associated with higher local tissue invasiveness?
- Recommended surgical margin adjustment (standard / wider / significantly wider)
- Specific mutation driving this recommendation

## 🩸 VASCULAR AGGRESSION  
Based on the detected TP53 mutations, state:
- Does this variant upregulate angiogenic pathways?
- Intraoperative bleeding risk: LOW / MODERATE / HIGH
- Specific preparation recommendations for the surgical team

## 💊 ADJUVANT TIMELINE
Based on structural and functional mutation analysis, state:
- Optimal window for initiating post-operative chemotherapy
- Which agents are structurally relevant to detected mutations
- Which standard agents may show reduced efficacy (resistance flags)

---

CRITICAL: Every recommendation must be directly tied to the specific mutations provided.
Do not give generic oncology advice. Be mutation-specific. Be surgical-grade precise."""


# HTML template for the surgical dashboard UI
SURGICAL_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>PRE-INCISION BRIEF · TP53</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #000;
    color: #fff;
    font-family: 'Courier New', monospace;
    min-height: 100vh;
    padding: 32px;
  }}
  .header {{
    border-bottom: 3px solid #ff3366;
    padding-bottom: 16px;
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .title {{
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    color: #ff3366;
  }}
  .subtitle {{
    font-size: 0.8rem;
    color: #666;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}
  .timestamp {{
    font-size: 0.85rem;
    color: #444;
    text-align: right;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }}
  .card {{
    border: 1px solid #333;
    padding: 24px;
    border-radius: 4px;
    background: #0a0a0a;
  }}
  .card-high {{ border-top: 4px solid #ff3366; }}
  .card-moderate {{ border-top: 4px solid #ff9f43; }}
  .card-low {{ border-top: 4px solid #00ff88; }}
  .card-label {{
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 12px;
  }}
  .card-title {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 16px;
    color: #fff;
  }}
  .card-content {{
    font-size: 0.85rem;
    line-height: 1.7;
    color: #b0b0b0;
    white-space: pre-wrap;
  }}
  .mutations-bar {{
    background: #0a0a0a;
    border: 1px solid #333;
    padding: 16px 24px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .mut-chip {{
    padding: 6px 16px;
    border-radius: 2px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
  }}
  .mut-hotspot {{
    background: rgba(255,51,102,0.2);
    color: #ff3366;
    border: 1px solid rgba(255,51,102,0.4);
  }}
  .mut-other {{
    background: rgba(255,159,67,0.15);
    color: #ff9f43;
    border: 1px solid rgba(255,159,67,0.3);
  }}
  .alert-bar {{
    background: rgba(255,51,102,0.1);
    border: 1px solid rgba(255,51,102,0.3);
    padding: 14px 24px;
    margin-bottom: 24px;
    font-size: 0.85rem;
    color: #ff3366;
    letter-spacing: 0.05em;
  }}
  .footer {{
    border-top: 1px solid #222;
    padding-top: 16px;
    font-size: 0.65rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="title">PRE-INCISION BRIEF</div>
    <div class="subtitle">TP53 Genomic Surgical Intelligence · {accession}</div>
  </div>
  <div class="timestamp">
    Generated: {timestamp}<br>
    Platform: TP53 RAG v1.0 · Gemma 4 · Local
  </div>
</div>

{alert_bar}

<div class="mutations-bar">
  <span style="color:#666;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;
               align-self:center;">MUTATIONS:</span>
  {mutation_chips}
</div>

<div class="grid">
  <div class="card {infiltration_class}">
    <div class="card-label">01 · Infiltration Risk</div>
    <div class="card-title">⚠ Surgical Margins</div>
    <div class="card-content">{infiltration_content}</div>
  </div>
  <div class="card {vascular_class}">
    <div class="card-label">02 · Vascular Aggression</div>
    <div class="card-title">🩸 Bleeding Risk</div>
    <div class="card-content">{vascular_content}</div>
  </div>
  <div class="card {adjuvant_class}">
    <div class="card-label">03 · Adjuvant Timeline</div>
    <div class="card-title">💊 Post-Op Protocol</div>
    <div class="card-content">{adjuvant_content}</div>
  </div>
</div>

<div class="footer">
  ⚠ FOR SURGICAL DECISION SUPPORT ONLY · All recommendations require attending surgeon
  confirmation · Generated by AI — not a substitute for clinical judgment ·
  TP53 RAG Platform · Powered by Gemma 4 (Ollama) · 100% Local Processing
</div>

</body>
</html>"""


class SurgicalBriefAgent:
    """
    Agent #10 — Pre-Incision Surgical Brief.

    Generates a high-contrast, large-font tactical briefing
    designed for operating room display. Answers three
    critical surgical questions in 90 seconds of reading.
    """

    def __init__(self, rag_chain: TP53RAGChain):
        self.rag_chain = rag_chain
        log.info("SurgicalBriefAgent (#10) initialised")

    def generate(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate pre-incision surgical brief.

        Returns:
            Dict with brief_text, html_dashboard, risk_levels
        """
        mutations = pipeline_data.get("mutations", [])
        accession = pipeline_data.get("accession", "TP53")

        log.info(f"Generating surgical brief for {accession} "
                 f"with {len(mutations)} mutations")

        # Generate brief via RAG
        result = self.rag_chain.query(
            question=(
                "Generate a pre-incision surgical brief answering: "
                "infiltration risk and margin recommendation, "
                "vascular aggression and bleeding risk level, "
                "adjuvant chemotherapy timeline and agent selection."
            ),
            pipeline_data=pipeline_data,
            agent_type="clinical_interpretation",
        )

        brief_text = result["answer"]

        # Parse risk levels from brief
        risk_levels = self._parse_risk_levels(brief_text)

        # Generate HTML dashboard
        html = self._generate_dashboard(
            brief_text, mutations, accession, risk_levels
        )

        # Save dashboard
        from pathlib import Path
        from datetime import datetime
        output_dir = Path("data/surgical_briefs")
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = output_dir / f"brief_{accession}_{ts}.html"
        html_path.write_text(html, encoding="utf-8")
        log.info(f"Surgical brief saved: {html_path}")

        return {
            "brief_text": brief_text,
            "html_dashboard": html,
            "html_path": str(html_path),
            "risk_levels": risk_levels,
            "sources": result.get("sources", []),
        }

    def _parse_risk_levels(self, brief_text: str) -> Dict[str, str]:
        """Extract risk levels from brief text."""
        text_upper = brief_text.upper()
        return {
            "infiltration": "high" if "WIDER" in text_upper or "SIGNIFICANTLY" in text_upper
                           else "moderate" if "ADJUST" in text_upper else "low",
            "vascular": "high" if "HIGH" in text_upper and "BLEED" in text_upper
                       else "moderate" if "MODERATE" in text_upper else "low",
            "adjuvant": "high" if "RESISTANCE" in text_upper else "moderate",
        }

    def _generate_dashboard(
        self,
        brief_text: str,
        mutations: list,
        accession: str,
        risk_levels: Dict,
    ) -> str:
        """Generate the HTML surgical dashboard."""
        from datetime import datetime

        # Parse sections from brief
        sections = self._parse_sections(brief_text)

        # Mutation chips
        hotspot_labels = {
            "R175H", "R248W", "R248Q", "R273H",
            "R273C", "G245S", "R249S", "R282W"
        }
        chips = ""
        has_hotspot = False
        for m in mutations:
            aa = m.get("amino_acid_change", str(m))
            is_hot = aa in hotspot_labels
            if is_hot:
                has_hotspot = True
            chips += (
                f'<span class="mut-chip '
                f'{"mut-hotspot" if is_hot else "mut-other"}">'
                f'{aa}</span>'
            )

        alert_bar = ""
        if has_hotspot:
            alert_bar = (
                '<div class="alert-bar">🚨 CRITICAL: HOTSPOT MUTATION(S) DETECTED — '
                'REVIEW ALL MARGIN AND ADJUVANT DECISIONS WITH ATTENDING ONCOLOGIST '
                'BEFORE PROCEEDING</div>'
            )

        risk_class = {
            "high": "card-high",
            "moderate": "card-moderate",
            "low": "card-low",
        }

        return SURGICAL_DASHBOARD_HTML.format(
            accession=accession,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            alert_bar=alert_bar,
            mutation_chips=chips if chips else "<span style='color:#444'>None detected</span>",
            infiltration_class=risk_class.get(risk_levels["infiltration"], "card-moderate"),
            infiltration_content=sections.get("infiltration", brief_text[:300]),
            vascular_class=risk_class.get(risk_levels["vascular"], "card-moderate"),
            vascular_content=sections.get("vascular", "See full brief."),
            adjuvant_class=risk_class.get(risk_levels["adjuvant"], "card-moderate"),
            adjuvant_content=sections.get("adjuvant", "See full brief."),
        )

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse three sections from brief text."""
        sections = {}
        current = None
        buffer = []

        for line in text.split("\n"):
            line_lower = line.lower()
            if "infiltration" in line_lower or "margin" in line_lower:
                if current and buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = "infiltration"
                buffer = []
            elif "vascular" in line_lower or "bleeding" in line_lower:
                if current and buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = "vascular"
                buffer = []
            elif "adjuvant" in line_lower or "chemotherapy" in line_lower:
                if current and buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = "adjuvant"
                buffer = []
            elif current and line.strip():
                buffer.append(line.strip())

        if current and buffer:
            sections[current] = "\n".join(buffer).strip()

        return sections
