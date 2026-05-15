"""
============================================================
TP53 RAG Platform - Multilingual Patient Report Agent
============================================================
Generates clinically accurate, culturally appropriate
reports in English and Swahili.

IMPORTANT — This is NOT a translation wrapper.
The Swahili report is generated DIRECTLY in Swahili
using Kenya Ministry of Health terminology, cultural
analogies, and medically accurate framing.

Cultural design principles:
  - Kenya MoH standard terminology throughout
  - Terrifying jargon replaced with cultural analogies
  - Warm, communal, family-inclusive tone
  - Automatic Swahili medical glossary appended
  - Acknowledges family role in health decisions
============================================================
"""

from typing import Dict, Any, Optional
from agents.rag_chain import TP53RAGChain
from utils.logger import log


CLINICAL_REPORT_PROMPT = """You are a senior oncologist preparing a clinical summary for 
a multidisciplinary tumor board at a Kenyan referral hospital.

Based on the TP53 analysis results provided, write a concise clinical report:

1. KEY FINDINGS: Most significant mutations and classifications
2. FUNCTIONAL IMPACT: How mutations affect p53 protein function
3. CANCER ASSOCIATIONS: Relevant cancer types and prognosis
4. THERAPEUTIC CONSIDERATIONS: Include BOTH international agents AND 
   locally available alternatives from Kenya Essential Medicines List
5. SURGICAL IMPLICATIONS: Margin considerations, bleeding risk, adjuvant timing
6. RECOMMENDED ACTIONS: Next steps achievable in Kenyan clinical context

Be precise and clinically actionable. Note when recommendations require 
resources not available in Kenya. Maximum 400 words."""


PATIENT_ENGLISH_PROMPT = """You are a compassionate Kenyan doctor explaining genetic 
test results to a patient and their family members who are present.

Write a warm, clear explanation:
1. WHAT WAS TESTED: Simple explanation of what a genetic test is
2. WHAT WE FOUND: Results without causing panic
3. WHAT THIS MEANS: Practical health implications in everyday language
4. WHAT HAPPENS NEXT: Clear, achievable steps available in Kenya
5. SUPPORT: Acknowledge difficulty and that the team is a partner

Avoid: mutation, pathogenic, variant, frameshift, codon, malignant.
Replace with plain equivalents. Address patient AND family.
Be honest but compassionate. Maximum 300 words."""


PATIENT_SWAHILI_PROMPT = """Wewe ni daktari mwenye huruma katika hospitali ya Kenya 
unayeelezea matokeo ya kipimo cha maumbile kwa mgonjwa na familia yake.

Andika maelezo ya moja kwa moja kwa Kiswahili safi, ukitumia:
- Istilahi za Wizara ya Afya Kenya (MoH Kenya)
- Mifano ya kila siku kuelezea dhana ngumu
- Lugha ya heshima na ya karibu
- Ukubalifu wa mfumo wa familia katika maamuzi ya afya

MWONGOZO WA UANDISHI:

1. KIPIMO KILIKUWA NINI:
   Elezea kwa mfano rahisi. Mfano: "Tumechunguza ramani ya 
   mwili wako — kama mpango wa ujenzi wa nyumba."

2. TULIPATA NINI:
   Elezea matokeo bila kutisha. Badilisha maneno magumu:
   - "mabadiliko ya jeni" badala ya "mutation"
   - "hitilafu ndogo katika mpango wa mwili" badala ya "pathogenic variant"
   - "seli zinakua haraka zaidi" badala ya "malignant proliferation"

3. MAANA YAK KWA AFYA YAKO:
   Elezea athari za vitendo kwa lugha ya kawaida.

4. HATUA ZINAZOFUATA:
   Taja hatua zinazowezekana KENYA — dawa zinazopatikana,
   hospitali za rufaa, msaada wa serikali.

5. UJUMBE WA KUTIA MOYO:
   Kumbuka kwamba afya ni safari ya pamoja — daktari,
   familia, na mgonjwa wote ni timu moja.

MWISHO — ongeza orodha fupi ya maneno magumu (Kiswahili = Kiingereza).

Maneno 300 upeo. Kiswahili safi — bila maneno ya Kiingereza isipokuwa lazima."""


SWAHILI_GLOSSARY = {
    "Jeni": "Gene — mpango wa maumbile wa mwili",
    "Kromosomu": "Chromosome — mkoba unaobeba maelekezo ya mwili",
    "Mabadiliko ya jeni": "Genetic mutation — hitilafu katika mpango wa mwili",
    "TP53": "Jeni linalomlinda mwili dhidi ya ukuaji mbaya wa seli",
    "Seli": "Cell — kioo kidogo cha mwili",
    "Uvimbe": "Tumour — mkusanyiko wa seli zinazokua bila mpango",
    "Dawa ya kuzuia saratani": "Chemotherapy — matibabu ya dawa kali",
    "Upasuaji": "Surgery — operesheni",
    "Rufaa": "Referral — kupelekwa kwa daktari mwingine au hospitali nyingine",
}


class MultilingualReportAgent:
    """
    Generates clinical and patient-facing reports in
    English and culturally nuanced Swahili.
    """

    def __init__(self, rag_chain: TP53RAGChain):
        self.rag_chain = rag_chain
        log.info("MultilingualReportAgent initialised")

    def generate(
        self,
        pipeline_data: Dict[str, Any],
        agent_results: Optional[Dict] = None,
        equity_context: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate all three reports."""
        enriched = dict(pipeline_data)
        if equity_context:
            enriched["african_equity_context"] = equity_context
        if agent_results:
            enriched["prior_findings"] = {
                name: (result.answer[:300] if hasattr(result, 'answer') else str(result))
                for name, result in agent_results.items()
                if hasattr(result, 'answer') and result.answer
            }

        log.info("Generating clinical report (English)...")
        clinical = self._query(CLINICAL_REPORT_PROMPT, enriched)

        log.info("Generating patient report (English)...")
        patient_en = self._query(PATIENT_ENGLISH_PROMPT, enriched)

        log.info("Generating patient report (Swahili — direct, culturally nuanced)...")
        patient_sw = self._query(PATIENT_SWAHILI_PROMPT, enriched)

        glossary = self._format_glossary()
        patient_sw_full = patient_sw + "\n\n" + glossary

        log.info("All multilingual reports generated ✓")
        return {
            "clinical_report_en": clinical,
            "patient_report_en": patient_en,
            "patient_report_sw": patient_sw_full,
            "glossary_sw": glossary,
        }

    def _query(self, system_prompt: str, pipeline_data: Dict) -> str:
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Analysis results:\n{data}\n\nGenerate the report now."),
            ])
            chain = prompt | self.rag_chain.llm | StrOutputParser()
            return chain.invoke({
                "data": self.rag_chain._format_pipeline_data(pipeline_data)
            })
        except Exception as e:
            log.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

    def _format_glossary(self) -> str:
        lines = ["\n---\n📖 KAMUSI YA MANENO YA KIMATIBABU (Medical Glossary):"]
        for sw, en in SWAHILI_GLOSSARY.items():
            lines.append(f"• {sw}: {en}")
        return "\n".join(lines)
