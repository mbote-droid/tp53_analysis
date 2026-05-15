"""
TP53 RAG Platform - Drug Discovery Insights Agent (#8)
Therapeutic insights grounded in Kenya real-world access.
"""

from typing import Dict, Any, List
from agents.rag_chain import TP53RAGChain
from utils.logger import log

KENYA_AVAILABLE_DRUGS = {
    "5-fluorouracil": {"keml": True, "notes": "Available at referral hospitals"},
    "cisplatin": {"keml": True, "notes": "Available at county referral hospitals"},
    "doxorubicin": {"keml": True, "notes": "Available at KNH and major referral hospitals"},
    "paclitaxel": {"keml": True, "notes": "Available at national referral hospitals"},
    "carboplatin": {"keml": True, "notes": "Available at national referral hospitals"},
    "tamoxifen": {"keml": True, "notes": "Widely available"},
    "APR-246": {"keml": False, "notes": "Clinical trial only — not available in Kenya"},
    "idasanutlin": {"keml": False, "notes": "Clinical trial only"},
    "PC14586": {"keml": False, "notes": "Phase 1/2 trial only"},
}

RESISTANCE_PROMPT = """You are a clinical pharmacologist at a Kenyan oncology centre.
For the detected TP53 mutations:
1. Which standard agents show reduced efficacy? (5-FU, cisplatin, doxorubicin, paclitaxel)
2. Mechanistic basis for resistance
3. Best alternative regimen available in Kenya
Be direct and clinically actionable."""


class DrugDiscoveryAgent:
    """Agent #8 — Drug Discovery and Therapeutic Insights."""

    def __init__(self, rag_chain: TP53RAGChain):
        self.rag_chain = rag_chain
        log.info("DrugDiscoveryAgent (#8) initialised")

    def analyse(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        mutations = pipeline_data.get("mutations", [])
        if not mutations:
            return {"error": "No mutations — drug analysis not applicable"}

        labels = [m.get("amino_acid_change", str(m)) for m in mutations]
        log.info(f"Drug analysis for: {labels}")

        keml_context = self._keml_context()
        enriched = {**pipeline_data, "kenya_medicines_context": keml_context}

        drug_result = self.rag_chain.query(
            question=(
                f"Therapeutic implications for TP53 mutations: {', '.join(labels)}. "
                f"Include Kenya Essential Medicines List availability for all recommendations."
            ),
            pipeline_data=enriched,
            agent_type="clinical_interpretation",
        )

        resistance = self._resistance_profile(enriched)
        novel = self._novel_angles(enriched, labels)

        return {
            "mutations_analysed": labels,
            "drug_insights": drug_result["answer"],
            "resistance_profile": resistance,
            "novel_therapeutic_angles": novel,
            "kenya_availability": keml_context,
            "sources": drug_result.get("sources", []),
        }

    def _keml_context(self) -> str:
        available = [f"  ✓ {d.title()} — {i['notes']}"
                     for d, i in KENYA_AVAILABLE_DRUGS.items() if i["keml"]]
        unavailable = [f"  ✗ {d} — {i['notes']}"
                       for d, i in KENYA_AVAILABLE_DRUGS.items() if not i["keml"]]
        return (
            "KENYA ESSENTIAL MEDICINES (KEML) CONTEXT:\n"
            "Available at Kenyan referral hospitals:\n" + "\n".join(available) +
            "\n\nNot available in Kenya:\n" + "\n".join(unavailable) +
            "\n\nPrioritise recommendations from available list."
        )

    def _resistance_profile(self, data: Dict) -> str:
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_messages([
                ("system", RESISTANCE_PROMPT),
                ("human", "Mutation profile:\n{data}"),
            ])
            return (prompt | self.rag_chain.llm | StrOutputParser()).invoke(
                {"data": self.rag_chain._format_pipeline_data(data)}
            )
        except Exception as e:
            return f"Resistance analysis failed: {e}"

    def _novel_angles(self, data: Dict, labels: List[str]) -> str:
        try:
            return self.rag_chain.query(
                question=f"Novel therapeutic strategies for {', '.join(labels)}: "
                         "structural vulnerabilities, synthetic lethality, neoantigen potential.",
                pipeline_data=data,
                agent_type="clinical_interpretation",
            )["answer"]
        except Exception as e:
            return f"Novel angles failed: {e}"
