"""
============================================================
TP53 RAG Platform - Multi-Agent Dispatcher
============================================================
This is what transforms the single-tool TP53 pipeline into
a PLATFORM AI — serving multiple specialised functions from
one shared Gemma 4 inference layer.

Architecture:
  TP53 pipeline output
    → AgentDispatcher
      → MutationAgent      (interprets mutations)
      → ORFAgent           (interprets open reading frames)
      → PhylogeneticsAgent (interprets cross-species data)
      → DomainAgent        (interprets protein domains)
      → ClinicalAgent      (clinical significance)
      → ReportAgent        (synthesises all findings)
    → Unified platform response

This is the key architectural innovation for the hackathon:
one model (Gemma 4) serving six specialised clinical/research
functions via intelligent routing and RAG grounding.
============================================================
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agents.rag_chain import TP53RAGChain
from knowledge_base.vector_store import TP53VectorStore
from utils.logger import log


@dataclass
class AgentResult:
    """Structured result from a single agent."""
    agent: str
    question: str
    answer: str
    sources: List[Dict]
    success: bool
    error: Optional[str] = None


class AgentDispatcher:
    """
    Dispatches TP53 pipeline results to multiple specialised agents
    and aggregates responses into a unified platform output.

    This is the core of the platform AI pattern — one system,
    many functions, all powered by local Gemma 4.
    """

    # Predefined questions each agent answers automatically
    # when given pipeline output. This enables fully automated
    # analysis without user input — key for n8n integration.
    AUTOMATIC_QUERIES = {
        "mutation_analysis": (
            "Analyse the detected mutations. Classify each as hotspot or non-hotspot, "
            "describe their functional impact, clinical associations, and therapeutic implications."
        ),
        "orf_analysis": (
            "Interpret the discovered open reading frames. Which known p53 isoforms do they "
            "correspond to? What is their biological significance?"
        ),
        "phylogenetic_analysis": (
            "Interpret the cross-species phylogenetic analysis. What does the conservation "
            "pattern reveal about functionally critical positions?"
        ),
        "domain_annotation": (
            "Interpret the protein domain annotations. Describe each domain's function and "
            "how the identified domains relate to known p53 biology."
        ),
        "clinical_interpretation": (
            "Provide clinical interpretation of these findings. What is the clinical "
            "significance? What cancer associations exist? What are the therapeutic implications?"
        ),
    }

    def __init__(self, vector_store: TP53VectorStore):
        self.rag_chain = TP53RAGChain(vector_store=vector_store)
        log.info("AgentDispatcher initialised with 6 specialised agents")

    def dispatch_single(
        self,
        agent_type: str,
        pipeline_data: Dict[str, Any],
        custom_question: Optional[str] = None,
    ) -> AgentResult:
        """
        Dispatch to a single agent.

        Args:
            agent_type: One of mutation_analysis, orf_analysis, phylogenetic_analysis,
                       domain_annotation, clinical_interpretation, report_generation
            pipeline_data: TP53 pipeline output dict
            custom_question: Override the default automatic query

        Returns:
            AgentResult with grounded Gemma 4 response
        """
        question = custom_question or self.AUTOMATIC_QUERIES.get(
            agent_type,
            f"Analyse the provided TP53 data from a {agent_type} perspective."
        )

        try:
            result = self.rag_chain.query(
                question=question,
                pipeline_data=pipeline_data,
                agent_type=agent_type,
            )
            return AgentResult(
                agent=agent_type,
                question=question,
                answer=result["answer"],
                sources=result["sources"],
                success=True,
            )
        except Exception as e:
            log.error(f"Agent '{agent_type}' failed: {e}")
            return AgentResult(
                agent=agent_type,
                question=question,
                answer="",
                sources=[],
                success=False,
                error=str(e),
            )

    def dispatch_all(
        self,
        pipeline_data: Dict[str, Any],
        include_report: bool = True,
    ) -> Dict[str, AgentResult]:
        """
        Dispatch pipeline output to ALL agents simultaneously.
        This is the full platform AI run — six analyses from one call.

        Args:
            pipeline_data: Complete TP53 pipeline output
            include_report: Whether to generate a synthesis report after agents

        Returns:
            Dict mapping agent_name → AgentResult
        """
        log.info("═" * 60)
        log.info("  TP53 Multi-Agent Platform — Full Dispatch")
        log.info(f"  Agents: {len(self.AUTOMATIC_QUERIES)} + report")
        log.info("═" * 60)

        results = {}

        for agent_type, question in self.AUTOMATIC_QUERIES.items():
            log.info(f"Dispatching to agent: {agent_type}")
            results[agent_type] = self.dispatch_single(
                agent_type=agent_type,
                pipeline_data=pipeline_data,
            )

        # Final synthesis report
        if include_report:
            log.info("Generating synthesis report...")
            # Collect successful agent answers for the report
            agent_summaries = {
                name: result.answer[:500]
                for name, result in results.items()
                if result.success and result.answer
            }

            report_data = {
                **pipeline_data,
                "agent_analysis_summaries": agent_summaries,
            }

            results["report_generation"] = self.dispatch_single(
                agent_type="report_generation",
                pipeline_data=report_data,
                custom_question=(
                    "Generate a comprehensive TP53 analysis report synthesising all findings "
                    "from the mutation analysis, ORF discovery, phylogenetic analysis, "
                    "domain annotation, and clinical interpretation agents."
                ),
            )

        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        log.info(f"Dispatch complete: {successful}/{len(results)} agents succeeded")

        return results

    def interactive_query(
        self,
        question: str,
        pipeline_data: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Handle a free-form user question about any aspect of TP53 analysis.
        Routes automatically to the best agent.

        This is the conversational interface — users type natural language,
        the platform routes and responds intelligently.
        """
        result = self.rag_chain.query(
            question=question,
            pipeline_data=pipeline_data,
        )
        return AgentResult(
            agent=result["agent_used"],
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            success=True,
        )

    def format_platform_output(
        self,
        results: Dict[str, AgentResult],
        include_sources: bool = False,
    ) -> str:
        """
        Format all agent results into a human-readable platform report.
        Used for CLI output and the Streamlit app display.
        """
        sections = []
        agent_labels = {
            "mutation_analysis": "🧬 Mutation Analysis",
            "orf_analysis": "🔬 ORF & Isoform Analysis",
            "phylogenetic_analysis": "🌳 Phylogenetic Analysis",
            "domain_annotation": "🏗️ Protein Domain Annotation",
            "clinical_interpretation": "🏥 Clinical Interpretation",
            "report_generation": "📋 Comprehensive Report",
        }

        for agent_type, label in agent_labels.items():
            if agent_type not in results:
                continue

            result = results[agent_type]
            section = [f"\n{'═' * 60}", f"  {label}", f"{'═' * 60}"]

            if result.success:
                section.append(result.answer)
            else:
                section.append(f"[Agent failed: {result.error}]")

            if include_sources and result.sources:
                section.append("\n  Sources:")
                for src in result.sources[:3]:
                    section.append(
                        f"  • [{src['category']}] {src['content_preview'][:100]}... "
                        f"(relevance: {src['relevance_score']})"
                    )

            sections.append("\n".join(section))

        return "\n".join(sections)
