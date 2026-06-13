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
      → PathologyAgent     (H&E slide analysis)
      → TNMAgent           (AJCC/UICC staging + Kenya roadmap)
    → Unified platform response

One model (Gemma 4 e2B) serving 16 specialised clinical/research
functions via intelligent routing, RAG grounding, and thread-safe
shared state (recursive/telepathic inter-agent communication).
============================================================
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agents.rag_chain import TP53RAGChain
from knowledge_base.vector_store import TP53VectorStore
from utils.logger import log
from utils.shared_state import shared_state


def _demo_mode() -> bool:
    """True when the DEMO_MODE env flag is set. In demo mode the dispatcher
    returns clearly-labelled canned answers for fast, offline video demos
    instead of running the (slower) real RAG pipeline. Default = OFF (real)."""
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


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
        log.info("AgentDispatcher initialised with 16 specialised agents")

    def dispatch_single(
        self,
        agent_type: str,
        pipeline_data: Dict[str, Any],
        custom_question: Optional[str] = None,
    ) -> AgentResult:
        """Run one agent. Real, grounded RAG by default; only DEMO_MODE returns
        the clearly-labelled canned answers below (for fast offline demos)."""
        question = custom_question or self.AUTOMATIC_QUERIES.get(
            agent_type,
            f"Analyse the provided TP53 data from a {agent_type} perspective."
        )

        # ── Real path (DEFAULT): full grounded, safety-checked RAG pipeline.
        # rag_chain.query already guarantees a non-empty, PII-scrubbed answer
        # with self-correction + zero-result fallback.
        if not _demo_mode():
            try:
                result = self.rag_chain.query(
                    question=question,
                    pipeline_data=pipeline_data,
                    agent_type=agent_type,
                )
                answer = (result.get("answer") or "").strip()
                return AgentResult(
                    agent=result.get("agent_used", agent_type),
                    question=question,
                    answer=answer,
                    sources=result.get("sources", []),
                    success=bool(answer),
                )
            except Exception as e:
                log.error(f"Agent '{agent_type}' failed: {e}")
                return AgentResult(
                    agent=agent_type, question=question,
                    answer=f"[{agent_type}] agent unavailable: {e}",
                    sources=[], success=False, error=str(e),
                )

        # ── DEMO_MODE only: canned, clearly-labelled illustrative answers ──
        mock_answers = {
            "mutation_analysis": (
                "🧬 MUTATION ANALYSIS SUMMARY:\n"
                "• Detected Mutation: R248W (Position 742, codon change CGG→TGG)\n"
                "• Classification: HIGH-CONFIDENCE PATHOGENIC HOTSPOT\n"
                "• Structural Impact: This is a direct contact mutation within the core DNA-binding domain. "
                "It disrupts the direct interaction with the DNA minor groove without changing the overall protein folding conformation.\n"
                "• Therapeutic Implication: Candidate variant eligible for target rescue via APR-246 reactivation therapy."
            ),
            "orf_analysis": (
                "🔬 ORF ANALYSIS SUMMARY:\n"
                "• Primary Frame Identified: +1 frame (positions 203-1384, length 1181 bp).\n"
                "• Isoform Match: Corresponds directly to the full-length canonical p53α protein (393 amino acids).\n"
                "• Alternative Segment Detected: +2 open reading frame presents an active isoform fragment containing truncated "
                "regulatory structures with altered transactivation capability."
            ),
            "phylogenetic_analysis": (
                "🔬 PHYLOGENETIC CONSERVATION PROFILE:\n"
                "• Alignment Summary: Analyzed sequences across 5 benchmark species (H. sapiens, P. troglodytes, M. musculus, R. norvegicus, D. rerio).\n"
                "• Critical Finding: Codons 175, 248, and 273 display a 100% strict conservation score across evolutionary tracks. "
                "Any alteration at position 248 introduces radical evolutionary divergence and confirms functional vulnerability."
            ),
            "domain_annotation": (
                "🧬 DOMAIN ANNOTATION TRACK:\n"
                "• Domain Mapped: Core DNA-Binding Domain (Pfam database: PF00870, spanning residues 94-292).\n"
                "• Structural Integrity: R248W variant maps squarely onto the minor-groove binding loop. This mapping explains why "
                "structural activity drops off sharply while overall tetramerization architecture remains stable."
            ),
            "clinical_interpretation": (
                "⚕️ CLINICAL SIGNIFIED PROFILE:\n"
                "• Pathogenicity Score: Highly Pathogenic (Validated across ClinVar and IARC curated datasets).\n"
                "• Disease Correlation: Strictly tied to classical Li-Fraumeni Syndrome phenotypes and highly aggressive somatically acquired cancers.\n"
                "• Therapeutic Roadmap: High resistance tracking for standard DNA-damaging chemotherapies (Carboplatin/Doxorubicin). "
                "Urgent protocol escalation to APR-246 or emerging zinc rescue compounds is recommended."
            ),
            "report_generation": (
                "📋 UNIFIED CLINICAL ONCOLOGY DOSSIER:\n"
                "This report synthesises multi-agent data tracking for sample NM_000546. The tumor profile contains a core cluster of "
                "deleterious variants, driven primarily by the R248W DNA contact hotspot. Complete multi-agent pipeline profiling confirms "
                "loss of tumor suppression functionality, structural context vulnerability within the core binding site, and an evolutionary "
                "conservation index that screams high-risk pathogenesis. Recommendation: Flag for structural reactivation therapeutics."
            ),
        }

        import time
        time.sleep(0.4)  # smooth cadence for the demo only
        banner = ("[DEMO DATA] Illustrative canned output for an offline demo "
                  "(DEMO_MODE is ON) — NOT real analysis. Unset DEMO_MODE for "
                  "real grounded results.\n\n")
        return AgentResult(
            agent=agent_type,
            question=question,
            answer=banner + mock_answers.get(
                agent_type, "Analysis complete (demo placeholder)."),
            sources=[{"document": "DEMO — not a real source", "relevance": 0.0}],
            success=True,
        )

    def dispatch_all(
        self,
        pipeline_data: Dict[str, Any],
        include_report: bool = True,
        slide_image_path: Optional[str] = None,
    ) -> Dict[str, AgentResult]:
        """
        Dispatch pipeline output to ALL agents.
        Includes pathology vision (Agent #15) and TNM staging (Agent #16).

        Agent communication is recursive/telepathic — agents write structured
        Python objects directly into shared_state, and downstream agents read
        from it without any text serialisation step.

        Args:
            pipeline_data: Complete TP53 pipeline output
            include_report: Whether to generate synthesis report after agents
            slide_image_path: Optional path to H&E slide image for pathology agent

        Returns:
            Dict mapping agent_name → AgentResult
        """
        log.info("═" * 60)
        log.info("  TP53 Multi-Agent Platform — Full Dispatch (16 agents)")
        log.info("═" * 60)

        results = {}
        shared_state.update("pipeline_data", pipeline_data)

        # ── Core RAG agents ───────────────────────────────────────
        for agent_type, question in self.AUTOMATIC_QUERIES.items():
            log.info(f"Dispatching to agent: {agent_type}")
            results[agent_type] = self.dispatch_single(
                agent_type=agent_type,
                pipeline_data=pipeline_data,
            )
            shared_state.update_agent_output(
                agent_type,
                results[agent_type].__dict__
            )

        # ── Agent #15: Pathology Vision ───────────────────────────
        # Runs if a slide image path is provided.
        # Result written to shared_state for TNM agent to read directly.
        pathology_result = {}
        if slide_image_path:
            try:
                from agents.pathology_vision import PathologyVisionAgent
                log.info("Dispatching to agent: pathology_vision")
                pv_agent = PathologyVisionAgent(rag_chain=self.rag_chain)
                pathology_result = pv_agent.process_slide(
                    image_path=slide_image_path,
                    mutation_data=pipeline_data,
                )
                shared_state.update_agent_output("pathology_vision", pathology_result)
                results["pathology_vision"] = AgentResult(
                    agent="pathology_vision",
                    question="Analyse H&E slide and correlate with TP53 mutations.",
                    answer=pathology_result.get("llm_narration", "Pathology analysis complete."),
                    sources=[],
                    success=pathology_result.get("success", False),
                )
                log.info(f"Pathology agent complete — top tissue: {pathology_result.get('top_tissue', 'unknown')}")
            except Exception as e:
                log.warning(f"Pathology agent failed: {e}")
                results["pathology_vision"] = AgentResult(
                    agent="pathology_vision",
                    question="Pathology slide analysis",
                    answer=f"Pathology agent unavailable: {e}",
                    sources=[],
                    success=False,
                    error=str(e),
                )
        else:
            log.info("Pathology agent skipped — no slide image provided")

        # ── Agent #16: TNM Staging ────────────────────────────────
        # Reads pathology_result directly from shared_state.
        # No text serialisation — pure Python object passing.
        # Falls back gracefully if pathology was not run.
        try:
            from agents.tnm_staging import TNMStagingAgent
            log.info("Dispatching to agent: tnm_staging")

            # Read from shared_state (telepathic communication pattern)
            stored_pathology = shared_state.get_all_outputs().get(
                "pathology_vision", pathology_result
            )

            # If no pathology slide was processed, use a minimal fallback
            # so TNM can still run rule-based staging from mutation + VAF alone
            if not stored_pathology or not stored_pathology.get("success"):
                stored_pathology = {
                    "success": False,
                    "top_tissue": "Unknown",
                    "tissue_classifications": [],
                    "mutation_correlations": [],
                }

            tnm_agent = TNMStagingAgent(rag_chain=self.rag_chain)
            tnm_result = tnm_agent.stage(
                pathology_result=stored_pathology,
                pipeline_data=pipeline_data,
            )

            # Write TNM result back to shared_state
            shared_state.update_agent_output("tnm_staging", tnm_result)

            stage_group = tnm_result.get("stage_group", "Unknown")
            t = tnm_result.get("T", {}).get("code", "?")
            n = tnm_result.get("N", {}).get("code", "?")
            m = tnm_result.get("M", {}).get("code", "?")

            results["tnm_staging"] = AgentResult(
                agent="tnm_staging",
                question="Stage cancer using AJCC/UICC 8th Edition TNM criteria.",
                answer=(
                    f"🏥 TNM STAGING RESULT:\n"
                    f"• T Stage: {t} — {tnm_result.get('T', {}).get('description', '')}\n"
                    f"• N Stage: {n} — {tnm_result.get('N', {}).get('description', '')}\n"
                    f"• M Stage: {m} — {tnm_result.get('M', {}).get('description', '')}\n"
                    f"• Overall Stage: {stage_group}\n"
                    f"• Equity Flag: {tnm_result.get('equity_flag') or 'None'}\n\n"
                    f"{tnm_result.get('llm_narration', '')}\n\n"
                    f"📋 Next Steps ({len(tnm_result.get('next_steps', []))} actions):\n" +
                    "\n".join(
                        f"  {s['priority']}. {s['action']} → {s['kenya_resource']}"
                        for s in tnm_result.get("next_steps", [])[:3]
                    )
                ),
                sources=[],
                success=True,
            )
            log.info(f"TNM staging complete — Stage {stage_group} ({t} {n} {m})")

        except Exception as e:
            log.warning(f"TNM staging agent failed: {e}")
            results["tnm_staging"] = AgentResult(
                agent="tnm_staging",
                question="TNM staging",
                answer=f"TNM staging unavailable: {e}",
                sources=[],
                success=False,
                error=str(e),
            )

        # ── Synthesis report ──────────────────────────────────────
        if include_report:
            log.info("Generating synthesis report...")
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
                    "domain annotation, clinical interpretation, pathology vision, "
                    "and TNM staging agents."
                ),
            )

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
        """
        sections = []
        agent_labels = {
            "mutation_analysis":     "🧬 Mutation Analysis",
            "orf_analysis":          "🔬 ORF & Isoform Analysis",
            "phylogenetic_analysis": "🌳 Phylogenetic Analysis",
            "domain_annotation":     "🏗️ Protein Domain Annotation",
            "clinical_interpretation": "🏥 Clinical Interpretation",
            "pathology_vision":      "🔬 Pathology Slide Analysis",
            "tnm_staging":           "📍 TNM Staging & Clinical Roadmap",
            "report_generation":     "📋 Comprehensive Report",
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
                        f"  • [{src.get('category','?')}] "
                        f"{str(src.get('content_preview',''))[:100]}... "
                        f"(relevance: {src.get('relevance_score', src.get('relevance', '?'))})"
                    )
            sections.append("\n".join(section))

        return "\n".join(sections)