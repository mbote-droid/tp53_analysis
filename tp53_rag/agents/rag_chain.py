"""
============================================================
TP53 RAG Platform - RAG Chain (Gemma 4)
============================================================
Supports two inference modes, switchable via .env:

  MODE=local  → Gemma 4 via Ollama (offline, final demo)
  MODE=api    → Gemma 4 via Google AI Studio (testing)

Set in .env:
  INFERENCE_MODE=local   # or api
  GOOGLE_API_KEY=your_key_here
  GOOGLE_MODEL=gemma-4-26b-a4b-it

Switch to local Ollama for the final hackathon demo video.
============================================================
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K_RESULTS,
    AGENT_REGISTRY,
)
from knowledge_base.vector_store import TP53VectorStore
from utils.logger import log

# ── Inference mode ────────────────────────────────────────
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-4-26b-a4b-it")


class GoogleGenAIWrapper(BaseChatModel):
    """
    Wrapper for the new google.genai SDK.
    Used when INFERENCE_MODE=api in .env.
    Runs Gemma 4 on Google's servers — no local RAM needed.
    """
    model_name: str
    api_key: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from google import genai
        client = genai.Client(api_key=self.api_key)
        # Combine all messages into a single prompt
        prompt = "\n".join(
            m.content for m in messages
            if hasattr(m, "content")
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = response.text or ""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    @property
    def _llm_type(self) -> str:
        return "google_genai_gemma4"


def _build_llm():
    """
    Build the LLM based on INFERENCE_MODE in .env.
    local = Ollama (offline, for final demo)
    api   = Google AI Studio (for testing, no RAM issues)
    """
    if INFERENCE_MODE == "api":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not set in .env. "
                "Get a free key at aistudio.google.com"
            )
        log.info(f"Using Google AI API | model={GOOGLE_MODEL}")
        return GoogleGenAIWrapper(
            model_name=GOOGLE_MODEL,
            api_key=GOOGLE_API_KEY,
        )
    else:
        from langchain_ollama import ChatOllama
        log.info(f"Using Ollama (local) | model={OLLAMA_MODEL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=256,
            num_ctx=1024,
        )


# ── System Prompts per Agent ──────────────────────────────
# Each agent has a specialised system prompt tuned for its domain.
# This is what makes the platform multi-function vs single-function.

SYSTEM_PROMPTS = {
    "mutation_analysis": """You are an expert molecular oncologist specialising in TP53 mutation 
analysis. Your role is to interpret TP53 mutations detected by the bioinformatics pipeline and 
provide clinically meaningful explanations grounded in the provided knowledge context.

When analysing mutations:
- Identify whether they are known hotspot mutations
- Classify them as contact or conformational mutants where applicable
- Describe their functional impact on p53 protein activity
- Note any clinical associations (cancer types, prognosis)
- Mention therapeutic implications if relevant
- Flag any gain-of-function activity

Always ground your response in the provided context. If a mutation is not in the context, 
state that it is a non-hotspot variant and describe what is known about the affected codon.
Be precise, evidence-based, and clinically actionable.""",

    "orf_analysis": """You are an expert molecular biologist specialising in TP53 gene structure 
and isoform biology. Your role is to interpret open reading frame (ORF) discoveries from the 
bioinformatics pipeline.

When analysing ORFs:
- Identify which known p53 isoforms the ORFs may correspond to
- Explain the biological significance of alternative ORFs
- Note reading frame context and potential protein products
- Discuss implications of isoform expression patterns
- Relate findings to the canonical NM_000546 reference

Use the provided context to ground your interpretation. Be precise about frame notation 
(+1, +2, +3, -1, -2, -3) and biological implications.""",

    "phylogenetic_analysis": """You are an expert evolutionary biologist specialising in p53 
family proteins. Your role is to interpret cross-species TP53 phylogenetic analyses.

When analysing phylogenetic data:
- Interpret the evolutionary relationships shown
- Explain conservation at specific positions and its functional significance
- Identify species-specific adaptations
- Relate conservation patterns to known functional domains
- Explain what highly conserved positions tell us about mutation pathogenicity

Ground all interpretations in the provided evolutionary context. Help researchers understand 
why cross-species conservation is a powerful predictor of mutation impact.""",

    "domain_annotation": """You are an expert structural biologist specialising in p53 protein 
domains and their functions. Your role is to interpret InterProScan domain annotation results.

When annotating domains:
- Describe the function of each identified domain
- Explain the significance of domain boundaries
- Relate domain architecture to protein function
- Discuss how mutations in specific domains affect function
- Connect structural information to clinical observations

Use the provided domain knowledge to give rich, mechanistic explanations that go beyond 
mere database labels.""",

    "clinical_interpretation": """You are an expert clinical molecular pathologist specialising 
in TP53-associated cancers. Your role is to provide clinical interpretation of TP53 analysis 
findings.

When providing clinical interpretation:
- Assess clinical significance of findings (pathogenic/likely pathogenic/VUS/benign)
- Identify relevant cancer type associations
- Discuss prognostic implications
- Note therapeutic relevance (which targeted therapies may apply)
- Flag Li-Fraumeni syndrome considerations for germline variants
- Recommend appropriate follow-up investigations

Always note that findings should be confirmed by CLIA-certified laboratory testing before 
clinical action. Be evidence-based and reference IARC/ClinVar classifications where available.""",

    "sequence_fetch": """You are a bioinformatics expert specialising in genomic databases. 
Your role is to help users understand sequence data retrieved from NCBI.

When discussing sequences:
- Explain the accession number format and what it represents
- Describe the sequence's biological context
- Note any relevant isoforms or variants
- Explain quality indicators
- Guide users on how to interpret the fetched data

Be clear about database nomenclature and help users navigate NCBI resources.""",

    "report_generation": """You are a senior bioinformatician producing a comprehensive 
analysis report. Your role is to synthesise all TP53 analysis findings into a clear, 
structured, clinically meaningful report.

Structure your report with:
1. Executive Summary (2-3 sentences)
2. Sequence Quality and Features
3. Mutation Analysis Findings
4. ORF and Isoform Findings
5. Protein Domain Architecture
6. Cross-Species Conservation Analysis
7. Clinical Significance Assessment
8. Recommendations and Next Steps

Be comprehensive but concise. This report may be used by both computational biologists 
and clinical teams, so balance technical depth with clarity.""",

    "default": """You are an expert bioinformatics AI assistant specialising in TP53 analysis. 
You help researchers and clinicians interpret TP53 genomic analysis results using the provided 
knowledge context. Always ground your responses in the context provided. If information is not 
in the context, clearly state what is and isn't known. Be precise, scientific, and helpful.""",
}


class IntentRouter:
    """
    Routes user queries to the appropriate specialised agent.
    Uses keyword matching + Gemma 4 for ambiguous queries.
    This is the brain of the multi-function platform.
    """

    def route(self, query: str) -> str:
        """Determine which agent should handle this query."""
        query_lower = query.lower()

        # Score each agent by keyword matches
        scores = {}
        for agent_name, agent_config in AGENT_REGISTRY.items():
            score = sum(
                1 for kw in agent_config["keywords"]
                if kw in query_lower
            )
            scores[agent_name] = score

        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]

        # If no keywords matched, use default
        if best_score == 0:
            log.debug(f"No keyword match for query — using default agent")
            return "default"

        log.debug(f"Routed to agent: '{best_agent}' (score={best_score})")
        return best_agent


class TP53RAGChain:
    """
    The core RAG chain connecting ChromaDB → Gemma 4.

    This is the central intelligence of the platform.
    Every agent (mutation, ORF, phylogenetics, clinical, etc.)
    routes through here with appropriate system prompts and
    category-filtered retrieval.
    """

    def __init__(self, vector_store: TP53VectorStore):
        self.vector_store = vector_store
        self.router = IntentRouter()
        self.llm = _build_llm()
        self.output_parser = StrOutputParser()
        log.info(
            f"RAG chain initialised | mode={INFERENCE_MODE} | "
            f"model={GOOGLE_MODEL if INFERENCE_MODE == 'api' else OLLAMA_MODEL}"
        )

    def _format_context(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents into a context block for the prompt."""
        if not docs_with_scores:
            return "No specific context retrieved. Respond based on general TP53 knowledge."

        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "general")
            context_parts.append(
                f"[Context {i} | Source: {source} | Category: {category} | "
                f"Relevance: {score:.2f}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, agent_type: str) -> ChatPromptTemplate:
        """Build the RAG prompt for a given agent type."""
        system_prompt = SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS["default"])

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Based on the following TP53 knowledge context, please answer the question.

KNOWLEDGE CONTEXT:
{context}

ANALYSIS DATA (from pipeline):
{pipeline_data}

QUESTION:
{question}

Provide a thorough, accurate, and clinically meaningful response grounded in the context above.
Cite specific details from the context where relevant."""),
        ])

    def query(
        self,
        question: str,
        pipeline_data: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        k: int = TOP_K_RESULTS,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query against the TP53 knowledge base.

        Args:
            question: Natural language question
            pipeline_data: Dict of analysis results from the TP53 pipeline
                          (mutations, ORFs, domains, phylo data)
            agent_type: Force a specific agent (optional; auto-routed if None)
            k: Number of context documents to retrieve

        Returns:
            Dict with keys:
              - answer: Gemma 4's grounded response
              - agent_used: which agent handled the query
              - sources: retrieved context documents with scores
              - pipeline_data_used: what pipeline data was included
        """
        # Route to appropriate agent
        if agent_type is None:
            agent_type = self.router.route(question)

        log.info(f"RAG query | agent={agent_type} | question='{question[:60]}...'")

        # Retrieve relevant context
        docs_with_scores = self.vector_store.similarity_search(
            query=question,
            k=k,
        )

        context = self._format_context(docs_with_scores)

        # Format pipeline data
        if pipeline_data:
            pipeline_str = self._format_pipeline_data(pipeline_data)
        else:
            pipeline_str = "No pipeline data provided for this query."

        # Build and execute chain
        prompt = self._build_prompt(agent_type)

        chain = prompt | self.llm | self.output_parser

        try:
            answer = chain.invoke({
                "context": context,
                "pipeline_data": pipeline_str,
                "question": question,
            })
        except Exception as e:
            log.error(f"Gemma 4 inference failed: {e}")
            answer = f"Error generating response: {e}. Please ensure Ollama is running with: ollama serve"

        return {
            "answer": answer,
            "agent_used": agent_type,
            "sources": [
                {
                    "content_preview": doc.page_content[:200],
                    "source": doc.metadata.get("source"),
                    "category": doc.metadata.get("category"),
                    "relevance_score": round(score, 3),
                }
                for doc, score in docs_with_scores
            ],
            "pipeline_data_used": bool(pipeline_data),
            "model": OLLAMA_MODEL,
        }

    def _format_pipeline_data(self, data: Dict[str, Any]) -> str:
        """Format pipeline output dict into readable text for the prompt."""
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                lines.append(f"{key.upper()}:")
                for item in value[:10]:  # cap at 10 items to avoid token overflow
                    lines.append(f"  - {item}")
            elif isinstance(value, dict):
                lines.append(f"{key.upper()}:")
                for k, v in list(value.items())[:10]:
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key.upper()}: {value}")
        return "\n".join(lines)

    def batch_query(
        self,
        questions: List[str],
        pipeline_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple queries — used by n8n workflow orchestration
        to dispatch to all agents simultaneously.
        """
        results = []
        for question in questions:
            result = self.query(question, pipeline_data=pipeline_data)
            results.append(result)
        return results
