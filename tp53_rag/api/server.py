"""
============================================================
TP53 RAG Platform - FastAPI Server (n8n Integration Layer)
============================================================
Exposes the RAG platform as REST endpoints that n8n can call.

This is what connects the intelligence layer (RAG + Gemma 4)
to the automation layer (n8n workflows).

Endpoints:
  POST /analyse          - Full multi-agent dispatch
  POST /query            - Single free-form query
  POST /agent/{type}     - Specific agent query
  GET  /health           - Health check
  GET  /stats            - Vector store statistics

n8n calls these endpoints via HTTP Request nodes to:
  1. Trigger analysis on new accession IDs
  2. Route results to appropriate agents
  3. Aggregate and deliver platform responses
============================================================
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from knowledge_base.ingestion import TP53DocumentIngester
from knowledge_base.vector_store import TP53VectorStore
from agents.dispatcher import AgentDispatcher
from config.settings import AGENT_REGISTRY
from utils.logger import log

# ── Global platform state ─────────────────────────────────
vector_store: Optional[TP53VectorStore] = None
dispatcher: Optional[AgentDispatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise platform on startup."""
    global vector_store, dispatcher

    log.info("Initialising TP53 RAG Platform...")

    vector_store = TP53VectorStore()

    if vector_store.is_built():
        log.info("Loading existing vector store...")
        vector_store.load()
    else:
        log.info("Building vector store from scratch...")
        ingester = TP53DocumentIngester()
        documents = ingester.ingest_all()
        vector_store.build(documents)

    dispatcher = AgentDispatcher(vector_store=vector_store)
    log.info("TP53 RAG Platform ready ✓")

    yield  # App runs here

    log.info("TP53 RAG Platform shutting down")


app = FastAPI(
    title="TP53 RAG Platform API",
    description=(
        "Multi-agent RAG platform for TP53 bioinformatics analysis, "
        "powered by Gemma 4 via Ollama. Built for the Gemma 4 Good Hackathon."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────

class PipelineData(BaseModel):
    """TP53 pipeline output passed to agents."""
    accession: Optional[str] = Field(None, example="NM_000546")
    sequence_length: Optional[int] = None
    gc_content: Optional[float] = None
    mutations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    orfs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    protein_domains: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    phylo_species: Optional[List[str]] = Field(default_factory=list)
    codon_frequencies: Optional[Dict[str, float]] = None
    amino_acid_composition: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AnalyseRequest(BaseModel):
    """Request for full multi-agent platform dispatch."""
    pipeline_data: PipelineData
    include_report: bool = True
    include_sources: bool = False


class QueryRequest(BaseModel):
    """Request for a single free-form query."""
    question: str = Field(..., example="What is the clinical significance of R248W mutation?")
    pipeline_data: Optional[PipelineData] = None
    agent_type: Optional[str] = Field(
        None,
        example="clinical_interpretation",
        description="Force a specific agent. Auto-routed if not provided.",
    )


class AgentQueryRequest(BaseModel):
    """Request for a specific agent."""
    pipeline_data: PipelineData
    custom_question: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — used by n8n to verify platform is running."""
    return {
        "status": "healthy",
        "platform": "TP53 RAG Platform",
        "model": "Gemma 4 (Ollama)",
        "vector_store": vector_store.get_stats() if vector_store else "not_initialised",
    }


@app.get("/stats")
async def get_stats():
    """Vector store and platform statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Platform not initialised")
    return {
        "vector_store": vector_store.get_stats(),
        "available_agents": list(AGENT_REGISTRY.keys()),
        "agent_descriptions": {
            name: config["description"]
            for name, config in AGENT_REGISTRY.items()
        },
    }


@app.post("/analyse")
async def analyse(request: AnalyseRequest):
    """
    Full platform dispatch — runs all 6 agents on pipeline data.

    This is the primary n8n integration endpoint.
    n8n triggers this after the TP53 pipeline completes,
    passing all results for AI-powered interpretation.
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Platform not initialised")

    pipeline_dict = request.pipeline_data.model_dump(exclude_none=True)

    log.info(f"Full analysis request | accession={pipeline_dict.get('accession', 'unknown')}")

    results = dispatcher.dispatch_all(
        pipeline_data=pipeline_dict,
        include_report=request.include_report,
    )

    formatted = dispatcher.format_platform_output(
        results,
        include_sources=request.include_sources,
    )

    return {
        "status": "success",
        "accession": pipeline_dict.get("accession"),
        "agents_run": list(results.keys()),
        "results": {
            name: {
                "answer": result.answer,
                "success": result.success,
                "sources": result.sources if request.include_sources else [],
                "error": result.error,
            }
            for name, result in results.items()
        },
        "formatted_report": formatted,
    }


@app.post("/query")
async def query(request: QueryRequest):
    """
    Single free-form query — conversational interface.
    Auto-routes to the most appropriate agent.
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Platform not initialised")

    pipeline_dict = request.pipeline_data.model_dump(exclude_none=True) if request.pipeline_data else None

    result = dispatcher.interactive_query(
        question=request.question,
        pipeline_data=pipeline_dict,
    )

    return {
        "status": "success",
        "question": request.question,
        "agent_used": result.agent,
        "answer": result.answer,
        "sources": result.sources,
        "success": result.success,
    }


@app.post("/agent/{agent_type}")
async def run_agent(agent_type: str, request: AgentQueryRequest):
    """
    Run a specific named agent.

    Available agents:
    - mutation_analysis
    - orf_analysis
    - phylogenetic_analysis
    - domain_annotation
    - clinical_interpretation
    - report_generation
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Platform not initialised")

    if agent_type not in AGENT_REGISTRY and agent_type != "report_generation":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{agent_type}'. "
                   f"Available: {list(AGENT_REGISTRY.keys())}",
        )

    pipeline_dict = request.pipeline_data.model_dump(exclude_none=True)

    result = dispatcher.dispatch_single(
        agent_type=agent_type,
        pipeline_data=pipeline_dict,
        custom_question=request.custom_question,
    )

    return {
        "status": "success" if result.success else "error",
        "agent": result.agent,
        "question": result.question,
        "answer": result.answer,
        "sources": result.sources,
        "error": result.error,
    }


@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base(
    background_tasks: BackgroundTasks,
    include_ncbi: bool = True,
    include_uniprot: bool = True,
):
    """
    Rebuild the vector store from scratch.
    Runs in background — returns immediately.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Platform not initialised")

    def _rebuild():
        ingester = TP53DocumentIngester()
        documents = ingester.ingest_all(
            include_ncbi=include_ncbi,
            include_uniprot=include_uniprot,
        )
        vector_store.build(documents, force_rebuild=True)
        log.info("Knowledge base rebuilt successfully")

    background_tasks.add_task(_rebuild)
    return {"status": "rebuilding", "message": "Knowledge base rebuild started in background"}


# ── Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT

    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
