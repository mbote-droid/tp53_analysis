"""
============================================================
TP53 RAG Platform - Main Entry Point
============================================================
CLI for building the knowledge base, running queries,
and starting the API server for n8n integration.

Usage:
  python main.py build              # Build the knowledge base
  python main.py query              # Interactive query mode
  python main.py demo               # Run a demo analysis
  python main.py serve              # Start the FastAPI server
  python main.py test               # Test with sample pipeline data
============================================================
"""

import sys
import json
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint

console = Console()


def build_knowledge_base(args):
    """Build and persist the TP53 RAG knowledge base."""
    console.print(Panel(
        "[bold green]TP53 RAG Knowledge Base Builder[/bold green]\n"
        "Ingesting TP53 domain knowledge → ChromaDB",
        border_style="green",
    ))

    from knowledge_base.ingestion import TP53DocumentIngester
    from knowledge_base.vector_store import TP53VectorStore

    ingester = TP53DocumentIngester()
    documents = ingester.ingest_all(
        include_ncbi=not args.offline,
        include_uniprot=not args.offline,
        include_user_docs=True,
    )

    store = TP53VectorStore()
    store.build(documents, force_rebuild=args.force)

    stats = store.get_stats()
    console.print(Panel(
        f"[bold green]✓ Knowledge Base Ready[/bold green]\n\n"
        f"Total embeddings: {stats['total_embeddings']}\n"
        f"Embedding model: {stats['embedding_model']}\n"
        f"Location: {stats['persist_directory']}",
        border_style="green",
    ))


def interactive_query(args):
    """Interactive RAG query mode."""
    from knowledge_base.vector_store import TP53VectorStore
    from agents.dispatcher import AgentDispatcher

    console.print(Panel(
        "[bold cyan]TP53 RAG Platform — Interactive Query Mode[/bold cyan]\n"
        "Powered by Gemma 4 via Ollama\n"
        "Type 'exit' to quit",
        border_style="cyan",
    ))

    store = TP53VectorStore()
    if not store.is_built():
        console.print("[red]Knowledge base not built. Run: python main.py build[/red]")
        sys.exit(1)

    store.load()
    dispatch = AgentDispatcher(vector_store=store)

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue

            console.print("\n[yellow]Thinking...[/yellow]")
            result = dispatch.interactive_query(question=question)

            console.print(f"\n[bold green]Agent: {result.agent}[/bold green]")
            console.print(Panel(result.answer, border_style="green"))

            if result.sources:
                console.print(f"\n[dim]Sources: {len(result.sources)} documents retrieved[/dim]")

        except KeyboardInterrupt:
            break

    console.print("\n[dim]Goodbye.[/dim]")


def run_demo(args):
    """Run a demo analysis with sample TP53 pipeline data."""
    from knowledge_base.vector_store import TP53VectorStore
    from agents.dispatcher import AgentDispatcher

    console.print(Panel(
        "[bold magenta]TP53 RAG Platform — Demo Analysis[/bold magenta]\n"
        "Simulating output from the TP53 bioinformatics pipeline\n"
        "Running all 6 agents powered by Gemma 4",
        border_style="magenta",
    ))

    # Sample pipeline output (mirrors actual TP53 pipeline output)
    DEMO_PIPELINE_DATA = {
        "accession": "NM_000546",
        "species": "Homo sapiens",
        "sequence_length": 2591,
        "gc_content": 52.8,
        "mutations": [
            {"position": 524, "original": "G", "mutant": "A",
             "codon_change": "CGT→CAT", "amino_acid_change": "R175H"},
            {"position": 742, "original": "C", "mutant": "T",
             "codon_change": "CGG→TGG", "amino_acid_change": "R248W"},
            {"position": 818, "original": "G", "mutant": "A",
             "codon_change": "CGT→CAT", "amino_acid_change": "R273H"},
        ],
        "orfs": [
            {"frame": "+1", "start": 203, "end": 1384, "length": 1181,
             "description": "Main ORF - full length p53α (393 aa)"},
            {"frame": "+2", "start": 521, "end": 788, "length": 267,
             "description": "Alternative ORF - possible isoform fragment"},
        ],
        "protein_domains": [
            {"database": "Pfam", "name": "P53", "accession": "PF00870",
             "start": 94, "end": 292, "score": 245.6,
             "description": "p53 DNA-binding domain"},
            {"database": "Pfam", "name": "P53_tetramer", "accession": "PF07710",
             "start": 323, "end": 356, "score": 89.2,
             "description": "p53 tetramerization domain"},
        ],
        "phylo_species": [
            "Homo sapiens (NM_000546)",
            "Pan troglodytes (NM_001123020)",
            "Mus musculus (NM_011640)",
            "Rattus norvegicus (NM_009895)",
            "Danio rerio (NM_001271820)",
        ],
        "codon_frequencies": {
            "CGT": 0.18, "CGC": 0.22, "CGA": 0.08,
            "CGG": 0.24, "AGA": 0.15, "AGG": 0.13,
        },
        "amino_acid_composition": {
            "R": 0.076, "G": 0.044, "P": 0.055,
            "S": 0.071, "T": 0.056, "C": 0.031,
        },
    }

    store = TP53VectorStore()
    if not store.is_built():
        console.print("[red]Knowledge base not built. Run: python main.py build[/red]")
        sys.exit(1)

    store.load()
    dispatch = AgentDispatcher(vector_store=store)

    console.print("\n[yellow]Running multi-agent analysis...[/yellow]\n")

    results = dispatch.dispatch_all(
        pipeline_data=DEMO_PIPELINE_DATA,
        include_report=True,
    )

    output = dispatch.format_platform_output(results, include_sources=False)
    console.print(output)

    # Save demo output
    output_path = Path("demo_output.txt")
    output_path.write_text(output, encoding="utf-8")
    console.print(f"\n[dim]Output saved to: {output_path}[/dim]")


def start_server(args):
    """Start the FastAPI server for n8n integration."""
    console.print(Panel(
        "[bold blue]TP53 RAG Platform API Server[/bold blue]\n"
        "Starting FastAPI server for n8n integration\n"
        f"URL: http://localhost:8000\n"
        f"Docs: http://localhost:8000/docs",
        border_style="blue",
    ))

    import uvicorn
    from config.settings import API_HOST, API_PORT

    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=args.dev,
        log_level="info",
    )


def run_visualise(args):
    """Run the 3D structure visualisation agent."""
    from knowledge_base.vector_store import TP53VectorStore
    from agents.rag_chain import TP53RAGChain
    from agents.structure_viz.agent import StructureVisualisationAgent
    from agents.structure_viz.web_app import launch as launch_webapp
    import subprocess, sys

    console.print(Panel(
        "[bold magenta]TP53 Structure Visualisation Agent[/bold magenta]\n"
        "ESMFold + ESM-2 + Mol* + Plotly · 100% local\n"
        f"Accession: {args.accession}",
        border_style="magenta",
    ))

    # Load RAG chain for Gemma 4 narration (optional)
    rag_chain = None
    store = TP53VectorStore()
    if store.is_built():
        store.load()
        rag_chain = TP53RAGChain(vector_store=store)
        console.print("[green]RAG chain loaded — Gemma 4 narration enabled[/green]")
    else:
        console.print("[yellow]Knowledge base not built — structural analysis only (no LLM narration)[/yellow]")

    pipeline_data = {
        "accession": args.accession,
        "mutations": [
            {"position": 175, "original": "C", "mutant": "A", "amino_acid_change": "R175H"},
            {"position": 248, "original": "C", "mutant": "T", "amino_acid_change": "R248W"},
            {"position": 273, "original": "G", "mutant": "A", "amino_acid_change": "R273H"},
        ],
    }

    agent = StructureVisualisationAgent(rag_chain=rag_chain)
    result = agent.run(
        pipeline_data=pipeline_data,
        use_esmfold=not args.no_esmfold,
        generate_embeddings=not args.no_embeddings,
    )

    console.print(f"\n[green]Structure analysis complete in {result['prediction_time_seconds']}s[/green]")

    if getattr(args, 'notebook', False):
        console.print("[cyan]Opening Jupyter notebook...[/cyan]")
        nb_path = Path("agents/structure_viz/visualise.ipynb")
        subprocess.Popen([sys.executable, "-m", "jupyter", "notebook", str(nb_path)])
    else:
        console.print("[cyan]Launching web app at http://localhost:5001[/cyan]")
        launch_webapp(result, accession=args.accession)


def main():
    parser = argparse.ArgumentParser(
        description="TP53 RAG Platform — Gemma 4 Hackathon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build
    build_parser = subparsers.add_parser("build", help="Build the RAG knowledge base")
    build_parser.add_argument("--offline", action="store_true",
                               help="Use only curated knowledge (no internet needed)")
    build_parser.add_argument("--force", action="store_true",
                               help="Force rebuild even if already exists")

    # query
    subparsers.add_parser("query", help="Interactive query mode")

    # demo
    subparsers.add_parser("demo", help="Run demo with sample pipeline data")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--dev", action="store_true", help="Enable hot reload")

    # visualise
    vis_parser = subparsers.add_parser("visualise", help="Run 3D structure visualisation agent")
    vis_parser.add_argument("--accession", default="NM_000546", help="Accession ID")
    vis_parser.add_argument("--no-esmfold", action="store_true", help="Skip ESMFold, use PDB fallback")
    vis_parser.add_argument("--no-embeddings", action="store_true", help="Skip ESM-2 embeddings")
    vis_parser.add_argument("--notebook", action="store_true", help="Open Jupyter notebook viewer")

    args = parser.parse_args()

    if args.command == "build":
        build_knowledge_base(args)
    elif args.command == "query":
        interactive_query(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "serve":
        start_server(args)
    elif args.command == "visualise":
        run_visualise(args)
    else:
        # Default: show help with platform overview
        console.print(Panel(
            "[bold]TP53 RAG Platform[/bold]\n"
            "Multi-agent bioinformatics AI powered by Gemma 4 via Ollama\n\n"
            "[cyan]Commands:[/cyan]\n"
            "  python main.py build    — Build RAG knowledge base\n"
            "  python main.py query    — Interactive query mode\n"
            "  python main.py demo     — Demo with sample data\n"
            "  python main.py serve    — Start API server (for n8n)\n\n"
            "[yellow]Quick start:[/yellow]\n"
            "  1. ollama pull gemma4\n"
            "  2. ollama pull nomic-embed-text\n"
            "  3. cp .env.example .env  (edit with your ENTREZ_EMAIL)\n"
            "  4. python main.py build\n"
            "  5. python main.py demo",
            border_style="bold white",
            title="[bold white]TP53 RAG Platform — Gemma 4 Hackathon[/bold white]",
        ))
        parser.print_help()


if __name__ == "__main__":
    main()
