"""
============================================================
Precision Onco Africa - Main Entry Point & CLI
============================================================
Architecture:
  KnowledgeBaseCommands  — build
  QueryCommands          — query, demo, interactive
  ServerCommands         — serve, visualise
  TestCommands           — test-rag, test-variant, test-immuno, test-dossier
  PlatformCLI            — orchestrator, arg parsing, entry point

Usage:
  python main.py build
  python main.py query
  python main.py demo
  python main.py serve
  python main.py visualise
  python main.py test-rag
  python main.py test-variant
  python main.py test-immuno
  python main.py test-dossier
  python main.py list-agents
============================================================
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Agent registry ────────────────────────────────────────────────
AGENT_REGISTRY: Dict[str, Dict] = {
    "mutation_analysis":      {"class": "RAGChain",               "description": "TP53 hotspot variant classification",          "keywords": ["mutation", "variant", "hotspot", "r175h", "r248w"]},
    "variant_curator":        {"class": "VariantCurator",         "description": "Pathogenicity scoring (IARC/ClinVar/COSMIC)",   "keywords": ["pathogenic", "clinvar", "cosmic", "vus"]},
    "drug_discovery":         {"class": "RAGChain",               "description": "Therapeutic targeting, APR-246, KEML",         "keywords": ["drug", "therapy", "apr-246", "mdm2", "keml"]},
    "immunogenicity":         {"class": "ImmunogenicityPredictor","description": "TME profiling, checkpoint response",            "keywords": ["immune", "tme", "checkpoint", "pd-l1"]},
    "clinical_interpretation":{"class": "RAGChain",               "description": "Clinical significance, prognosis",             "keywords": ["clinical", "prognosis", "cancer", "li-fraumeni"]},
    "liquid_biopsy":          {"class": "RAGChain",               "description": "ctDNA VAF trend, resistance detection",        "keywords": ["liquid biopsy", "ctdna", "vaf", "resistance"]},
    "gene_expression":        {"class": "RAGChain",               "description": "Pathway analysis, TME modelling",              "keywords": ["expression", "pathway", "rna", "tme"]},
    "enzyme_design":          {"class": "RAGChain",               "description": "p53 reactivation, PROTAC, zinc rescue",        "keywords": ["enzyme", "reactivation", "protac", "zinc"]},
    "dossier_compiler":       {"class": "DossierCompiler",        "description": "Academic/pharma report, FHIR R4, PDF",         "keywords": ["report", "dossier", "fhir", "pdf"]},
}

# ── Shared demo data ──────────────────────────────────────────────
DEMO_PIPELINE_DATA: Dict[str, Any] = {
    "accession": "NM_000546", "species": "Homo sapiens",
    "sequence_length": 2591,  "gc_content": 52.8,
    "mutations": [
        {"position": 524, "original": "G", "mutant": "A", "codon_change": "CGT→CAT", "amino_acid_change": "R175H", "classification": "hotspot", "mutation_class": "conformational"},
        {"position": 742, "original": "C", "mutant": "T", "codon_change": "CGG→TGG", "amino_acid_change": "R248W", "classification": "hotspot", "mutation_class": "contact"},
        {"position": 818, "original": "G", "mutant": "A", "codon_change": "CGT→CAT", "amino_acid_change": "R273H", "classification": "hotspot", "mutation_class": "contact"},
    ],
    "orfs": [
        {"frame": "+1", "start": 203, "end": 1384, "length": 1181, "description": "Main ORF — full length p53α (393 aa)"},
        {"frame": "+2", "start": 521, "end": 788,  "length": 267,  "description": "Alternative ORF — possible isoform fragment"},
    ],
    "protein_domains": [
        {"database": "Pfam", "name": "P53",         "accession": "PF00870", "start": 94,  "end": 292, "score": 245.6, "description": "p53 DNA-binding domain"},
        {"database": "Pfam", "name": "P53_tetramer","accession": "PF07710", "start": 323, "end": 356, "score": 89.2,  "description": "p53 tetramerization domain"},
    ],
    "phylo_species":    ["Homo sapiens (NM_000546)", "Pan troglodytes (NM_001123020)", "Mus musculus (NM_011640)", "Rattus norvegicus (NM_009895)", "Danio rerio (NM_001271820)"],
    "vaf_data":         {"R175H": 22.3, "R248W": 21.4, "R273H": 11.3},
    "keml_drugs":       ["Carboplatin", "Doxorubicin"],
    "apr246_eligible":  ["R175H", "R248W"],
}


# ══════════════════════════════════════════════════════════════════
# 1. Knowledge Base Commands
# ══════════════════════════════════════════════════════════════════

class KnowledgeBaseCommands:
    """Handles all knowledge base build operations. Isolated — safe to fail."""

    @staticmethod
    def build(args):
        console.print(Panel(
            "[bold green]TP53 RAG Knowledge Base Builder[/bold green]\n"
            "Ingesting TP53 domain knowledge → ChromaDB",
            border_style="green",
        ))
        try:
            from knowledge_base.ingestion import TP53DocumentIngester
            from knowledge_base.vector_store import TP53VectorStore
            ingester  = TP53DocumentIngester()
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
                f"Total embeddings : {stats['total_embeddings']}\n"
                f"Embedding model  : {stats['embedding_model']}\n"
                f"Location         : {stats['persist_directory']}",
                border_style="green",
            ))
        except Exception as e:
            console.print(f"[red]✗ Build failed: {e}[/red]")
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 2. Query & Demo Commands
# ══════════════════════════════════════════════════════════════════

class QueryCommands:
    """Handles interactive queries and demo runs. Each method fails independently."""

    @staticmethod
    def _load_dispatcher():
        from knowledge_base.vector_store import TP53VectorStore
        from agents.dispatcher import AgentDispatcher
        store = TP53VectorStore()
        if not store.is_built():
            console.print("[red]Knowledge base not built. Run: python main.py build[/red]")
            sys.exit(1)
        store.load()
        return AgentDispatcher(vector_store=store)

    @staticmethod
    def interactive(args):
        console.print(Panel(
            "[bold cyan]Precision Onco Africa — Interactive Query Mode[/bold cyan]\n"
            "Powered by Gemma 4 via llama.cpp  |  Type 'exit' to quit",
            border_style="cyan",
        ))
        dispatch = QueryCommands._load_dispatcher()
        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                if question.lower() in ("exit", "quit", "q"):
                    break
                if question.lower() == "agents":
                    PlatformCLI.list_agents(None)
                    continue
                if not question:
                    continue
                console.print("\n[yellow]Thinking...[/yellow]")
                result = dispatch.interactive_query(question=question)
                console.print(f"\n[bold green]Agent: {result.agent}[/bold green]")
                console.print(Panel(result.answer, border_style="green"))
                if result.sources:
                    console.print(f"[dim]Sources: {len(result.sources)} docs[/dim]")
            except KeyboardInterrupt:
                break
        console.print("\n[dim]Goodbye.[/dim]")

    @staticmethod
    def demo(args):
        console.print(Panel(
            "[bold magenta]Precision Onco Africa — Full Demo[/bold magenta]\n"
            "9 specialised agents · Gemma 4 · llama.cpp · HIPAA · FHIR R4",
            border_style="magenta",
        ))
        dispatch = QueryCommands._load_dispatcher()
        console.print("\n[yellow]Dispatching to all agents...[/yellow]\n")
        results = dispatch.dispatch_all(pipeline_data=DEMO_PIPELINE_DATA, include_report=True)
        output  = dispatch.format_platform_output(results, include_sources=False)
        console.print(output)
        QueryCommands._run_new_agents()
        output_path = Path("demo_output.txt")
        output_path.write_text(output, encoding="utf-8")
        console.print(f"\n[dim]Saved to: {output_path}[/dim]")

    @staticmethod
    def _run_new_agents():
        """Run the 3 specialised agents — each wrapped so one failure doesn't stop others."""
        for label, module, cls_name, runner in [
            ("Variant Curator",          "agents.variant_curator",  "VariantCurator",          QueryCommands._run_variant),
            ("Immunogenicity Predictor", "agents.immunogenicity",   "ImmunogenicityPredictor", QueryCommands._run_immuno),
            ("Dossier Compiler",         "agents.dossier_compiler", "DossierCompiler",         QueryCommands._run_dossier),
        ]:
            console.print(f"\n[bold blue]─── {label} ───[/bold blue]")
            try:
                runner()
            except Exception as e:
                console.print(f"[yellow]  Skipped: {e}[/yellow]")

    @staticmethod
    def _run_variant():
        from agents.variant_curator import VariantCurator
        vc = VariantCurator()
        for mut in ["R175H", "R248W", "R273H"]:
            r = vc.classify(mut)
            console.print(f"  {mut}: {r.get('pathogenicity','—')} | ClinVar={r.get('clinvar_class','—')}")

    @staticmethod
    def _run_immuno():
        from agents.immunogenicity import ImmunogenicityPredictor
        predictor = ImmunogenicityPredictor()
        for mut in ["R175H", "R248W", "R273H"]:
            r = predictor.predict(mut)
            console.print(f"  {mut}: TME={r.get('tme_status','—')} | Checkpoint={r.get('checkpoint_recommendation','—')}")

    @staticmethod
    def _run_dossier():
        from agents.dossier_compiler import DossierCompiler
        # compile(agent_results, mutation, cancer_type, mode)
        agent_results = {
            "mutation_analysis": "R175H conformational mutant, high oncogenicity.",
            "clinical_interpretation": "Pathogenic — breast, colorectal, AML associations.",
            "drug_discovery": "APR-246 HIGH priority. Carboplatin KEML-available.",
        }
        for mode in ["academic", "pharma"]:
            r = DossierCompiler().compile(
                agent_results=agent_results,
                mutation="R175H",
                cancer_type="Breast Cancer",
                mode=mode,
            )
            console.print(f"  [{mode}] {str(r.get('summary','—'))[:100]}...")


# ══════════════════════════════════════════════════════════════════
# 3. Server Commands
# ══════════════════════════════════════════════════════════════════

class ServerCommands:
    """Handles API server and 3D visualisation. Isolated from query logic."""

    @staticmethod
    def serve(args):
        console.print(Panel(
            "[bold blue]Precision Onco Africa API Server[/bold blue]\n"
            "URL : http://localhost:8000\n"
            "Docs: http://localhost:8000/docs",
            border_style="blue",
        ))
        try:
            import uvicorn
            from config.settings import API_HOST, API_PORT
            uvicorn.run("api.server:app", host=API_HOST, port=API_PORT,
                        reload=args.dev, log_level="info")
        except Exception as e:
            console.print(f"[red]✗ Server failed: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def visualise(args):
        import subprocess
        console.print(Panel(
            "[bold magenta]TP53 Structure Visualisation[/bold magenta]\n"
            f"ESMFold + Mol* + Plotly · Accession: {args.accession}",
            border_style="magenta",
        ))
        try:
            from knowledge_base.vector_store import TP53VectorStore
            from agents.rag_chain import TP53RAGChain
            from agents.structure_viz.agent import StructureVisualisationAgent
            from agents.structure_viz.web_app import launch as launch_webapp

            rag_chain = None
            store = TP53VectorStore()
            if store.is_built():
                store.load()
                rag_chain = TP53RAGChain(vector_store=store)
                console.print("[green]RAG narration enabled[/green]")

            agent  = StructureVisualisationAgent(rag_chain=rag_chain)
            result = agent.run(
                pipeline_data={"accession": args.accession, "mutations": [
                    {"position": 175, "amino_acid_change": "R175H"},
                    {"position": 248, "amino_acid_change": "R248W"},
                    {"position": 273, "amino_acid_change": "R273H"},
                ]},
                use_esmfold=not args.no_esmfold,
                generate_embeddings=not args.no_embeddings,
            )
            console.print(f"[green]Done in {result['prediction_time_seconds']}s[/green]")
            if getattr(args, "notebook", False):
                subprocess.Popen([sys.executable, "-m", "jupyter", "notebook",
                                  "agents/structure_viz/visualise.ipynb"])
            else:
                launch_webapp(result, accession=args.accession)
        except Exception as e:
            console.print(f"[red]✗ Visualise failed: {e}[/red]")
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 4. Test Commands
# ══════════════════════════════════════════════════════════════════

class TestCommands:
    """
    Independent test runners — each test is fully isolated.
    One failing test never blocks others.
    """

    @staticmethod
    def test_rag(args):
        console.print(Panel("[bold]Testing RAG Chain[/bold]", border_style="cyan"))
        try:
            from agents.rag_chain import TP53RAGChain
            from knowledge_base.vector_store import TP53VectorStore
            store = TP53VectorStore()
            if store.is_built():
                store.load()
            chain  = TP53RAGChain(vector_store=store if store.is_built() else None)
            result = chain.query(
                question="What are the therapeutic implications of R175H?",
                pipeline_data=DEMO_PIPELINE_DATA,
                agent_type="mutation_analysis",
            )
            console.print(f"[green]✓ RAG chain OK[/green]")
            console.print(f"  Agent={result['agent_used']} | Sources={len(result['sources'])} | Cache={result['cache_hit']} | Retries={result['retries']}")
            console.print(Panel(result["answer"][:300] + "...", border_style="green"))
        except Exception as e:
            console.print(f"[red]✗ RAG FAILED: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def test_variant(args):
        console.print(Panel("[bold]Testing Variant Curator[/bold]", border_style="cyan"))
        try:
            from agents.variant_curator import VariantCurator
            vc = VariantCurator()
            for mut in ["R175H", "R248W", "R273H", "G245S"]:
                r = vc.classify(mut)
                console.print(f"  ✓ {mut}: {r.get('pathogenicity','—')} | ClinVar={r.get('clinvar_class','—')}")
            console.print("[green]✓ Variant curator OK[/green]")
        except Exception as e:
            console.print(f"[red]✗ Variant curator FAILED: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def test_immuno(args):
        console.print(Panel("[bold]Testing Immunogenicity Predictor[/bold]", border_style="cyan"))
        try:
            from agents.immunogenicity import ImmunogenicityPredictor
            r = ImmunogenicityPredictor().predict("R175H")
            console.print(f"  TME          : {r.get('tme_status','—')}")
            
            console.print(f"  Checkpoint   : {r.get('checkpoint_recommendation','—')}")
            console.print(f"  Neoantigens  : {r.get('neoantigen_count','—')}")
            console.print("[green]✓ Immunogenicity OK[/green]")
        except Exception as e:
            console.print(f"[red]✗ Immunogenicity FAILED: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def test_dossier(args):
        console.print(Panel("[bold]Testing Dossier Compiler[/bold]", border_style="cyan"))
        try:
            from agents.dossier_compiler import DossierCompiler
            dc = DossierCompiler()
            for mode in ["academic", "pharma"]:
                r = dc.compile(agent_results={"mutation_analysis": "R175H conformational mutant.", "clinical_interpretation": "Pathogenic.", "drug_discovery": "APR-246 HIGH priority."}, mutation="R175H", cancer_type="Breast Cancer", mode=mode)
                console.print(f"  ✓ [{mode}] {str(r.get('summary','—'))[:80]}...")
            console.print("[green]✓ Dossier compiler OK[/green]")
        except Exception as e:
            console.print(f"[red]✗ Dossier FAILED: {e}[/red]")
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 5. Platform CLI — orchestrator only
# ══════════════════════════════════════════════════════════════════

class PlatformCLI:
    """
    Pure orchestrator — builds args, routes to command classes.
    Contains zero business logic.
    """

    COMMANDS = {
        "build":        (KnowledgeBaseCommands.build,   "Build RAG knowledge base"),
        "query":        (QueryCommands.interactive,      "Interactive Q&A mode"),
        "demo":         (QueryCommands.demo,             "Full 9-agent demo"),
        "serve":        (ServerCommands.serve,           "FastAPI server for n8n"),
        "visualise":    (ServerCommands.visualise,       "3D structure viewer"),
        "test-rag":     (TestCommands.test_rag,          "Test RAG chain"),
        "test-variant": (TestCommands.test_variant,      "Test variant curator"),
        "test-immuno":  (TestCommands.test_immuno,       "Test immunogenicity"),
        "test-dossier": (TestCommands.test_dossier,      "Test dossier compiler"),
        "list-agents":  (lambda a: PlatformCLI.list_agents(a), "Show all agents"),
    }

    @staticmethod
    def list_agents(args):
        table = Table(title="Precision Onco Africa — Registered Agents", border_style="cyan")
        table.add_column("Agent",       style="bold cyan", no_wrap=True)
        table.add_column("Class",       style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Keywords",    style="dim")
        for name, cfg in AGENT_REGISTRY.items():
            table.add_row(name, cfg["class"], cfg["description"],
                          ", ".join(cfg.get("keywords", [])[:3]))
        console.print(table)

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Precision Onco Africa — Daktari Genomed Labs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub = parser.add_subparsers(dest="command")

        bp = sub.add_parser("build");      bp.add_argument("--offline", action="store_true"); bp.add_argument("--force", action="store_true")
        sub.add_parser("query")
        sub.add_parser("demo")
        sp = sub.add_parser("serve");      sp.add_argument("--dev", action="store_true")
        vp = sub.add_parser("visualise");  vp.add_argument("--accession", default="NM_000546"); vp.add_argument("--no-esmfold", action="store_true"); vp.add_argument("--no-embeddings", action="store_true"); vp.add_argument("--notebook", action="store_true")
        sub.add_parser("test-rag")
        sub.add_parser("test-variant")
        sub.add_parser("test-immuno")
        sub.add_parser("test-dossier")
        sub.add_parser("list-agents")
        return parser

    @staticmethod
    def show_help(parser):
        console.print(Panel(
            "[bold]Precision Onco Africa[/bold]\n"
            "Multi-agent bioinformatics AI · Gemma 4 · llama.cpp · HIPAA · FHIR R4\n\n"
            "[cyan]Commands:[/cyan]\n"
            "  python main.py build          — Build RAG knowledge base\n"
            "  python main.py query          — Interactive Q&A\n"
            "  python main.py demo           — Full multi-agent demo\n"
            "  python main.py serve          — FastAPI server (n8n)\n"
            "  python main.py visualise      — 3D structure viewer\n"
            "  python main.py test-rag       — Test RAG chain\n"
            "  python main.py test-variant   — Test variant curator\n"
            "  python main.py test-immuno    — Test immunogenicity\n"
            "  python main.py test-dossier   — Test dossier compiler\n"
            "  python main.py list-agents    — Show all agents\n\n"
            "[yellow]Quick start (8GB RAM Local Edge Optimization):[/yellow]\n" 
            " 1. ollama create gemma4-lowmem -f ./Modelfile\n"
            " 2. ollama run gemma4-lowmem\n"
            " 3. cp .env.example .env (add ENTREZ_EMAIL)\n"
            " 4. python main.py build\n"
            " 5. python main.py demo\n"
            " 6. streamlit run app.py\n",

            border_style="bold white",
            title="[bold white]Precision Onco Africa — Daktari Genomed Labs[/bold white]",
        ))
        parser.print_help()

    @classmethod
    def run(cls):
        parser = cls.build_parser()
        args   = parser.parse_args()
        if args.command in cls.COMMANDS:
            cls.COMMANDS[args.command][0](args)
        else:
            cls.show_help(parser)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    PlatformCLI.run()
