"""
============================================================
TP53 RAG Platform - Test Suite
============================================================
Tests for the ingestion pipeline, vector store, RAG chain,
and multi-agent dispatcher.

Run: pytest tests/ -v
     pytest tests/ --cov=. --cov-report=html
============================================================
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from knowledge_base.ingestion import TP53DocumentIngester, CURATED_TP53_KNOWLEDGE
from agents.rag_chain import IntentRouter


class TestCuratedKnowledge:
    """Validate the curated TP53 knowledge base content."""

    def test_curated_knowledge_not_empty(self):
        assert len(CURATED_TP53_KNOWLEDGE) > 0

    def test_curated_knowledge_has_required_fields(self):
        for item in CURATED_TP53_KNOWLEDGE:
            assert "content" in item
            assert "metadata" in item
            assert len(item["content"].strip()) > 50

    def test_curated_covers_critical_categories(self):
        categories = {item["metadata"]["category"] for item in CURATED_TP53_KNOWLEDGE}
        required = {"mutations", "clinical", "protein_domains", "gene_overview"}
        assert required.issubset(categories)

    def test_hotspot_mutations_present(self):
        all_content = " ".join(item["content"] for item in CURATED_TP53_KNOWLEDGE)
        hotspots = ["R175H", "R248W", "R248Q", "R273H", "R273C", "G245S"]
        for hotspot in hotspots:
            assert hotspot in all_content, f"Hotspot {hotspot} missing from knowledge base"

    def test_clinical_syndromes_covered(self):
        all_content = " ".join(item["content"] for item in CURATED_TP53_KNOWLEDGE)
        assert "Li-Fraumeni" in all_content
        assert "MDM2" in all_content


class TestIngester:
    """Test the document ingestion pipeline."""

    def test_load_curated_returns_documents(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_curated_offline_flag(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        for doc in docs:
            assert doc.metadata.get("offline_available") is True

    def test_chunk_documents(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)
        # Chunks should be >= docs (each doc may split into multiple chunks)
        assert len(chunks) >= len(docs)

    def test_chunk_size_respected(self):
        from config.settings import CHUNK_SIZE
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)
        oversized = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.2]
        assert len(oversized) == 0, f"{len(oversized)} chunks exceed CHUNK_SIZE"

    def test_user_documents_empty_dir(self, tmp_path):
        ingester = TP53DocumentIngester()
        docs = ingester.load_user_documents(directory=tmp_path)
        assert docs == []


class TestIntentRouter:
    """Test the query routing logic."""

    def setup_method(self):
        self.router = IntentRouter()

    def test_mutation_keywords(self):
        queries = [
            "What is the impact of this mutation?",
            "Analyse the detected variants",
            "Tell me about the SNV at position 524",
        ]
        for q in queries:
            assert self.router.route(q) == "mutation_analysis", f"Failed for: {q}"

    def test_clinical_keywords(self):
        queries = [
            "What is the clinical significance?",
            "Is this pathogenic or benign?",
            "What cancer types are associated?",
        ]
        for q in queries:
            assert self.router.route(q) == "clinical_interpretation", f"Failed for: {q}"

    def test_phylogenetic_keywords(self):
        queries = [
            "Interpret the phylogenetic tree",
            "How conserved is this across species?",
            "What does the evolutionary analysis show?",
        ]
        for q in queries:
            assert self.router.route(q) == "phylogenetic_analysis", f"Failed for: {q}"

    def test_orf_keywords(self):
        result = self.router.route("Interpret the open reading frames found")
        assert result == "orf_analysis"

    def test_domain_keywords(self):
        result = self.router.route("What protein domains were annotated?")
        assert result == "domain_annotation"

    def test_no_match_returns_default(self):
        result = self.router.route("Hello there")
        assert result == "default"


class TestVectorStoreIntegration:
    """Integration tests for the vector store (require Ollama)."""

    @pytest.mark.integration
    def test_build_and_query(self, tmp_path):
        """Full build + query cycle. Requires Ollama running."""
        from knowledge_base.ingestion import TP53DocumentIngester
        from knowledge_base.vector_store import TP53VectorStore
        from config import settings
        settings.CHROMA_DIR = tmp_path / "chroma_test"

        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)

        store = TP53VectorStore()
        store.build(chunks[:20])  # Use subset for speed

        results = store.similarity_search("R175H mutation cancer", k=3)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, _ in results)


class TestPipelineDataFormatting:
    """Test that pipeline data is correctly formatted for agents."""

    def test_format_mutations(self):
        from agents.rag_chain import TP53RAGChain
        chain = TP53RAGChain.__new__(TP53RAGChain)  # Skip __init__
        chain.llm = None
        chain.router = IntentRouter()
        chain.output_parser = None

        data = {
            "mutations": [
                {"position": 524, "amino_acid_change": "R175H"},
                {"position": 742, "amino_acid_change": "R248W"},
            ]
        }
        formatted = chain._format_pipeline_data(data)
        assert "R175H" in formatted
        assert "R248W" in formatted

    def test_format_empty_data(self):
        from agents.rag_chain import TP53RAGChain
        chain = TP53RAGChain.__new__(TP53RAGChain)
        chain.llm = None
        chain.router = IntentRouter()
        chain.output_parser = None

        formatted = chain._format_pipeline_data({})
        assert formatted == ""
