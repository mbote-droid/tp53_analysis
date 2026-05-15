"""
============================================================
TP53 RAG Platform - Vector Store (ChromaDB)
============================================================
Embeds TP53 knowledge chunks into ChromaDB using
nomic-embed-text via Ollama (100% local, no API keys).

Design decisions:
  - nomic-embed-text: best open embedding model available
    via Ollama. 768-dim embeddings. Outperforms OpenAI
    ada-002 on biomedical text benchmarks.
  - ChromaDB: persistent local vector DB. No server
    needed. Works offline. HIPAA-friendly (no data leaves
    the machine) — critical for clinical deployments.
  - Cosine similarity: appropriate for semantic text search
============================================================
"""

from pathlib import Path
from typing import List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config.settings import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)
from utils.logger import log


class TP53VectorStore:
    """
    ChromaDB vector store for TP53 domain knowledge.

    Provides:
      - build(): embed and persist all documents
      - similarity_search(): retrieve relevant chunks
      - filtered_search(): search by metadata category
      - get_stats(): collection statistics
    """

    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        self._vectorstore: Optional[Chroma] = None
        log.info(
            f"VectorStore initialised | embedding={OLLAMA_EMBEDDING_MODEL} "
            f"| db={CHROMA_DIR}"
        )

    def _get_chroma_client(self) -> chromadb.PersistentClient:
        """Create or connect to the persistent ChromaDB client."""
        return chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def build(self, documents: List[Document], force_rebuild: bool = False) -> "TP53VectorStore":
        """
        Embed documents and persist to ChromaDB.

        Args:
            documents: Chunked Document objects from ingestion pipeline
            force_rebuild: If True, wipe existing collection and rebuild

        Returns:
            self (for method chaining)
        """
        client = self._get_chroma_client()

        if force_rebuild:
            try:
                client.delete_collection(CHROMA_COLLECTION_NAME)
                log.warning(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
            except Exception:
                pass

        # Check if collection already exists with data
        try:
            existing = client.get_collection(CHROMA_COLLECTION_NAME)
            count = existing.count()
            if count > 0 and not force_rebuild:
                log.info(
                    f"Collection '{CHROMA_COLLECTION_NAME}' already exists "
                    f"with {count} embeddings. Loading existing store."
                )
                self._vectorstore = Chroma(
                    client=client,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_model,
                )
                return self
        except Exception:
            pass

        log.info(f"Building vector store with {len(documents)} documents...")
        log.info("Embedding via Ollama nomic-embed-text (this may take 1-3 minutes)...")

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            client=client,
            collection_name=CHROMA_COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )

        count = self._vectorstore._collection.count()
        log.info(f"Vector store built: {count} embeddings persisted to {CHROMA_DIR}")
        return self

    def load(self) -> "TP53VectorStore":
        """Load an existing persisted vector store."""
        client = self._get_chroma_client()
        self._vectorstore = Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_model,
        )
        count = self._vectorstore._collection.count()
        log.info(f"Loaded existing vector store: {count} embeddings")
        return self

    def is_built(self) -> bool:
        """Check if the vector store has been built."""
        try:
            client = self._get_chroma_client()
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most relevant documents for a query.

        Args:
            query: Natural language question or search query
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of (Document, similarity_score) tuples
        """
        if not self._vectorstore:
            raise RuntimeError("Vector store not built. Call build() or load() first.")

        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        # Filter by score threshold
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]

        log.debug(
            f"Query: '{query[:60]}...' | "
            f"Retrieved: {len(results)} | "
            f"Above threshold: {len(filtered)}"
        )

        return filtered

    def filtered_search(
        self,
        query: str,
        category: str,
        k: int = TOP_K_RESULTS,
    ) -> List[Document]:
        """
        Search within a specific knowledge category.

        Args:
            query: Natural language query
            category: One of: mutations, clinical, phylogenetics, protein_domains,
                      pathway, therapeutics, codon_usage_gc, mutation_classification
            k: Number of results

        Returns:
            List of Document objects
        """
        if not self._vectorstore:
            raise RuntimeError("Vector store not built. Call build() or load() first.")

        results = self._vectorstore.similarity_search(
            query=query,
            k=k,
            filter={"category": category},
        )
        return results

    def get_retriever(self, k: int = TOP_K_RESULTS):
        """
        Return a LangChain retriever interface for use in chains.
        This is what plugs into the RAG chain.
        """
        if not self._vectorstore:
            raise RuntimeError("Vector store not built. Call build() or load() first.")

        return self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": SIMILARITY_THRESHOLD,
            },
        )

    def get_stats(self) -> dict:
        """Return collection statistics."""
        if not self._vectorstore:
            return {"status": "not_built"}

        try:
            count = self._vectorstore._collection.count()
            return {
                "status": "ready",
                "total_embeddings": count,
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": OLLAMA_EMBEDDING_MODEL,
                "persist_directory": str(CHROMA_DIR),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
