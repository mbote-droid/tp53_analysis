"""
============================================================
TP53 RAG Platform - Vector Store (ChromaDB Optimized)
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
    """Persistent local vector DB wrapper with cached connection parameters."""

    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        self._vectorstore: Optional[Chroma] = None

        # Fixed: Establish a single persistent connection to stop thread locks
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        log.info(f"VectorStore initialised | db connection singleton cached.")

    def build(self, documents: List[Document], force_rebuild: bool = False) -> "TP53VectorStore":
        if force_rebuild:
            try:
                self._client.delete_collection(CHROMA_COLLECTION_NAME)
                log.warning(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
            except Exception:
                pass
        try:
            existing = self._client.get_collection(CHROMA_COLLECTION_NAME)
            count = existing.count()
            if count > 0 and not force_rebuild:
                log.info(f"Collection '{CHROMA_COLLECTION_NAME}' exists. Loading existing store.")
                self._vectorstore = Chroma(
                    client=self._client,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_model,
                )
                return self
        except Exception:
            pass
        log.info(f"Building vector store with {len(documents)} documents via Ollama...")
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            client=self._client,
            collection_name=CHROMA_COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return self

    def load(self) -> "TP53VectorStore":
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_model,
        )
        return self

    def is_built(self) -> bool:
        try:
            collection = self._client.get_collection(CHROMA_COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Tuple[Document, float]]:
        if not self._vectorstore:
            raise RuntimeError("Vector store not loaded. Call load() first.")
        results = self._vectorstore.similarity_search_with_relevance_scores(query=query, k=k)
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        return filtered

    def filtered_search(self, query: str, category: str, k: int = TOP_K_RESULTS) -> List[Document]:
        if not self._vectorstore:
            raise RuntimeError("Vector store not loaded. Call load() first.")
        return self._vectorstore.similarity_search(query=query, k=k, filter={"category": category})

    def get_retriever(self, k: int = TOP_K_RESULTS):
        if not self._vectorstore:
            raise RuntimeError("Vector store not loaded. Call load() first.")
        return self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": SIMILARITY_THRESHOLD},
        )

    def get_stats(self) -> dict:
        if not self._vectorstore:
            return {"status": "not_built"}
        try:
            count = self._vectorstore._collection.count()
            return {
                "status": "ready",
                "total_embeddings": count,
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": OLLAMA_EMBEDDING_MODEL,
                # BUG FIX: Key had a stray newline inside the string:
                # "persist_directory\n" → "persist_directory"
                "persist_directory": str(CHROMA_DIR),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
