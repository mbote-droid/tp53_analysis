"""
============================================================
Hybrid Search — BM25 + Vector Embeddings Fusion
utils/hybrid_search.py
============================================================
Combines lexical (BM25) and semantic (vector) search.
Weighted fusion: 0.7 vector + 0.3 BM25 for best recall.
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    """Single search result with metadata."""
    document_id: str
    content: str
    score: float
    source: str  # "vector" or "bm25"
    metadata: Dict = None


class HybridSearchEngine:
    """Fusion of BM25 lexical + vector semantic search."""
    
    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        Args:
            vector_weight: Importance of vector search (0-1)
            bm25_weight: Importance of BM25 (0-1)
        """
        if abs(vector_weight + bm25_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to ~1.0")
        
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_index = None
        self.documents = {}
        self.doc_ids = []
    
    def build_bm25_index(self, documents: List[Dict]):
        """
        Build BM25 index from document list.
        
        Args:
            documents: List of {"id": str, "text": str, "metadata": dict}
        """
        texts = []
        self.documents = {}
        self.doc_ids = []
        
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(self.documents)}")
            text = doc.get("text", "")
            
            # Tokenize (simple whitespace + lowercase)
            tokens = text.lower().split()
            texts.append(tokens)
            
            self.documents[doc_id] = {
                "text": text,
                "metadata": doc.get("metadata", {}),
                "original_id": doc_id,
            }
            self.doc_ids.append(doc_id)
        
        # Build BM25
        if texts:
            self.bm25_index = BM25Okapi(texts)
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Cosine similarity search on embeddings.
        
        Args:
            query_embedding: Query vector (e.g., from sentence-transformers)
            doc_embeddings: Dict of {doc_id: embedding_vector}
            top_k: Return top K results
        
        Returns:
            List of SearchResult ordered by score (descending)
        """
        results = []
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return results
        
        for doc_id, doc_embedding in doc_embeddings.items():
            doc_norm = np.linalg.norm(doc_embedding)
            if doc_norm == 0:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
            
            if doc_id in self.documents:
                results.append(SearchResult(
                    document_id=doc_id,
                    content=self.documents[doc_id]["text"],
                    score=float(similarity),
                    source="vector",
                    metadata=self.documents[doc_id]["metadata"],
                ))
        
        # Sort by score descending
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        return results
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        BM25 lexical search.
        
        Args:
            query: Search query string
            top_k: Return top K results
        
        Returns:
            List of SearchResult ordered by score (descending)
        """
        if not self.bm25_index or not self.doc_ids:
            return []
        
        # Tokenize query
        tokens = query.lower().split()
        
        # BM25 scores
        scores = self.bm25_index.get_scores(tokens)
        
        results = []
        for doc_id, score in zip(self.doc_ids, scores):
            if score > 0:  # Only include documents with non-zero score
                results.append(SearchResult(
                    document_id=doc_id,
                    content=self.documents[doc_id]["text"],
                    score=float(score),
                    source="bm25",
                    metadata=self.documents[doc_id]["metadata"],
                ))
        
        # Sort by score descending
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        return results
    
    def fused_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        doc_embeddings: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Fused search combining BM25 and vector scores.
        
        Args:
            query: Search query string
            query_embedding: Query vector embedding
            doc_embeddings: Dict of {doc_id: embedding}
            top_k: Return top K results
        
        Returns:
            Merged & scored results
        """
        # Get results from both methods
        vector_results = self.vector_search(query_embedding, doc_embeddings, top_k=top_k * 2)
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        
        # Normalize scores to [0, 1]
        v_scores = [r.score for r in vector_results]
        b_scores = [r.score for r in bm25_results]
        
        v_max = max(v_scores) if v_scores else 1.0
        b_max = max(b_scores) if b_scores else 1.0
        
        # Merge results by document ID with weighted scores
        fused = {}
        
        for result in vector_results:
            doc_id = result.document_id
            normalized_score = result.score / v_max if v_max > 0 else 0
            weighted_score = normalized_score * self.vector_weight
            
            fused[doc_id] = SearchResult(
                document_id=doc_id,
                content=result.content,
                score=weighted_score,
                source="hybrid",
                metadata=result.metadata,
            )
        
        for result in bm25_results:
            doc_id = result.document_id
            normalized_score = result.score / b_max if b_max > 0 else 0
            weighted_score = normalized_score * self.bm25_weight
            
            if doc_id in fused:
                # Add to existing score
                fused[doc_id].score += weighted_score
            else:
                fused[doc_id] = SearchResult(
                    document_id=doc_id,
                    content=result.content,
                    score=weighted_score,
                    source="hybrid",
                    metadata=result.metadata,
                )
        
        # Sort by fused score
        results = sorted(fused.values(), key=lambda x: x.score, reverse=True)[:top_k]
        return results
    
    def query(
        self,
        q: str,
        query_embedding: Optional[np.ndarray] = None,
        doc_embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Convenience method: if embeddings provided, use fused; else BM25 only.
        
        Args:
            q: Query string
            query_embedding: Optional query embedding
            doc_embeddings: Optional document embeddings dict
            top_k: Return top K results
        """
        if query_embedding is not None and doc_embeddings is not None:
            return self.fused_search(q, query_embedding, doc_embeddings, top_k=top_k)
        else:
            return self.bm25_search(q, top_k=top_k)
