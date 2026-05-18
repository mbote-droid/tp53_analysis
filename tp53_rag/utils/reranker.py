"""
============================================================
Reranker — Cross-Encoder based Re-ranking
utils/reranker.py
============================================================
Takes top-20 results from search and reranks to top-5
using a lightweight cross-encoder model.
Improves relevance of final LLM context.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RankedDocument:
    """Document with rerank score."""
    document_id: str
    content: str
    relevance_score: float
    original_rank: int
    metadata: Dict = None


class CrossEncoderReranker:
    """
    Lightweight reranking using a cross-encoder-style scorer.
    In production, this would use sentence-transformers:
      from sentence_transformers import CrossEncoder
      model = CrossEncoder("ms-marco-MiniLM-L-6-v2")
    
    For now, we use a simple heuristic scorer for the hackathon demo.
    """
    
    def __init__(self, model_name: str = "heuristic"):
        """
        Args:
            model_name: "heuristic" (lightweight), or actual model path
        """
        self.model_name = model_name
        self.model = None
        
        # Try to load cross-encoder if available
        if model_name != "heuristic":
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
            except ImportError:
                print(f"Warning: sentence-transformers not installed. Falling back to heuristic.")
                self.model = None
    
    def _heuristic_score(self, query: str, document: str) -> float:
        """
        Simple heuristic scorer (no ML model required).
        Scores based on:
        - Query term frequency in document
        - Query term density
        - Term position (earlier = better)
        """
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Tokenize query
        query_terms = [t for t in query_lower.split() if len(t) > 2]
        
        if not query_terms:
            return 0.0
        
        score = 0.0
        doc_tokens = doc_lower.split()
        
        # Term frequency
        term_matches = sum(1 for term in query_terms if term in doc_lower)
        score += (term_matches / len(query_terms)) * 0.4
        
        # Term density (proportion of query terms in document)
        match_positions = []
        for i, token in enumerate(doc_tokens):
            if any(term in token for term in query_terms):
                match_positions.append(i)
        
        if match_positions:
            # Earlier matches are better
            avg_position = np.mean(match_positions) / max(len(doc_tokens), 1)
            position_score = 1.0 - min(avg_position, 1.0)
            score += position_score * 0.3
        
        # Keyword overlap in first 100 tokens (summary effect)
        first_tokens = doc_tokens[:100]
        first_section_matches = sum(1 for term in query_terms if term in ' '.join(first_tokens))
        score += (first_section_matches / max(len(query_terms), 1)) * 0.3
        
        return min(score, 1.0)
    
    def _cross_encoder_score(self, query: str, documents: List[str]) -> List[float]:
        """Use actual cross-encoder model if available."""
        if not self.model:
            return [self._heuristic_score(query, doc) for doc in documents]
        
        try:
            # sentence-transformers cross-encoder returns logits
            query_doc_pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(query_doc_pairs)
            
            # Normalize to [0, 1]
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = scores
            
            return normalized.tolist()
        except Exception as e:
            print(f"Cross-encoder scoring failed: {e}. Falling back to heuristic.")
            return [self._heuristic_score(query, doc) for doc in documents]
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        diversity_filter: bool = True,
    ) -> List[RankedDocument]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query string
            documents: List of {"id": str, "content": str, "source": str, "metadata": dict}
            top_k: Return top K results
            diversity_filter: Avoid 2 documents from same source
        
        Returns:
            List of RankedDocument (top_k sorted by relevance_score desc)
        """
        if not documents:
            return []
        
        doc_contents = [doc.get("content", "") for doc in documents]
        scores = self._cross_encoder_score(query, doc_contents)
        
        # Build ranked list
        ranked = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            ranked.append(RankedDocument(
                document_id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                relevance_score=float(score),
                original_rank=i,
                metadata=doc.get("metadata", {}),
            ))
        
        # Sort by relevance score
        ranked = sorted(ranked, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filter if requested
        if diversity_filter:
            ranked = self._apply_diversity_filter(ranked, top_k)
        else:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _apply_diversity_filter(
        self,
        ranked_docs: List[RankedDocument],
        top_k: int,
    ) -> List[RankedDocument]:
        """
        Ensure diversity: no 2 documents from same source.
        """
        selected = []
        seen_sources = set()
        
        for doc in ranked_docs:
            source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
            
            if source not in seen_sources:
                selected.append(doc)
                seen_sources.add(source)
                
                if len(selected) >= top_k:
                    break
        
        # If we haven't filled top_k, add more regardless of source
        if len(selected) < top_k:
            for doc in ranked_docs:
                if doc not in selected:
                    selected.append(doc)
                    if len(selected) >= top_k:
                        break
        
        return selected[:top_k]
    
    def rerank_and_merge(
        self,
        query: str,
        search_results: List[Dict],
        top_k: int = 5,
    ) -> List[RankedDocument]:
        """
        Convenience: accepts search results from HybridSearchEngine.
        Converts to standard format and reranks.
        """
        docs_for_rerank = []
        for result in search_results:
            docs_for_rerank.append({
                "id": result.get("document_id", result.get("id", "")),
                "content": result.get("content", ""),
                "source": result.get("source", "unknown"),
                "metadata": result.get("metadata", {}),
            })
        
        return self.rerank(query, docs_for_rerank, top_k=top_k)
