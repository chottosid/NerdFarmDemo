"""Hybrid search combining BM25 keyword search with semantic vector search.

This module implements hybrid retrieval using Reciprocal Rank Fusion (RRF)
to combine results from multiple search methods.
"""

import logging
from typing import Optional

from .bm25_store import BM25Store
from .store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearch:
    """Combines BM25 keyword search with semantic vector search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods.
    RRF formula: score = sum(1 / (k + rank)) for each ranking system
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        rrf_k: int = 60,  # RRF constant
    ):
        """Initialize hybrid search.

        Args:
            vector_store: Vector store for semantic search
            bm25_store: BM25 store for keyword search
            rrf_k: RRF constant (default 60)
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (0.0-1.0)
                         0.0 = keyword only, 1.0 = vector only
                         Default 0.5 = equal blend
            doc_ids: Optional filter by document IDs

        Returns:
            List of search results sorted by combined score
        """
        # Get more candidates from each method for better fusion
        candidate_k = k * 3

        # 1. Vector search
        vector_results = await self.vector_store.search(
            query, k=candidate_k, doc_ids=doc_ids
        )
        logger.debug(
            "Vector search returned %d results for query: %s",
            len(vector_results),
            query[:50],
        )

        # 2. BM25 keyword search
        bm25_results = self.bm25_store.search(query, k=candidate_k)
        logger.debug(
            "BM25 search returned %d results for query: %s",
            len(bm25_results),
            query[:50],
        )

        # 3. Combine using RRF
        combined = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            alpha=alpha,
            k=k,
        )

        return combined

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        alpha: float,
        k: int,
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = alpha * rrf_vector + (1 - alpha) * rrf_keyword
        where rrf_x = 1 / (rrf_k + rank)

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for vector search
            k: Number of results to return

        Returns:
            Combined and sorted results
        """
        # Build rank maps
        vector_ranks = {}
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            vector_ranks[chunk_id] = rank + 1  # 1-indexed

        bm25_ranks = {}
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            bm25_ranks[chunk_id] = rank + 1  # 1-indexed

        # Calculate RRF scores
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        combined_scores = {}

        for chunk_id in all_chunk_ids:
            # RRF score from vector
            vector_rrf = 0.0
            if chunk_id in vector_ranks:
                vector_rrf = 1.0 / (self.rrf_k + vector_ranks[chunk_id])

            # RRF score from BM25
            bm25_rrf = 0.0
            if chunk_id in bm25_ranks:
                bm25_rrf = 1.0 / (self.rrf_k + bm25_ranks[chunk_id])

            # Combined score with alpha weighting
            combined_scores[chunk_id] = alpha * vector_rrf + (1 - alpha) * bm25_rrf

        # Get full result data
        results_map = {}
        for result in vector_results + bm25_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in results_map:
                results_map[chunk_id] = result

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )[:k]

        # Build final results
        final_results = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id].copy()
            result["hybrid_score"] = combined_scores[chunk_id]
            final_results.append(result)

        logger.debug(
            "Hybrid search returned %d results with top score %.4f",
            len(final_results),
            final_results[0]["hybrid_score"] if final_results else 0.0,
        )

        return final_results
