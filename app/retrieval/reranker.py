"""Cross-encoder reranker for improved retrieval quality.

This module provides reranking using cross-encoder models from sentence-transformers
to improve the quality of search results after initial retrieval.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for search results.

    Uses a cross-encoder model to score query-document pairs and rerank
    results. Cross-encoders are more accurate than bi-encoders but slower,
    so they're used as a second-stage reranker on a smaller candidate set.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize reranker.

        Args:
            model_name: Name of the cross-encoder model to use
                                Default: ms-marco-MiniLM-L-6-v2 (fast, good quality)
        """
        self.model_name = model_name
        self._model = None  # Lazy load to avoid loading at import time

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker model: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
                logger.info("Reranker model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, reranking disabled"
                )
                return None
            except Exception as e:
                logger.error("Failed to load reranker model: %s", e)
                return None
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: List of search results with 'content' field
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of results sorted by cross-encoder score
        """
        if not results:
            return []

        if not self.model:
            logger.warning("Reranker model not available, returning original results")
            return results[:top_k]

        # Create query-document pairs
        pairs = [(query, result.get("content", "")) for result in results]

        # Score with cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            return results[:top_k]

        # Sort by reranker score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build final results with reranker score
        final_results = []
        for result, score in scored_results[:top_k]:
            reranked_result = result.copy()
            reranked_result["reranker_score"] = float(score)
            final_results.append(reranked_result)

        logger.debug(
            "Reranked %d results, top score: %.4f",
            len(final_results),
            final_results[0]["reranker_score"] if final_results else 0.0,
        )

        return final_results

    def is_available(self) -> bool:
        """Check if the reranker is available."""
        return self.model is not None
