"""BM25 keyword search implementation.

This module provides keyword-based search using the BM25 algorithm
to complement semantic vector search for hybrid retrieval.
"""

import logging
import re
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 keyword search for document chunks.

    Maintains an in-memory index of document chunks for fast keyword search.
    Uses the Okapi BM25 variant which is best for short to medium-length documents.
    """

    def __init__(self):
        """Initialize BM25 store."""
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.metadatas: list[dict] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the BM25 index.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (filename, page_num, etc.)
        """
        self.documents.append(content)
        self.doc_ids.append(doc_id)
        self.metadatas.append(metadata or {})
        self._rebuild_index()
        logger.debug("Added document %s to BM25 index", doc_id)

    def add_documents(
        self,
        doc_ids: list[str],
        contents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add multiple documents at once.

        Args:
            doc_ids: List of document IDs
            contents: List of document contents
            metadatas: Optional list of metadata dicts
        """
        for i, doc_id in enumerate(doc_ids):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else None
                self.add_document(doc_id, contents[i], metadata)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[dict]:
        """Search for documents using BM25.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dicts with chunk_id, content, score, metadata
        """
        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append({
                        "chunk_id": self.doc_ids[idx],
                        "content": self.documents[idx],
                        "score": float(scores[idx]),
                        "metadata": self.metadatas[idx],
                    })

        return results

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index after adding documents."""
        if not self.documents:
            self.bm25 = None
            return

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.debug("Rebuilt BM25 index with %d documents", len(self.documents))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Simple tokenization that:
        - Lowercases text
        - Removes punctuation
        - Splits on whitespace

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Split on whitespace
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]  # Filter single characters

    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.bm25 = None

    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.documents)
