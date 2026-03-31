"""Retriever for document search with citations.

Supports:
- Semantic vector search (ChromaDB)
- BM25 keyword search
- Hybrid search (vector + keyword with RRF fusion)
- Cross-encoder reranking
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings

from .bm25_store import BM25Store
from .hybrid_search import HybridSearch
from .reranker import Reranker
from .store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with citation info."""

    chunk_id: str
    content: str
    source_doc_id: str
    filename: str
    page_num: int
    similarity_score: float
    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None


@dataclass
class SourceReference:
    """Reference to source document location."""

    document_id: str
    filename: str
    page: int
    surrounding_context: str


class Retriever:
    """High-level retriever for document search.

    Supports multiple retrieval modes:
    - Semantic: Vector search only
    - Keyword: BM25 search only
    - Hybrid: Combined vector + keyword with RRF fusion
    - Reranked: Hybrid + cross-encoder reranking
    """

    def __init__(
        self,
        use_hybrid: bool | None = True,
        use_reranker: bool | None = True,
    ):
        """Initialize retriever.

        Args:
            use_hybrid: Enable hybrid search (vector + keyword). None uses settings.
            use_reranker: Enable cross-encoder reranking. None uses settings.
        """
        settings = get_settings()

        # Use settings if not explicitly specified
        if use_hybrid is None:
            use_hybrid = settings.use_hybrid_search
        if use_reranker is None:
            use_reranker = settings.use_reranker

        # Core components
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

        # Hybrid search
        self.use_hybrid = use_hybrid
        if use_hybrid:
            self.hybrid_search = HybridSearch(self.vector_store, self.bm25_store)
            logger.info("Hybrid search enabled (alpha=%.2f)", settings.hybrid_alpha)
        else:
            self.hybrid_search = None

        # Reranker
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = Reranker(model_name=settings.reranker_model)
            logger.info("Reranker enabled (model=%s)", settings.reranker_model)
        else:
            self.reranker = None

        # Sync BM25 index with existing vector store documents
        self._sync_bm25_index()

    def _sync_bm25_index(self) -> None:
        """Sync BM25 index with existing documents in vector store."""
        try:
            # Get all documents from ChromaDB
            results = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )

            if results["documents"]:
                self.bm25_store.add_documents(
                    doc_ids=results["ids"],
                    contents=results["documents"],
                    metadatas=results["metadatas"],
                )
                logger.info(
                    "Synced %d documents to BM25 index", len(results["documents"])
                )
        except Exception as e:
            logger.warning("Failed to sync BM25 index: %s", e)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query
            k: Number of results
            doc_ids: Optional filter by document IDs

        Returns:
            List of RetrievedChunk objects
        """
        settings = get_settings()

        # Stage 1: Initial retrieval
        if self.use_hybrid and self.hybrid_search:
            # Hybrid search: get more candidates for reranking
            candidates = await self.hybrid_search.search(
                query=query,
                k=k * 3,  # Get more for reranking
                alpha=settings.hybrid_alpha,
                doc_ids=doc_ids,
            )
        else:
            # Fallback to vector search only
            candidates = await self.vector_store.search(query, k=k * 3, doc_ids=doc_ids)

        # Stage 2: Reranking
        if self.use_reranker and self.reranker and self.reranker.is_available():
            candidates = await self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=k,
            )

        # Convert to RetrievedChunk objects
        chunks = []
        for r in candidates[:k]:
            chunk = RetrievedChunk(
                chunk_id=r["chunk_id"],
                content=r["content"],
                source_doc_id=r["source_doc_id"],
                filename=r["filename"],
                page_num=r["page_num"],
                similarity_score=r.get("similarity_score", 0.0),
                hybrid_score=r.get("hybrid_score"),
                reranker_score=r.get("reranker_score"),
            )
            chunks.append(chunk)

        logger.debug(
            "Retrieved %d chunks for query: %s",
            len(chunks),
            query[:50],
        )

        return chunks

    async def get_source_reference(
        self,
        chunk: RetrievedChunk,
    ) -> SourceReference:
        """Get detailed source reference for a chunk.

        Args:
            chunk: Retrieved chunk

        Returns:
            SourceReference with context
        """
        # For now, use the chunk content as context
        # In a full implementation, we'd fetch surrounding text from the original document
        return SourceReference(
            document_id=chunk.source_doc_id,
            filename=chunk.filename,
            page=chunk.page_num,
            surrounding_context=chunk.content[:200] + "...",
        )

    def format_chunks_for_prompt(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Format retrieved chunks for inclusion in a prompt.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted string with citations
        """
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"[{i}] Source: {chunk.filename}, Page {chunk.page_num}\n"
                f"{chunk.content}\n"
            )

        return "\n".join(formatted)

    def add_document_to_bm25(
        self,
        doc_id: str,
        content: str,
        metadata: dict,
    ) -> None:
        """Add a document to the BM25 index.

        Call this when adding new documents to keep BM25 in sync.

        Args:
            doc_id: Document/chunk ID
            content: Document content
            metadata: Document metadata
        """
        self.bm25_store.add_document(doc_id, content, metadata)
