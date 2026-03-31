"""Vector store using ChromaDB."""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.document_processor.schemas import ExtractedDocument

from .embeddings import EmbeddingClient


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    COLLECTION_NAME = "documents"

    def __init__(self):
        """Initialize vector store."""
        settings = get_settings()

        # Ensure persist directory exists
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_client = EmbeddingClient()

    async def add_document(self, doc: ExtractedDocument) -> int:
        """Add document chunks to the vector store.

        Args:
            doc: Extracted document to add

        Returns:
            Number of chunks added
        """
        # Chunk with page tracking
        chunks_with_pages = self._chunk_document_with_pages(doc)
        if not chunks_with_pages:
            return 0

        chunks = [c[0] for c in chunks_with_pages]
        page_nums = [c[1] for c in chunks_with_pages]

        # Generate embeddings
        embeddings = await self.embedding_client.embed(chunks)

        # Generate IDs and metadata
        ids = [f"{doc.id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_doc_id": doc.id,
                "filename": doc.filename,
                "page_num": page_nums[i],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    async def search(
        self,
        query: str,
        k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Search for similar chunks.

        Args:
            query: Search query
            k: Number of results to return
            doc_ids: Optional list of document IDs to filter by

        Returns:
            List of retrieved chunks with metadata
        """
        # Generate query embedding
        query_embedding = await self.embedding_client.embed_single(query)

        # Build filter if doc_ids provided
        where_filter = None
        if doc_ids:
            where_filter = {"source_doc_id": {"$in": doc_ids}}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - distance

                chunks.append({
                    "chunk_id": f"{metadata['source_doc_id']}_chunk_{metadata['chunk_index']}",
                    "content": doc,
                    "source_doc_id": metadata["source_doc_id"],
                    "filename": metadata["filename"],
                    "page_num": metadata["page_num"],
                    "similarity_score": similarity,
                })

        return chunks

    def _chunk_document_with_pages(
        self, doc: ExtractedDocument, chunk_size: int = 500
    ) -> list[tuple[str, int]]:
        """Chunk document into smaller pieces for embedding, tracking page numbers.

        Uses semantic chunking by paragraphs when possible. Each chunk is paired
        with the page number it originated from.

        Args:
            doc: Document to chunk
            chunk_size: Target size for each chunk in characters

        Returns:
            List of (chunk_text, page_num) tuples
        """
        chunks_with_pages = []

        # Process each page separately to maintain page boundaries
        for page in doc.pages:
            text = page.text
            page_num = page.page_num

            # Split by paragraphs
            paragraphs = text.split("\n\n")

            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If adding this paragraph exceeds chunk size, save current chunk
                if current_chunk and len(current_chunk) + len(para) > chunk_size:
                    chunks_with_pages.append((current_chunk.strip(), page_num))
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            # Don't forget the last chunk of the page
            if current_chunk.strip():
                chunks_with_pages.append((current_chunk.strip(), page_num))

        return chunks_with_pages

    def _chunk_document(self, doc: ExtractedDocument, chunk_size: int = 500) -> list[str]:
        """Chunk document into smaller pieces for embedding.

        Backward-compatible wrapper that returns only chunk text.

        Args:
            doc: Document to chunk
            chunk_size: Target size for each chunk in characters

        Returns:
            List of text chunks
        """
        chunks_with_pages = self._chunk_document_with_pages(doc, chunk_size)
        return [chunk for chunk, _ in chunks_with_pages]

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"source_doc_id": doc_id},
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def get_document_count(self) -> int:
        """Get total number of documents in store."""
        return self.collection.count()
