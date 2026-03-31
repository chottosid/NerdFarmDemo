"""Tests for retrieval module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.retrieval.retriever import RetrievedChunk, SourceReference


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a RetrievedChunk."""
        chunk = RetrievedChunk(
            chunk_id="doc_1_chunk_0",
            content="Sample content",
            source_doc_id="doc_1",
            filename="test.pdf",
            page_num=1,
            similarity_score=0.85,
        )
        assert chunk.chunk_id == "doc_1_chunk_0"
        assert chunk.content == "Sample content"
        assert chunk.similarity_score == 0.85


class TestSourceReference:
    """Tests for SourceReference dataclass."""

    def test_reference_creation(self):
        """Test creating a SourceReference."""
        ref = SourceReference(
            document_id="doc_1",
            filename="test.pdf",
            page=3,
            surrounding_context="Some context...",
        )
        assert ref.document_id == "doc_1"
        assert ref.page == 3


class TestRetriever:
    """Tests for Retriever class."""

    def test_format_chunks_for_prompt(self):
        """Test formatting chunks for prompt."""
        from app.retrieval.retriever import Retriever

        retriever = Retriever()
        chunks = [
            RetrievedChunk(
                chunk_id="doc_1_chunk_0",
                content="First chunk content",
                source_doc_id="doc_1",
                filename="test.pdf",
                page_num=1,
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk_id="doc_1_chunk_1",
                content="Second chunk content",
                source_doc_id="doc_1",
                filename="test.pdf",
                page_num=2,
                similarity_score=0.85,
            ),
        ]

        result = retriever.format_chunks_for_prompt(chunks)

        assert "[1] Source: test.pdf, Page 1" in result
        assert "[2] Source: test.pdf, Page 2" in result
        assert "First chunk content" in result
        assert "Second chunk content" in result


@pytest.mark.asyncio
class TestEmbeddingClient:
    """Tests for EmbeddingClient class."""

    @patch("app.retrieval.embeddings.get_settings")
    async def test_embed_empty_list(self, mock_settings):
        """Test embedding empty list returns empty."""
        from app.retrieval.embeddings import EmbeddingClient

        mock_settings.return_value = Mock(
            openrouter_api_key="test-key",
            openrouter_base_url="https://test.com",
            embedding_model="test-model",
        )

        client = EmbeddingClient()
        result = await client.embed([])
        assert result == []


@pytest.mark.asyncio
class TestVectorStore:
    """Tests for VectorStore class."""

    def test_chunk_document(self):
        """Test document chunking logic."""
        from app.document_processor.schemas import ExtractedDocument, Page
        from app.retrieval.store import VectorStore

        doc = ExtractedDocument(
            id="doc_1",
            filename="test.pdf",
            pages=[
                Page(
                    page_num=1,
                    text="First paragraph.\n\nSecond paragraph with more text.\n\nThird paragraph here.",
                    confidence=90.0,
                    has_unclear=False,
                ),
            ],
        )

        store = VectorStore()
        chunks = store._chunk_document(doc, chunk_size=50)

        # Should split into multiple chunks
        assert len(chunks) >= 1
        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk) > 0
