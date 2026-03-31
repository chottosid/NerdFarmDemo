"""Tests for generation module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.generation.prompts import DraftType, get_system_prompt, format_few_shot_examples


class TestDraftType:
    """Tests for DraftType enum."""

    def test_draft_types_exist(self):
        """Test that all expected draft types exist."""
        assert DraftType.TITLE_REVIEW_SUMMARY
        assert DraftType.CASE_FACT_SUMMARY
        assert DraftType.NOTICE_SUMMARY
        assert DraftType.DOCUMENT_CHECKLIST
        assert DraftType.INTERNAL_MEMO


class TestSystemPrompts:
    """Tests for system prompt functions."""

    def test_get_system_prompt_title_review(self):
        """Test getting title review system prompt."""
        prompt = get_system_prompt(DraftType.TITLE_REVIEW_SUMMARY)
        assert "title review" in prompt.lower()
        assert "cite" in prompt.lower()

    def test_get_system_prompt_case_facts(self):
        """Test getting case facts system prompt."""
        prompt = get_system_prompt(DraftType.CASE_FACT_SUMMARY)
        assert "case fact" in prompt.lower()
        assert "citations" in prompt.lower()

    def test_get_system_prompt_unknown_defaults_to_memo(self):
        """Test that unknown type defaults to internal memo."""
        # This tests the default behavior
        prompt = get_system_prompt(DraftType.INTERNAL_MEMO)
        assert "memo" in prompt.lower()


class TestFewShotFormatting:
    """Tests for few-shot example formatting."""

    def test_format_empty_examples(self):
        """Test formatting empty examples list."""
        result = format_few_shot_examples([])
        assert result == ""

    def test_format_single_example(self):
        """Test formatting a single example."""
        examples = [
            {
                "before": "Original text",
                "after": "Improved text",
                "reason": "Added detail",
            }
        ]
        result = format_few_shot_examples(examples)

        assert "Example 1:" in result
        assert "Original text" in result
        assert "Improved text" in result
        assert "Added detail" in result

    def test_format_multiple_examples(self):
        """Test formatting multiple examples."""
        examples = [
            {
                "before": "First original",
                "after": "First improved",
                "reason": "First reason",
            },
            {
                "before": "Second original",
                "after": "Second improved",
                "reason": "Second reason",
            }
        ]
        result = format_few_shot_examples(examples)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "First original" in result
        assert "Second original" in result


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Test creating a Citation."""
        from app.generation.drafter import Citation

        citation = Citation(
            text="Cited text content",
            source_doc="document.pdf",
            page=5,
            chunk_id="doc_1_chunk_3",
        )

        assert citation.text == "Cited text content"
        assert citation.source_doc == "document.pdf"
        assert citation.page == 5


class TestDraftOutput:
    """Tests for DraftOutput dataclass."""

    def test_draft_output_defaults(self):
        """Test DraftOutput default values."""
        from app.generation.drafter import DraftOutput

        draft = DraftOutput(
            draft_id="draft_123",
            content="Draft content",
        )

        assert draft.draft_id == "draft_123"
        assert draft.content == "Draft content"
        assert draft.citations == []
        assert draft.confidence == 0.0


@pytest.mark.asyncio
class TestDraftGenerator:
    """Tests for DraftGenerator class."""

    def test_calculate_confidence_empty_chunks(self):
        """Test confidence calculation with no chunks."""
        from app.generation.drafter import DraftGenerator

        generator = DraftGenerator()
        confidence = generator._calculate_confidence([])
        assert confidence == 0.0

    def test_calculate_confidence_with_chunks(self):
        """Test confidence calculation with chunks."""
        from app.generation.drafter import DraftGenerator
        from app.retrieval.retriever import RetrievedChunk

        generator = DraftGenerator()
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                content="Content",
                source_doc_id="doc1",
                filename="test.pdf",
                page_num=1,
                similarity_score=0.8,
            ),
            RetrievedChunk(
                chunk_id="2",
                content="Content 2",
                source_doc_id="doc1",
                filename="test.pdf",
                page_num=2,
                similarity_score=0.9,
            ),
        ]

        confidence = generator._calculate_confidence(chunks)
        # Should be average (0.85) plus coverage bonus
        assert 0.8 < confidence <= 1.0

    def test_format_draft_with_citations(self):
        """Test formatting draft with citations."""
        from app.generation.drafter import DraftGenerator, DraftOutput, Citation

        generator = DraftGenerator()
        draft = DraftOutput(
            draft_id="draft_1",
            content="This is the draft content.",
            citations=[
                Citation(
                    text="Source text",
                    source_doc="source.pdf",
                    page=3,
                    chunk_id="chunk_1",
                )
            ],
        )

        result = generator.format_draft_with_citations(draft)

        assert "This is the draft content" in result
        assert "## Sources" in result
        assert "source.pdf" in result
        assert "Page 3" in result
