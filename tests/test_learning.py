"""Tests for learning module."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.learning.simple_edit_store import SimpleEditStore, EditExample, QualityGate


class TestQualityGate:
    """Tests for QualityGate class."""

    def test_meaningful_edit_passes(self):
        """Test that substantial edits pass quality gate."""
        gate = QualityGate()
        is_valid, score = gate.is_meaningful(
            "The plaintiff filed a case.",
            "The plaintiff, Martha Smith, filed a case against Johnson Properties LLC.",
        )
        assert is_valid is True
        assert score > 0.5

    def test_trivial_edit_rejected(self):
        """Test that trivial edits are rejected."""
        gate = QualityGate()
        is_valid, score = gate.is_meaningful(
            "The plaintiff filed.",
            "The plaintiff filed.",  # Same text
        )
        assert is_valid is False
        assert score == 0.0

    def test_whitespace_only_rejected(self):
        """Test that whitespace-only changes are rejected."""
        gate = QualityGate()
        is_valid, score = gate.is_meaningful(
            "Hello world",
            "Hello  world",  # Just extra space
        )
        assert is_valid is False

    def test_short_text_rejected(self):
        """Test that very short texts are rejected."""
        gate = QualityGate()
        is_valid, score = gate.is_meaningful(
            "Hi",
            "Hello",
        )
        assert is_valid is False


class TestEditExample:
    """Tests for EditExample dataclass."""

    def test_edit_creation(self):
        """Test creating an EditExample."""
        edit = EditExample(
            edit_id="edit_123",
            before="Original text",
            after="Edited text",
            reason="Fixed typo",
        )

        assert edit.edit_id == "edit_123"
        assert edit.before == "Original text"
        assert edit.after == "Edited text"
        assert edit.reason == "Fixed typo"


class TestSimpleEditStore:
    """Tests for SimpleEditStore class."""

    @patch("app.learning.simple_edit_store.EmbeddingClient")
    def test_store_initialization(self, mock_embedding_client):
        """Test store initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": tmpdir, "OPENROUTER_API_KEY": "test-key"},
            ):
                store = SimpleEditStore()
                assert store.collection is not None

    @pytest.mark.asyncio
    @patch("app.learning.simple_edit_store.EmbeddingClient")
    async def test_save_edit_quality_rejected(self, mock_embedding_client):
        """Test that low-quality edits are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"CHROMA_PERSIST_DIR": tmpdir, "OPENROUTER_API_KEY": "test-key"},
            ):
                store = SimpleEditStore()

                # Try to save a trivial edit
                result = await store.save_edit(
                    before="Hello",
                    after="Hello",  # Same text
                    reason="No change",
                    draft_type="test",
                )

                assert result is None  # Should be rejected


class TestFewShotRetriever:
    """Tests for FewShotRetriever functionality."""

    def test_format_examples_for_prompt_empty(self):
        """Test formatting empty examples."""
        from app.learning import format_examples_for_prompt

        result = format_examples_for_prompt([])
        assert result == ""

    def test_format_examples_with_content(self):
        """Test formatting examples for prompt."""
        from app.learning import format_examples_for_prompt

        examples = [
            {
                "before": "Before text",
                "after": "After text",
                "reason": "Test reason",
            }
        ]

        result = format_examples_for_prompt(examples)

        assert "Example 1:" in result
        assert "Before text" in result
        assert "After text" in result
        assert "Test reason" in result
        assert "learn from these past improvements" in result.lower()
