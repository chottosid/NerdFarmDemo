"""Tests for learning module."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.learning.edit_store import OperatorEdit, EditPattern, EditStore


class TestOperatorEdit:
    """Tests for OperatorEdit dataclass."""

    def test_edit_creation(self):
        """Test creating an OperatorEdit."""
        edit = OperatorEdit(
            edit_id="edit_123",
            draft_id="draft_456",
            original_text="Original text",
            edited_text="Edited text",
            edit_reason="Fixed typo",
        )

        assert edit.edit_id == "edit_123"
        assert edit.draft_id == "draft_456"
        assert edit.original_text == "Original text"
        assert edit.edited_text == "Edited text"
        assert edit.edit_reason == "Fixed typo"


class TestEditPattern:
    """Tests for EditPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating an EditPattern."""
        pattern = EditPattern(
            change_type="addition",
            before_snippet="Before",
            after_snippet="After with more content",
            description="Content was expanded",
        )

        assert pattern.change_type == "addition"
        assert "expanded" in pattern.description


class TestEditStore:
    """Tests for EditStore class."""

    @patch("app.learning.edit_store.get_settings")
    def test_init_db(self, mock_settings):
        """Test database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_edits.db")
            mock_settings.return_value = Mock(
                sqlite_db_path=db_path,
            )

            store = EditStore()

            # Database file should exist
            assert os.path.exists(db_path)

    @patch("app.learning.edit_store.get_settings")
    def test_analyze_pattern_addition(self, mock_settings):
        """Test pattern detection for additions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = Mock(
                sqlite_db_path=os.path.join(tmpdir, "test.db"),
            )

            store = EditStore()
            pattern = store._analyze_pattern(
                "Short text",
                "Short text with lots of additional content and details",
            )

            assert pattern is not None
            assert pattern.change_type == "addition"

    @patch("app.learning.edit_store.get_settings")
    def test_analyze_pattern_reduction(self, mock_settings):
        """Test pattern detection for reductions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = Mock(
                sqlite_db_path=os.path.join(tmpdir, "test.db"),
            )

            store = EditStore()
            pattern = store._analyze_pattern(
                "This is a very long piece of text that will be significantly reduced",
                "Short",
            )

            assert pattern is not None
            assert pattern.change_type == "reduction"

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        store = EditStore.__new__(EditStore)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, rel=0.01)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, rel=0.01)

        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert 0.5 < similarity < 1.0


class TestFewShotRetriever:
    """Tests for FewShotRetriever class."""

    def test_format_examples_for_prompt_empty(self):
        """Test formatting empty examples."""
        from app.learning.few_shot import FewShotRetriever

        retriever = FewShotRetriever()
        result = retriever.format_examples_for_prompt([])
        assert result == ""

    def test_format_examples_for_prompt(self):
        """Test formatting examples for prompt."""
        from app.learning.few_shot import FewShotRetriever

        retriever = FewShotRetriever()
        examples = [
            {
                "before": "Before text",
                "after": "After text",
                "reason": "Test reason",
            }
        ]

        result = retriever.format_examples_for_prompt(examples)

        assert "Example 1:" in result
        assert "Before text" in result
        assert "After text" in result
        assert "Test reason" in result
        assert "learned improvements" in result.lower()
