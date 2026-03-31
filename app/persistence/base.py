"""Abstract interfaces for persistence layer.

This module defines protocols and base classes for the persistence layer,
allowing different database backends to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from app.document_processor.schemas import ExtractedDocument


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""

    backend: str = "sqlite"
    path: str = "./edits.db"

    # For future PostgreSQL support:
    # host: str = "localhost"
    # port: int = 5432
    # database: str = "nerdfarm"
    # username: str = ""
    # password: str = ""


class EditRepository(Protocol):
    """Protocol for edit storage backends.

    Defines the interface that all edit storage implementations must follow.
    """

    async def save(
        self,
        draft_id: str,
        original_text: str,
        edited_text: str,
        edit_reason: Optional[str] = None,
        document_context: str = "",
    ) -> str:
        """Save an edit and return its ID."""
        ...

    def get(self, edit_id: str) -> Optional[dict]:
        """Get an edit by ID."""
        ...

    def get_by_draft(self, draft_id: str) -> list[dict]:
        """Get all edits for a draft."""
        ...

    async def get_similar(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        """Find similar edits for few-shot prompting."""
        ...

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent edit history."""
        ...


class DocumentRepository(Protocol):
    """Protocol for document storage backends.

    Defines the interface that all document storage implementations must follow.
    """

    def save(self, doc: ExtractedDocument) -> None:
        """Save a document."""
        ...

    def get(self, doc_id: str) -> Optional[ExtractedDocument]:
        """Get a document by ID."""
        ...

    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        ...

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        ...


class DraftRepository(Protocol):
    """Protocol for draft storage backends."""

    def save(self, draft: dict) -> None:
        """Save a draft."""
        ...

    def get(self, draft_id: str) -> Optional[dict]:
        """Get a draft by ID."""
        ...

    def exists(self, draft_id: str) -> bool:
        """Check if a draft exists."""
        ...
