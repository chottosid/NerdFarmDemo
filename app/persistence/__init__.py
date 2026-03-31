"""Persistence layer for NerdFarm.

This module provides abstracted storage backends for:
- Edits (operator edit history)
- Documents (processed documents)
- Drafts (generated drafts)

The factory pattern allows easy switching between database backends.
"""

from .base import DatabaseConfig, DocumentRepository, DraftRepository, EditRepository
from .factory import RepositoryFactory
from .stores import DocumentStore, DraftStore

__all__ = [
    "DatabaseConfig",
    "EditRepository",
    "DocumentRepository",
    "DraftRepository",
    "RepositoryFactory",
    "DocumentStore",
    "DraftStore",
]
