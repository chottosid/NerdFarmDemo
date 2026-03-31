"""Lightweight JSON-based persistence for documents and drafts.

Replaces in-memory dicts with file-backed storage that survives
server restarts.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.document_processor.schemas import (
    DocumentMetadata,
    ExtractedDocument,
    Page,
)

logger = logging.getLogger(__name__)


class DocumentStore:
    """File-backed store for processed documents."""

    def __init__(self):
        """Initialize document store."""
        settings = get_settings()
        self.store_dir = Path(settings.upload_dir) / ".doc_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, doc: ExtractedDocument) -> None:
        """Save a processed document to disk.

        Args:
            doc: ExtractedDocument to persist
        """
        data = {
            "id": doc.id,
            "filename": doc.filename,
            "raw_text": doc.raw_text,
            "pages": [
                {
                    "page_num": p.page_num,
                    "text": p.text,
                    "confidence": p.confidence,
                    "has_unclear": p.has_unclear,
                }
                for p in doc.pages
            ],
            "metadata": {
                "total_pages": doc.metadata.total_pages,
                "avg_confidence": doc.metadata.avg_confidence,
                "file_size": doc.metadata.file_size,
                "file_type": doc.metadata.file_type,
            } if doc.metadata else None,
        }

        path = self.store_dir / f"{doc.id}.json"
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved document %s to %s", doc.id, path)

    def get(self, doc_id: str) -> Optional[ExtractedDocument]:
        """Load a document from disk.

        Args:
            doc_id: Document ID

        Returns:
            ExtractedDocument or None if not found
        """
        path = self.store_dir / f"{doc_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())

        pages = [
            Page(
                page_num=p["page_num"],
                text=p["text"],
                confidence=p["confidence"],
                has_unclear=p["has_unclear"],
            )
            for p in data["pages"]
        ]

        metadata = None
        if data.get("metadata"):
            m = data["metadata"]
            metadata = DocumentMetadata(
                total_pages=m["total_pages"],
                avg_confidence=m["avg_confidence"],
                file_size=m.get("file_size", 0),
                file_type=m.get("file_type", ""),
            )

        return ExtractedDocument(
            id=data["id"],
            filename=data["filename"],
            pages=pages,
            raw_text=data.get("raw_text", ""),
            metadata=metadata,
        )

    def delete(self, doc_id: str) -> bool:
        """Delete a document from disk.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        path = self.store_dir / f"{doc_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists.

        Args:
            doc_id: Document ID

        Returns:
            True if document exists
        """
        return (self.store_dir / f"{doc_id}.json").exists()


class DraftStore:
    """File-backed store for generated drafts."""

    def __init__(self):
        """Initialize draft store."""
        settings = get_settings()
        self.store_dir = Path(settings.upload_dir) / ".draft_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, draft) -> None:
        """Save a draft to disk.

        Args:
            draft: DraftOutput to persist
        """
        data = {
            "draft_id": draft.draft_id,
            "content": draft.content,
            "citations": [
                {
                    "text": c.text,
                    "source_doc": c.source_doc,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                }
                for c in draft.citations
            ],
            "confidence": draft.confidence,
            "draft_type": draft.draft_type,
            "query": draft.query,
            "generated_at": draft.generated_at.isoformat(),
        }

        path = self.store_dir / f"{draft.draft_id}.json"
        path.write_text(json.dumps(data, indent=2))

    def get(self, draft_id: str):
        """Load a draft from disk.

        Args:
            draft_id: Draft ID

        Returns:
            DraftOutput or None if not found
        """
        from datetime import datetime
        from app.generation.drafter import Citation, DraftOutput

        path = self.store_dir / f"{draft_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())

        citations = [
            Citation(
                text=c["text"],
                source_doc=c["source_doc"],
                page=c["page"],
                chunk_id=c["chunk_id"],
            )
            for c in data.get("citations", [])
        ]

        return DraftOutput(
            draft_id=data["draft_id"],
            content=data["content"],
            citations=citations,
            confidence=data.get("confidence", 0.0),
            generated_at=datetime.fromisoformat(data["generated_at"]),
            draft_type=data.get("draft_type", ""),
            query=data.get("query", ""),
        )

    def exists(self, draft_id: str) -> bool:
        """Check if a draft exists."""
        return (self.store_dir / f"{draft_id}.json").exists()
