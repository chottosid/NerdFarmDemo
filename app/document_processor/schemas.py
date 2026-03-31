"""Data models for document processing."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Page:
    """Represents a single processed page."""

    page_num: int
    text: str
    confidence: float
    has_unclear: bool


@dataclass
class DocumentMetadata:
    """Metadata about a processed document."""

    total_pages: int
    avg_confidence: float
    processing_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_size: int = 0
    file_type: str = ""


@dataclass
class ExtractedDocument:
    """Complete extracted document with text and metadata."""

    id: str
    filename: str
    pages: list[Page] = field(default_factory=list)
    raw_text: str = ""
    metadata: Optional[DocumentMetadata] = None
    structured_data: Optional[dict] = None
    """LLM-extracted entities: parties, dates, case IDs, amounts, etc."""

    def get_text_with_page_markers(self) -> str:
        """Return text with page markers for citation tracking."""
        result = []
        for page in self.pages:
            result.append(f"[PAGE {page.page_num}]\n{page.text}")
        return "\n\n".join(result)

