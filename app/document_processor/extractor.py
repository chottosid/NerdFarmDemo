"""Document extraction from PDFs, images, and text files.

Supports:
- Hybrid PDF extraction (digital-first, vision fallback for scanned)
- Vision-based processing for images (replaces OCR)
- Direct text file ingestion
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

import pdf2image
from PIL import Image

from app.config import get_settings
from .schemas import DocumentMetadata, ExtractedDocument, Page
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)

# Optional: PyMuPDF for digital PDF text extraction
try:
    import fitz as pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.info("PyMuPDF not available — all PDFs will use vision processing")


class DocumentExtractor:
    """Extract text from PDFs, images, and text files using vision models.

    For digital PDFs with embedded text, uses direct extraction.
    For images and scanned PDFs. uses GPT-4o Vision for accurate extraction
    that preserves tables. signatures. stamps. and layout.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"}

    def __init__(self):
        """Initialize document extractor with vision processor."""
        self.settings = get_settings()
        self.vision_processor = VisionProcessor() if self.settings.use_vision_for_images else None

    def extract(self, file_path: str, doc_id: Optional[str] = None, original_filename: Optional[str] = None) -> ExtractedDocument:
        """Extract text from a document file (sync wrapper).

        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (generated if not provided)
            original_filename: Original filename (if different from file_path name)

        Returns:
            ExtractedDocument with extracted text, metadata, and structured data
        """
        # Just call async version
        return asyncio.run(self.extract_async(file_path, doc_id, original_filename))

    async def extract_async(self, file_path: str, doc_id: Optional[str] = None, original_filename: Optional[str] = None) -> ExtractedDocument:
        """Async version of extract for use in async contexts.

        Args:
            file_path: Path to the document file
            doc_id: Optional document ID
            original_filename: Original filename (if different from file_path name)

        Returns:
            ExtractedDocument with extracted text, metadata, and structured data
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = doc_id or str(uuid.uuid4())
        file_size = path.stat().st_size

        # Default structured data
        structured_data = None

        if ext == ".pdf":
            pages, structured_data = await self._extract_pdf_async(path)
        elif ext == ".txt":
            pages = self._extract_text_file(path)
        else:
            pages, structured_data = await self._extract_image_async(path)

        raw_text = "\n\n".join(page.text for page in pages)
        avg_confidence = sum(p.confidence for p in pages) / len(pages) if pages else 0.0

        metadata = DocumentMetadata(
            total_pages=len(pages),
            avg_confidence=avg_confidence,
            file_size=file_size,
            file_type=ext,
        )

        return ExtractedDocument(
            id=doc_id,
            filename=original_filename or path.name,
            pages=pages,
            raw_text=raw_text,
            metadata=metadata,
            structured_data=structured_data,
        )

    async def _extract_pdf_async(self, path: Path) -> tuple[list[Page], dict | None]:
        """Extract text from PDF — try digital text first, vision fallback.

        Digital PDFs (with embedded text) are extracted directly for higher
        quality and speed. Scanned PDFs fall back to vision processing.

        Args:
            path: Path to PDF file

        Returns:
            Tuple of (List of Page objects, structured_data dict or None)
        """
        # Try digital extraction first (fast, free)
        if PYMUPDF_AVAILABLE:
            pages = self._try_digital_extraction(path)
            if pages and self._is_quality_sufficient(pages):
                logger.info("Using digital text extraction for %s", path.name)
                return pages, None  # No structured data from digital extraction

        # Fallback: Vision processing for scanned PDFs
        logger.info("Using vision processing for %s", path.name)
        return await self._extract_pdf_via_vision(path)

    def _try_digital_extraction(self, path: Path) -> Optional[list[Page]]:
        """Try extracting text directly from PDF without vision model."""
        try:
            doc = pymupdf.open(str(path))
            pages = []
            for i, fitz_page in enumerate(doc, start=1):
                text = fitz_page.get_text().strip()
                if text:
                    pages.append(Page(
                        page_num=i,
                        text=text,
                        confidence=99.0,
                        has_unclear=False,
                    ))
            doc.close()
            return pages if pages else None
        except Exception as e:
            logger.warning("Digital PDF extraction failed: %s", e)
            return None

    def _is_quality_sufficient(self, pages: list[Page]) -> bool:
        """Check if digitally extracted text is meaningful."""
        total_text = sum(len(p.text) for p in pages)
        # At least 50 chars per page on average means real text
        return total_text > 50 * len(pages)

    async def _extract_pdf_via_vision(self, path: Path) -> tuple[list[Page], dict]:
        """Extract text from PDF using vision model on each page image.

        Returns:
            Tuple of (pages, aggregated_structured_data)
        """
        images = pdf2image.convert_from_path(str(path))
        pages = []
        structured_data = {
            "tables": [],
            "signatures": [],
            "stamps_seals": [],
            "parties": [],
            "dates": [],
            "amounts": [],
            "case_ids": [],
            "key_terms": [],
            "document_type": None,
        }

        for i, image in enumerate(images, start=1):
            image = self.vision_processor._upscale_if_needed(image)
            page, extraction = await self.vision_processor.process_image(image, page_num=i)
            pages.append(page)
            # Aggregate structured data from all pages
            self._merge_structured_data(structured_data, extraction)

        return pages, structured_data

    async def _extract_image_async(self, path: Path) -> tuple[list[Page], dict]:
        """Extract text from an image file using vision model.

        Args:
            path: Path to image file

        Returns:
            Tuple of (pages list, structured_data dict)
        """
        image = Image.open(path)
        image = self.vision_processor._upscale_if_needed(image)
        page, extraction = await self.vision_processor.process_image(image, page_num=1)
        return [page], extraction

    def _merge_structured_data(self, target: dict, source: dict) -> None:
        """Merge structured data from a page extraction into target dict."""
        for key in ["tables", "signatures", "stamps_seals", "parties", "dates", "amounts", "case_ids", "key_terms"]:
            if key in source and source[key]:
                if isinstance(source[key], list):
                    target[key].extend(source[key])

        # Document type - take first non-null value
        if not target["document_type"] and source.get("document_type"):
            target["document_type"] = source["document_type"]

    def _extract_text_file(self, path: Path) -> list[Page]:
        """Extract text directly from a text file.

        Args:
            path: Path to text file

        Returns:
            List of Page objects
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Page(
                page_num=1,
                text=text,
                confidence=100.0,
                has_unclear=False,
            )
        ]
