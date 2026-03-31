"""API endpoints for document processing."""

import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.document_processor import DocumentExtractor, ExtractedDocument
from app.persistence import DocumentStore
from app.retrieval import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Persistent document store (file-backed, survives restarts)
_doc_store: DocumentStore | None = None


def get_doc_store() -> DocumentStore:
    """Get or create document store instance."""
    global _doc_store
    if _doc_store is None:
        _doc_store = DocumentStore()
    return _doc_store


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:
    """Upload and process a document.

    Args:
        file: Uploaded file (PDF, image, or text)

    Returns:
        Document ID and processing metadata
    """
    settings = get_settings()

    # Validate file type
    allowed_types = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}",
        )

    # Check file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size / (1024 * 1024):.1f}MB",
        )

    # Save file temporarily
    doc_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_path = os.path.join(settings.upload_dir, f"{doc_id}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # Process document with vision model (for images/scanned PDFs) or digital extraction
        extractor = DocumentExtractor()
        doc = await extractor.extract_async(temp_path, doc_id=doc_id, original_filename=file.filename)

        # Note: Vision model extracts structured data directly, no separate OCR correction needed
        # The vision processor already extracts: text, tables, signatures, stamps, entities

        # Store in vector database
        store = VectorStore()
        chunk_count = await store.add_document(doc)

        # Persist to file-backed store (survives restarts)
        doc_store = get_doc_store()
        doc_store.save(doc)

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "total_pages": doc.metadata.total_pages if doc.metadata else 0,
            "avg_confidence": doc.metadata.avg_confidence if doc.metadata else 0.0,
            "chunks_created": chunk_count,
            "has_unclear_sections": any(p.has_unclear for p in doc.pages),
            "structured_data": doc.structured_data,
        }

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("")
async def list_documents() -> list[dict]:
    """List all processed documents.

    Returns:
        List of document metadata
    """
    import json
    from pathlib import Path

    settings = get_settings()
    store_dir = Path(settings.upload_dir) / ".doc_store"

    # Get actual chunk counts from vector store
    store = VectorStore()

    documents = []
    if store_dir.exists():
        for doc_file in store_dir.glob("*.json"):
            try:
                data = json.loads(doc_file.read_text())
                doc_id = data["id"]

                # Query actual chunk count from ChromaDB
                chunk_results = store.collection.get(
                    where={"source_doc_id": doc_id},
                    include=[]
                )
                chunk_count = len(chunk_results["ids"]) if chunk_results["ids"] else 0

                documents.append({
                    "document_id": doc_id,
                    "filename": data["filename"],
                    "total_pages": len(data.get("pages", [])),
                    "chunks_created": chunk_count,
                    "avg_confidence": data.get("metadata", {}).get("avg_confidence", 0.0) if data.get("metadata") else 0.0,
                })
            except Exception as e:
                logger.warning(f"Failed to read document file {doc_file}: {e}")

    return documents


@router.get("/{document_id}")
async def get_document(document_id: str) -> dict:
    """Get a processed document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document details and metadata
    """
    doc_store = get_doc_store()
    doc = doc_store.get(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "total_pages": len(doc.pages),
        "metadata": {
            "total_pages": doc.metadata.total_pages if doc.metadata else 0,
            "avg_confidence": doc.metadata.avg_confidence if doc.metadata else 0.0,
            "file_type": doc.metadata.file_type if doc.metadata else "",
        },
        "pages": [
            {
                "page_num": p.page_num,
                "text": p.text,
                "confidence": p.confidence,
                "has_unclear": p.has_unclear,
            }
            for p in doc.pages
        ],
    }


@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str) -> dict:
    """Get document chunks from vector store.

    Args:
        document_id: Document ID

    Returns:
        List of document chunks with metadata
    """
    doc_store = get_doc_store()
    doc = doc_store.get(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    store = VectorStore()

    # Query all chunks for this document
    results = store.collection.get(
        where={"source_doc_id": document_id},
        include=["documents", "metadatas"],
    )

    chunks = []
    if results["documents"]:
        for i, (content, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            chunks.append({
                "chunk_id": results["ids"][i],
                "content": content,
                "page_num": metadata.get("page_num", 1),
            })

    return {
        "document_id": document_id,
        "filename": doc.filename,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Delete a document and its chunks.

    Args:
        document_id: Document ID

    Returns:
        Deletion confirmation
    """
    doc_store = get_doc_store()

    if not doc_store.exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from vector store
    store = VectorStore()
    deleted_chunks = store.delete_document(document_id)

    # Remove from persistent store
    doc_store.delete(document_id)

    # Remove uploaded file
    settings = get_settings()
    for ext in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"]:
        temp_path = os.path.join(settings.upload_dir, f"{document_id}{ext}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "document_id": document_id,
        "deleted": True,
        "chunks_removed": deleted_chunks,
    }
