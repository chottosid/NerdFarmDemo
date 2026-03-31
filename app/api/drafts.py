"""API endpoints for draft generation."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.generation import DraftGenerator, DraftType
from app.learning import SimpleEditStore, format_examples_for_prompt
from app.persistence import DraftStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/drafts", tags=["drafts"])

# Persistent draft store (file-backed, survives restarts)
_draft_store: DraftStore | None = None
_edit_store: SimpleEditStore | None = None


def get_draft_store() -> DraftStore:
    """Get or create draft store instance."""
    global _draft_store
    if _draft_store is None:
        _draft_store = DraftStore()
    return _draft_store


def get_edit_store() -> SimpleEditStore:
    """Get or create edit store instance."""
    global _edit_store
    if _edit_store is None:
        _edit_store = SimpleEditStore()
    return _edit_store


class GenerateDraftRequest(BaseModel):
    """Request model for draft generation."""

    query: str = Field(..., description="Query or request for the draft")
    draft_type: str = Field(
        default="internal_memo",
        description="Type of draft to generate",
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional list of document IDs to use",
    )


class GenerateDraftResponse(BaseModel):
    """Response model for draft generation."""

    draft_id: str
    content: str
    citations: list[dict]
    confidence: float
    draft_type: str
    edit_examples_used: int = Field(default=0, description="Number of past edits used for few-shot learning")
    # Grounding transparency fields
    is_grounded: bool = Field(default=True, description="Whether draft is grounded in source documents")
    grounding_warning: str = Field(default="", description="Warning if insufficient evidence")
    retrieved_chunks: list[dict] = Field(
        default_factory=list,
        description="Chunks retrieved to support this draft",
    )


@router.post("/generate", response_model=GenerateDraftResponse)
async def generate_draft(request: GenerateDraftRequest) -> GenerateDraftResponse:
    """Generate a draft document from query.

    Args:
        request: Draft generation request

    Returns:
        Generated draft with citations
    """
    # Validate draft type
    try:
        draft_type = DraftType(request.draft_type)
    except ValueError:
        valid_types = [dt.value for dt in DraftType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid draft_type. Valid types: {valid_types}",
        )

    # Retrieve similar past edits for few-shot learning
    few_shot_examples = None
    try:
        edit_store = get_edit_store()
        similar_edits = await edit_store.get_similar_edits(
            query=request.query,
            draft_type=request.draft_type,
            k=3,
        )
        if similar_edits:
            few_shot_examples = format_examples_for_prompt(similar_edits)
            logger.info(
                "Found %d relevant past edits to improve generation",
                len(similar_edits),
            )
    except Exception as e:
        logger.warning("Failed to retrieve edit examples: %s", e)

    # Generate draft with learned improvements
    generator = DraftGenerator()
    draft = await generator.generate(
        query=request.query,
        draft_type=draft_type,
        document_ids=request.document_ids,
        few_shot_examples=few_shot_examples,
    )

    # Persist for later retrieval
    draft_store = get_draft_store()
    draft_store.save(draft)

    return GenerateDraftResponse(
        draft_id=draft.draft_id,
        content=draft.content,
        citations=[
            {
                "text": c.text,
                "source_doc": c.source_doc,
                "page": c.page,
                "chunk_id": c.chunk_id,
            }
            for c in draft.citations
        ],
        confidence=draft.confidence,
        draft_type=draft.draft_type,
        edit_examples_used=len(similar_edits) if similar_edits else 0,
        is_grounded=getattr(draft, "is_grounded", True),
        grounding_warning=getattr(draft, "grounding_warning", ""),
        retrieved_chunks=[
            {
                "text": c.text,
                "source_doc": c.source_doc,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": c.score,
            }
            for c in getattr(draft, "retrieved_chunks", [])
        ],
    )


@router.get("/types/available")
async def get_draft_types() -> dict:
    """Get available draft types.

    Returns:
        List of available draft types with descriptions
    """
    return {
        "draft_types": [
            {
                "value": dt.value,
                "name": dt.name,
                "description": _get_draft_type_description(dt),
            }
            for dt in DraftType
        ]
    }


@router.get("/{draft_id}")
async def get_draft(draft_id: str) -> dict:
    """Get a generated draft by ID.

    Args:
        draft_id: Draft ID

    Returns:
        Draft content with citations
    """
    draft_store = get_draft_store()
    draft = draft_store.get(draft_id)

    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")

    return {
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


@router.get("/{draft_id}/formatted")
async def get_formatted_draft(draft_id: str) -> dict:
    """Get draft formatted with inline citations.

    Args:
        draft_id: Draft ID

    Returns:
        Formatted draft with citations section
    """
    draft_store = get_draft_store()
    draft = draft_store.get(draft_id)

    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    generator = DraftGenerator()

    formatted = generator.format_draft_with_citations(draft)

    return {
        "draft_id": draft_id,
        "formatted_content": formatted,
    }


def _get_draft_type_description(draft_type: DraftType) -> str:
    """Get human-readable description for draft type."""
    descriptions = {
        DraftType.TITLE_REVIEW_SUMMARY: "Summary of title review findings from property documents",
        DraftType.CASE_FACT_SUMMARY: "Organized summary of case facts and events",
        DraftType.NOTICE_SUMMARY: "Summary of legal notices with key dates and actions",
        DraftType.DOCUMENT_CHECKLIST: "Checklist of required or mentioned documents",
        DraftType.INTERNAL_MEMO: "Professional internal memorandum format",
    }
    return descriptions.get(draft_type, "General document draft")
