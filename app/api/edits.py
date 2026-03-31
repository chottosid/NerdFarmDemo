"""API endpoints for edit submission and learning.

Enhanced features:
- Quality gate filters trivial edits
- Feedback tracking for edit effectiveness
- Effectiveness reporting
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.learning import SimpleEditStore, EditExample

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edits", tags=["edits"])

# Shared store instance
_edit_store: SimpleEditStore | None = None


def get_edit_store() -> SimpleEditStore:
    """Get or create edit store instance."""
    global _edit_store
    if _edit_store is None:
        _edit_store = SimpleEditStore()
    return _edit_store


class SubmitEditRequest(BaseModel):
    """Request model for submitting an edit."""

    draft_id: str = Field(..., description="ID of the draft being edited")
    original_text: str = Field(..., description="Original text from the draft")
    edited_text: str = Field(..., description="Operator's edited version")
    edit_reason: str | None = Field(
        default=None,
        description="Optional reason for the edit",
    )
    draft_type: str = Field(
        default="global",
        description="Type of draft this applies to",
    )


class SubmitEditResponse(BaseModel):
    """Response model for edit submission."""

    edit_id: str | None = None
    learned: bool
    quality_rejected: bool = False
    duplicate_of: str | None = None
    message: str = ""


class FeedbackRequest(BaseModel):
    """Request model for recording feedback on generated drafts."""

    draft_id: str = Field(..., description="ID of the generated draft")
    was_accepted: bool = Field(..., description="Whether user accepted the draft")
    edit_ids_used: list[str] = Field(
        default_factory=list,
        description="Edit IDs that were used in this generation",
    )


class EditHistoryItem(BaseModel):
    """Model for edit history item."""

    edit_id: str
    before: str
    after: str
    reason: str | None
    draft_type: str
    quality_score: float = 0.5
    times_seen: int = 1
    times_retrieved: int = 0
    times_accepted: int = 0


@router.post("/submit", response_model=SubmitEditResponse)
async def submit_edit(request: SubmitEditRequest) -> SubmitEditResponse:
    """Submit an operator edit for learning.

    Enhanced features:
    - Quality gate filters trivial edits
    - Deduplication merges similar edits

    Args:
        request: Edit submission request

    Returns:
        Edit ID confirming storage, or rejection reason
    """
    store = get_edit_store()

    edit = await store.save_edit(
        before=request.original_text,
        after=request.edited_text,
        reason=request.edit_reason,
        draft_type=request.draft_type,
    )

    if edit is None:
        # Rejected by quality gate
        return SubmitEditResponse(
            edit_id=None,
            learned=False,
            quality_rejected=True,
            message="Edit rejected: too trivial (whitespace/punctuation only, or < 15% change)",
        )

    return SubmitEditResponse(
        edit_id=edit.edit_id,
        learned=True,
        quality_rejected=False,
        duplicate_of=None,
        message=f"Edit stored with quality score {edit.quality_score:.2f}",
    )


@router.post("/feedback")
async def record_feedback(request: FeedbackRequest) -> dict:
    """Record whether a generated draft was accepted or rejected.

    This helps track which edit examples are actually helpful.

    Args:
        request: Feedback with draft ID, acceptance status, and edit IDs used

    Returns:
        Confirmation of feedback recording
    """
    store = get_edit_store()

    if request.edit_ids_used:
        store.record_feedback(request.edit_ids_used, request.was_accepted)
        logger.info(
            f"Recorded feedback for draft {request.draft_id}: "
            f"{'accepted' if request.was_accepted else 'rejected'}, "
            f"{len(request.edit_ids_used)} edits tracked"
        )

    return {
        "status": "recorded",
        "draft_id": request.draft_id,
        "was_accepted": request.was_accepted,
        "edits_tracked": len(request.edit_ids_used),
    }


@router.get("/history")
async def get_edit_history(limit: int = 50) -> dict:
    """Get edit history.

    Args:
        limit: Maximum number of edits to return

    Returns:
        List of past edits with metadata
    """
    store = get_edit_store()
    history = store.get_recent_edits(limit=limit)

    return {
        "total": len(history),
        "edits": [
            EditHistoryItem(
                edit_id=e.edit_id,
                before=e.before[:200] + "..." if len(e.before) > 200 else e.before,
                after=e.after[:200] + "..." if len(e.after) > 200 else e.after,
                reason=e.reason,
                draft_type=e.draft_type,
                quality_score=e.quality_score,
                times_seen=e.times_seen,
                times_retrieved=e.times_retrieved,
                times_accepted=e.times_accepted,
            )
            for e in history
        ],
    }


@router.get("/similar")
async def get_similar_edits(query: str, k: int = 3, draft_type: str | None = None) -> dict:
    """Find similar past edits for a query.

    Uses weighted scoring: semantic (50%) + quality/recency (30%) + validation (20%)

    Args:
        query: Query to find similar edits for
        k: Number of similar edits to return
        draft_type: Optional filter by draft type

    Returns:
        List of similar edits with scores
    """
    store = get_edit_store()
    similar = await store.get_similar_edits(query, draft_type=draft_type, k=k)

    return {
        "query": query,
        "total": len(similar),
        "similar_edits": similar,
    }


@router.get("/count")
async def get_edit_count() -> dict:
    """Get total number of stored edits.

    Returns:
        Count of stored edits
    """
    store = get_edit_store()
    return {"count": store.count()}


@router.get("/effectiveness")
async def get_effectiveness_report() -> dict:
    """Get report on how well the learning system is working.

    Returns:
        Metrics on edit effectiveness
    """
    store = get_edit_store()
    return store.get_effectiveness_report()
