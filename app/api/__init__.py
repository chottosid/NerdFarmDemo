"""API module with FastAPI routers."""

from fastapi import APIRouter

from .documents import router as documents_router
from .drafts import router as drafts_router
from .edits import router as edits_router

__all__ = [
    "documents_router",
    "drafts_router",
    "edits_router",
]
