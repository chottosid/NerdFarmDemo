"""Generation module for draft creation."""

from .drafter import Citation, DraftGenerator, DraftOutput
from .llm import LLMClient
from .prompts import DraftType, build_draft_prompt, get_system_prompt

__all__ = [
    "LLMClient",
    "DraftType",
    "DraftGenerator",
    "DraftOutput",
    "Citation",
    "get_system_prompt",
    "build_draft_prompt",
]
