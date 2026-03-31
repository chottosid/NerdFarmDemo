"""Learning module for edit storage and few-shot retrieval.

Simplified version: Just store edit examples, retrieve similar ones.
No LLM calls on save, no rule extraction.
"""

from .simple_edit_store import (
    EditExample,
    OperatorEdit,  # Alias for backwards compatibility
    SimpleEditStore,
    format_examples_for_prompt,
)

__all__ = [
    "SimpleEditStore",
    "EditExample",
    "OperatorEdit",
    "format_examples_for_prompt",
]
