"""Prompt templates for draft generation."""

from enum import Enum


class DraftType(Enum):
    """Types of drafts that can be generated."""

    TITLE_REVIEW_SUMMARY = "title_review_summary"
    CASE_FACT_SUMMARY = "case_fact_summary"
    NOTICE_SUMMARY = "notice_summary"
    DOCUMENT_CHECKLIST = "document_checklist"
    INTERNAL_MEMO = "internal_memo"


# System prompts for each draft type
SYSTEM_PROMPTS = {
    DraftType.TITLE_REVIEW_SUMMARY: """You are a legal assistant helping to create title review summaries.
Your task is to analyze the provided documents and create a clear, accurate summary of title review findings.
Always cite specific sources from the provided context using [Source: filename, Page X] format.
If information is unclear or missing, note it rather than making assumptions.""",

    DraftType.CASE_FACT_SUMMARY: """You are a legal assistant creating case fact summaries.
Extract and organize the key facts from the provided documents into a coherent summary.
Include dates, parties involved, and key events with citations.
Use [Source: filename, Page X] format for all factual claims.""",

    DraftType.NOTICE_SUMMARY: """You are a legal assistant summarizing legal notices.
Extract the key information: parties involved, dates, required actions, and deadlines.
Format the summary clearly with citations using [Source: filename, Page X] format.""",

    DraftType.DOCUMENT_CHECKLIST: """You are a legal assistant creating document checklists.
Review the provided documents and create a checklist of items mentioned or required.
Include document names, deadlines, and responsible parties where available.
Use [Source: filename, Page X] format for each item.""",

    DraftType.INTERNAL_MEMO: """You are a legal assistant drafting internal memos.
Create a professional memo based on the provided documents with:
- Clear subject line
- Background summary
- Key findings with citations
- Recommended next steps
Use [Source: filename, Page X] format for all referenced information.""",
}


def get_system_prompt(draft_type: DraftType) -> str:
    """Get the system prompt for a draft type.

    Args:
        draft_type: Type of draft to generate

    Returns:
        System prompt string
    """
    return SYSTEM_PROMPTS.get(
        draft_type,
        SYSTEM_PROMPTS[DraftType.INTERNAL_MEMO],
    )


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format few-shot examples for prompt injection.

    Args:
        examples: List of edit examples with 'before', 'after', 'reason' keys

    Returns:
        Formatted examples string
    """
    if not examples:
        return ""

    formatted = ["When generating this draft, apply these learned improvements from similar past edits:\n"]

    for i, example in enumerate(examples, 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"- Original: \"{example.get('before', '')}\"")
        formatted.append(f"- Improved: \"{example.get('after', '')}\"")
        formatted.append(f"- Reason: {example.get('reason', 'No reason provided')}\n")

    formatted.append("Now generate the draft following these improvement patterns.")

    return "\n".join(formatted)


def format_rules(rules: list) -> str:
    """Format structured abstract rules for prompt injection.
    
    Args:
        rules: List of Rule objects.
        
    Returns:
        Formatted rules string.
    """
    if not rules:
        return ""

    formatted = ["You must strictly follow these constraints learned from past operator edits:\n"]
    for i, rule in enumerate(rules, 1):
        formatted.append(f"Constraint {i}: WHEN {rule.when} THEN {rule.then}")

    return "\n".join(formatted)


def build_draft_prompt(
    query: str,
    context: str,
    draft_type: DraftType,
    few_shot_examples: str | None = None,
) -> tuple[str, str]:
    """Build the complete prompt for draft generation.

    Args:
        query: User's query/request
        context: Retrieved context chunks
        draft_type: Type of draft to generate
        few_shot_examples: Optional formatted examples from past edits

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_system_prompt(draft_type)

    # Build user prompt
    prompt_parts = []

    if few_shot_examples:
        prompt_parts.append(few_shot_examples)
        prompt_parts.append("")

    prompt_parts.append("RETRIEVED DOCUMENT CONTEXT:")
    prompt_parts.append(context)
    prompt_parts.append("")
    prompt_parts.append("USER REQUEST:")
    prompt_parts.append(query)

    user_prompt = "\n".join(prompt_parts)

    return system_prompt, user_prompt
