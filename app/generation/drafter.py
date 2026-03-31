"""Draft generation engine with grounding validation.

Features:
- Insufficient evidence flagging when retrieval quality is low
- Retrieved chunks visibility for transparency
- Confidence scoring based on retrieval quality
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.retrieval import Retriever, RetrievedChunk

from .llm import LLMClient
from .prompts import DraftType, build_draft_prompt

logger = logging.getLogger(__name__)

# Thresholds for grounding validation
MIN_RETRIEVAL_CONFIDENCE = 0.3
MIN_RELEVANT_CHUNKS = 1
MIN_AVG_SIMILARITY = 0.4


@dataclass
class Citation:
    """A citation to source material."""

    text: str
    source_doc: str
    page: int
    chunk_id: str


@dataclass
class RetrievedChunkInfo:
    """Info about a retrieved chunk for API response."""

    text: str
    source_doc: str
    page: int
    chunk_id: str
    score: float


@dataclass
class DraftOutput:
    """Output from draft generation."""

    draft_id: str
    content: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    draft_type: str = ""
    query: str = ""
    # New fields for grounding transparency
    retrieved_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    is_grounded: bool = True  # False if insufficient evidence
    grounding_warning: str = ""


class DraftGenerator:
    """Generate grounded drafts using retrieved evidence.

    Features:
    - Validates retrieval quality before generation
    - Flags when insufficient evidence exists
    - Tracks retrieved chunks for transparency
    """

    def __init__(self):
        """Initialize draft generator."""
        self.llm = LLMClient()
        self.retriever = Retriever()

    async def generate(
        self,
        query: str,
        draft_type: DraftType,
        document_ids: Optional[list[str]] = None,
        few_shot_examples: Optional[str] = None,
    ) -> DraftOutput:
        """Generate a draft document with grounding validation.

        Args:
            query: User's query/request
            draft_type: Type of draft to generate
            document_ids: Optional filter by specific documents
            few_shot_examples: Optional formatted examples from past edits

        Returns:
            DraftOutput with content, citations, and grounding info
        """
        draft_id = str(uuid.uuid4())

        # Retrieve relevant chunks
        chunks = await self.retriever.retrieve(
            query=query,
            k=5,
            doc_ids=document_ids,
        )

        # Validate grounding - check if we have sufficient evidence
        grounding_check = self._validate_grounding(chunks)

        if not grounding_check["is_grounded"]:
            # Return early with insufficient evidence warning
            logger.warning(
                f"Insufficient evidence for query: {grounding_check['reason']}"
            )
            # Calculate actual confidence from what we retrieved
            actual_confidence = self._calculate_confidence(chunks) if chunks else 0.0
            # Build retrieved chunks info so user can see what was found
            retrieved_chunks_info = self._build_retrieved_chunks_info(chunks) if chunks else []

            return DraftOutput(
                draft_id=draft_id,
                content=self._format_insufficient_evidence_message(
                    query, grounding_check["reason"]
                ),
                citations=[],
                confidence=actual_confidence,
                draft_type=draft_type.value,
                query=query,
                retrieved_chunks=retrieved_chunks_info,
                is_grounded=False,
                grounding_warning=grounding_check["reason"],
            )

        # Format context for prompt
        context = self.retriever.format_chunks_for_prompt(chunks)

        # Build prompt
        system_prompt, user_prompt = build_draft_prompt(
            query=query,
            context=context,
            draft_type=draft_type,
            few_shot_examples=few_shot_examples,
        )

        # Generate draft
        content = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
        )

        # Extract citations from chunks
        citations = self._build_citations(chunks)

        # Build retrieved chunks info for transparency
        retrieved_chunks = self._build_retrieved_chunks_info(chunks)

        # Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(chunks)

        return DraftOutput(
            draft_id=draft_id,
            content=content,
            citations=citations,
            confidence=confidence,
            draft_type=draft_type.value,
            query=query,
            retrieved_chunks=retrieved_chunks,
            is_grounded=True,
            grounding_warning="",
        )

    def _validate_grounding(self, chunks: list[RetrievedChunk]) -> dict:
        """Validate that we have sufficient evidence for generation.

        Args:
            chunks: Retrieved chunks

        Returns:
            Dict with is_grounded bool and reason if not
        """
        if not chunks:
            return {
                "is_grounded": False,
                "reason": "No relevant documents found in the knowledge base",
            }

        if len(chunks) < MIN_RELEVANT_CHUNKS:
            return {
                "is_grounded": False,
                "reason": f"Only {len(chunks)} relevant chunk(s) found, need at least {MIN_RELEVANT_CHUNKS}",
            }

        # Check average similarity
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
        if avg_similarity < MIN_AVG_SIMILARITY:
            return {
                "is_grounded": False,
                "reason": f"Retrieved content has low relevance (score: {avg_similarity:.2f}, need: {MIN_AVG_SIMILARITY:.2f})",
            }

        # Check individual chunk scores
        high_quality_chunks = [c for c in chunks if c.similarity_score >= MIN_RETRIEVAL_CONFIDENCE]
        if not high_quality_chunks:
            return {
                "is_grounded": False,
                "reason": "No chunks meet minimum relevance threshold",
            }

        return {"is_grounded": True, "reason": ""}

    def _format_insufficient_evidence_message(
        self, query: str, reason: str
    ) -> str:
        """Format the insufficient evidence message.

        Args:
            query: Original query
            reason: Why evidence is insufficient

        Returns:
            Formatted warning message
        """
        return f"""⚠️ **Insufficient Evidence**

The provided documents do not contain enough relevant information to generate a meaningful response to your query.

**Reason:** {reason}

**Suggestions:**
- Upload additional documents related to this topic
- Try rephrasing your query with different keywords
- Check if the relevant documents have been processed

**Query:** {query}

Please add more relevant source materials and try again."""

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Build citation list from retrieved chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of Citation objects
        """
        citations = []
        seen = set()

        for chunk in chunks:
            # Avoid duplicate citations
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            # Truncate citation text if needed, with ellipsis only when actually truncated
            citation_text = chunk.content[:200]
            if len(chunk.content) > 200:
                citation_text += "..."

            citations.append(Citation(
                text=citation_text,
                source_doc=chunk.filename,
                page=chunk.page_num,
                chunk_id=chunk.chunk_id,
            ))

        return citations

    def _build_retrieved_chunks_info(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunkInfo]:
        """Build retrieved chunks info for API response.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of RetrievedChunkInfo objects
        """
        return [
            RetrievedChunkInfo(
                text=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                source_doc=chunk.filename,
                page=chunk.page_num,
                chunk_id=chunk.chunk_id,
                score=chunk.similarity_score,
            )
            for chunk in chunks
        ]

    def _calculate_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """Calculate confidence score based on retrieval quality.

        Args:
            chunks: Retrieved chunks

        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0

        # Average similarity score of top chunks
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)

        # Adjust based on number of relevant chunks found
        coverage_bonus = min(len(chunks) / 5.0, 1.0) * 0.1

        return min(avg_similarity + coverage_bonus, 1.0)

    def format_draft_with_citations(self, draft: DraftOutput) -> str:
        """Format draft with inline citations for display.

        Args:
            draft: Draft output

        Returns:
            Formatted string with citations section
        """
        parts = [draft.content]

        if not draft.is_grounded:
            return draft.content  # Already has warning formatted

        parts.append("\n---\n")
        parts.append("## Sources\n")

        for i, citation in enumerate(draft.citations, 1):
            parts.append(
                f"[{i}] {citation.source_doc}, Page {citation.page}\n"
            )

        return "\n".join(parts)
