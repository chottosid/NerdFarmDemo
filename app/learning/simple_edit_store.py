"""Enhanced edit storage using ChromaDB with quality filtering and weighted retrieval.

Features:
- Quality Gate: Filter trivial edits before storage
- Deduplication: Detect and merge similar edits
- Feedback Tracking: Record when edits help/hurt
- Weighted Retrieval: Score by recency, quality, and validation
"""

import difflib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.retrieval.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class EditExample:
    """A single edit example for few-shot learning."""

    edit_id: str
    before: str
    after: str
    reason: Optional[str] = None
    draft_type: str = "global"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Quality and tracking fields
    quality_score: float = 0.5
    times_seen: int = 1
    times_retrieved: int = 0
    times_accepted: int = 0
    times_rejected: int = 0


# Alias for backwards compatibility
OperatorEdit = EditExample


class QualityGate:
    """Filter edits to only store meaningful ones.

    Rejects:
    - Too short (< 20 chars)
    - Whitespace-only changes
    - Punctuation-only changes
    - Minor changes (< 15% of text)
    """

    MIN_LENGTH = 20
    MIN_CHANGE_RATIO = 0.15

    def is_meaningful(self, before: str, after: str) -> tuple[bool, float]:
        """Check if edit is worth storing.

        Args:
            before: Original text
            after: Edited text

        Returns:
            Tuple of (should_store, quality_score)
        """
        # Skip too short texts
        if len(before) < self.MIN_LENGTH or len(after) < self.MIN_LENGTH:
            logger.debug("Rejected: too short")
            return False, 0.0

        # Skip whitespace-only changes
        if before.strip() == after.strip():
            logger.debug("Rejected: whitespace-only")
            return False, 0.0

        # Skip punctuation-only changes
        if self._only_punctuation_changed(before, after):
            logger.debug("Rejected: punctuation-only")
            return False, 0.0

        # Calculate change ratio
        change_ratio = self._calculate_change_ratio(before, after)
        if change_ratio < self.MIN_CHANGE_RATIO:
            logger.debug(f"Rejected: change ratio {change_ratio:.2%} < {self.MIN_CHANGE_RATIO:.0%}")
            return False, 0.0

        # Compute quality score (0-1, higher is better)
        quality_score = min(change_ratio * 2, 1.0)
        logger.info(f"Accepted edit: quality={quality_score:.2f}, change={change_ratio:.2%}")
        return True, quality_score

    def _only_punctuation_changed(self, before: str, after: str) -> bool:
        """Check if only punctuation changed."""
        # Remove all punctuation and compare
        no_punct_before = re.sub(r'[^\w\s]', '', before)
        no_punct_after = re.sub(r'[^\w\s]', '', after)
        return no_punct_before == no_punct_after

    def _calculate_change_ratio(self, before: str, after: str) -> float:
        """Calculate what fraction of text changed.

        Uses difflib.SequenceMatcher for accurate comparison.
        """
        matcher = difflib.SequenceMatcher(None, before, after)
        similarity = matcher.ratio()  # 0-1, higher = more similar
        return 1 - similarity  # Convert to change ratio


class SimpleEditStore:
    """Store and retrieve edit examples using ChromaDB.

    Enhanced features:
    - Quality Gate: Filter trivial edits
    - Deduplication: Merge similar edits
    - Feedback Tracking: Record edit effectiveness
    - Weighted Retrieval: Score by recency, quality, validation
    """

    COLLECTION_NAME = "edit_examples"
    DEDUP_SIMILARITY_THRESHOLD = 0.92

    def __init__(self):
        """Initialize the simple edit store."""
        settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_client = EmbeddingClient()
        self.quality_gate = QualityGate()

    async def save_edit(
        self,
        before: str,
        after: str,
        reason: Optional[str] = None,
        draft_type: str = "global",
    ) -> Optional[EditExample]:
        """Save an edit example with quality filtering and deduplication.

        Args:
            before: Original text
            after: Edited/corrected text
            reason: Optional reason for the edit
            draft_type: Type of draft this applies to

        Returns:
            The saved EditExample, or None if rejected by quality gate
        """
        # Quality gate check
        is_valid, quality_score = self.quality_gate.is_meaningful(before, after)
        if not is_valid:
            return None

        # Check for duplicates
        duplicate_id = await self._find_near_duplicate(before, after, draft_type)
        if duplicate_id:
            # Update existing edit's count instead of creating new
            self._increment_field(duplicate_id, "times_seen")
            logger.info(f"Deduplicated: incremented times_seen for {duplicate_id[:8]}")
            return await self._get_by_id(duplicate_id)

        # Create new edit
        edit_id = str(uuid.uuid4())
        embedding_text = f"Before: {before}\nAfter: {after}"
        embedding = await self.embedding_client.embed_single(embedding_text)

        now = datetime.now(timezone.utc)
        self.collection.add(
            ids=[edit_id],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[{
                "before": before,
                "after": after,
                "reason": reason or "",
                "draft_type": draft_type,
                "timestamp": now.isoformat(),
                "quality_score": quality_score,
                "times_seen": 1,
                "times_retrieved": 0,
                "times_accepted": 0,
                "times_rejected": 0,
                "last_seen": now.isoformat(),
                "last_retrieved": "",
            }],
        )

        logger.info(f"Saved edit: {edit_id[:8]} (quality={quality_score:.2f})")

        return EditExample(
            edit_id=edit_id,
            before=before,
            after=after,
            reason=reason,
            draft_type=draft_type,
            quality_score=quality_score,
        )

    async def _find_near_duplicate(
        self,
        before: str,
        after: str,
        draft_type: str,
    ) -> Optional[str]:
        """Find if a very similar edit already exists.

        Args:
            before: Original text
            after: Edited text
            draft_type: Draft type filter

        Returns:
            Edit ID if duplicate found, None otherwise
        """
        if self.collection.count() == 0:
            return None

        # Embed the combined text
        query_text = f"Before: {before}\nAfter: {after}"
        embedding = await self.embedding_client.embed_single(query_text)

        # Search for very similar edits
        where_filter = {"draft_type": {"$in": ["global", draft_type]}}
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # similarity = 1 - distance/2 for normalized vectors
            similarity = 1 - distance / 2
            if similarity >= self.DEDUP_SIMILARITY_THRESHOLD:
                return results["ids"][0][0]

        return None

    def _increment_field(self, edit_id: str, field_name: str) -> None:
        """Increment a counter field in metadata."""
        results = self.collection.get(
            ids=[edit_id],
            include=["metadatas"],
        )
        if results["metadatas"]:
            meta = results["metadatas"][0]
            current = int(meta.get(field_name, 0))
            meta[field_name] = current + 1

            if field_name == "times_seen":
                meta["last_seen"] = datetime.now(timezone.utc).isoformat()
            elif field_name == "times_retrieved":
                meta["last_retrieved"] = datetime.now(timezone.utc).isoformat()

            self.collection.update(
                ids=[edit_id],
                metadatas=[meta],
            )

    async def _get_by_id(self, edit_id: str) -> Optional[EditExample]:
        """Get an edit by ID."""
        results = self.collection.get(
            ids=[edit_id],
            include=["metadatas"],
        )
        if results["metadatas"]:
            meta = results["metadatas"][0]
            return EditExample(
                edit_id=edit_id,
                before=meta["before"],
                after=meta["after"],
                reason=meta.get("reason"),
                draft_type=meta.get("draft_type", "global"),
                quality_score=meta.get("quality_score", 0.5),
                times_seen=meta.get("times_seen", 1),
                times_retrieved=meta.get("times_retrieved", 0),
                times_accepted=meta.get("times_accepted", 0),
                times_rejected=meta.get("times_rejected", 0),
            )
        return None

    async def get_similar_edits(
        self,
        query: str,
        draft_type: Optional[str] = None,
        k: int = 3,
    ) -> list[dict]:
        """Find similar past edits for few-shot prompting.

        Uses weighted scoring: semantic (50%) + quality/recency (30%) + validation (20%)

        Args:
            query: Query to find similar edits for
            draft_type: Optional filter by draft type
            k: Number of similar edits to return

        Returns:
            List of edit examples with before, after, reason, scores
        """
        if self.collection.count() == 0:
            return []

        # Generate embedding for query
        query_embedding = await self.embedding_client.embed_single(query)

        # Build filter if draft_type specified
        where_filter = None
        if draft_type:
            where_filter = {"draft_type": {"$in": ["global", draft_type]}}

        # Get more candidates than needed for reranking
        n_candidates = min(k * 4, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        # Apply weighted scoring
        now = datetime.now(timezone.utc)
        weighted_edits = []

        for i, meta in enumerate(results["metadatas"][0]):
            distance = results["distances"][0][i]
            # Convert distance to similarity (0-1)
            semantic_score = max(0, 1 - distance / 2)

            # Skip very dissimilar edits
            if semantic_score < 0.5:
                continue

            # Recency boost (affects quality component)
            timestamp_str = meta.get("timestamp", now.isoformat())
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (now - timestamp).days
            except:
                age_days = 365  # Default to old

            if age_days < 7:
                recency_multiplier = 1.2  # 20% boost
            elif age_days < 30:
                recency_multiplier = 1.0
            else:
                recency_multiplier = 0.8  # 20% decay

            # Quality score
            quality_score = float(meta.get("quality_score", 0.5))

            # Usage validation (more times seen = more validated)
            times_seen = int(meta.get("times_seen", 1))
            times_accepted = int(meta.get("times_accepted", 0))
            times_rejected = int(meta.get("times_rejected", 0))

            # Acceptance rate (if any feedback received)
            if times_accepted + times_rejected > 0:
                acceptance_rate = times_accepted / (times_accepted + times_rejected)
            else:
                acceptance_rate = 0.5  # Neutral if no feedback

            # Validation multiplier: seen multiple times + good acceptance
            validation_multiplier = min(1.0 + (times_seen * 0.05) + (acceptance_rate * 0.2), 1.5)

            # Combined weighted score
            weighted_score = (
                semantic_score * 0.5 +  # 50% semantic similarity
                quality_score * 0.3 * recency_multiplier +  # 30% quality with recency
                acceptance_rate * 0.2 * validation_multiplier  # 20% validation
            )

            weighted_edits.append({
                "edit_id": results["ids"][0][i],
                "before": meta["before"],
                "after": meta["after"],
                "reason": meta.get("reason", ""),
                "semantic_score": semantic_score,
                "quality_score": quality_score,
                "weighted_score": weighted_score,
                "times_seen": times_seen,
                "times_accepted": times_accepted,
            })

        # Sort by weighted score and return top k
        weighted_edits.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Record retrieval for analytics (fire and forget)
        top_edit_ids = [e["edit_id"] for e in weighted_edits[:k]]
        for edit_id in top_edit_ids:
            try:
                self._increment_field(edit_id, "times_retrieved")
            except Exception:
                pass  # Don't fail on tracking errors

        # Return clean format for prompt generation
        return [
            {
                "before": e["before"],
                "after": e["after"],
                "reason": e["reason"],
                "score": e["weighted_score"],
            }
            for e in weighted_edits[:k]
        ]

    def record_feedback(self, edit_ids: list[str], was_accepted: bool) -> None:
        """Record whether edits helped produce an accepted draft.

        Args:
            edit_ids: List of edit IDs that were used
            was_accepted: Whether the user accepted the draft
        """
        field = "times_accepted" if was_accepted else "times_rejected"
        for edit_id in edit_ids:
            try:
                self._increment_field(edit_id, field)
                logger.info(f"Recorded feedback for {edit_id[:8]}: {field}")
            except Exception as e:
                logger.warning(f"Failed to record feedback: {e}")

    def get_recent_edits(self, limit: int = 50) -> list[EditExample]:
        """Get recent edit examples.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent EditExample objects
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            limit=limit,
            include=["metadatas"],
        )

        edits = []
        if results["metadatas"]:
            for i, meta in enumerate(results["metadatas"]):
                edits.append(EditExample(
                    edit_id=results["ids"][i],
                    before=meta["before"],
                    after=meta["after"],
                    reason=meta.get("reason"),
                    draft_type=meta.get("draft_type", "global"),
                    quality_score=meta.get("quality_score", 0.5),
                    times_seen=meta.get("times_seen", 1),
                    times_retrieved=meta.get("times_retrieved", 0),
                    times_accepted=meta.get("times_accepted", 0),
                    times_rejected=meta.get("times_rejected", 0),
                ))

        return edits

    def get_effectiveness_report(self) -> dict:
        """Generate a report on edit effectiveness.

        Returns:
            Dict with metrics on how well edits are working
        """
        if self.collection.count() == 0:
            return {"total_edits": 0, "message": "No edits stored yet"}

        results = self.collection.get(
            include=["metadatas"],
        )

        total = len(results["metadatas"])
        total_retrieved = 0
        total_accepted = 0
        total_rejected = 0
        quality_scores = []

        for meta in results["metadatas"]:
            total_retrieved += int(meta.get("times_retrieved", 0))
            total_accepted += int(meta.get("times_accepted", 0))
            total_rejected += int(meta.get("times_rejected", 0))
            quality_scores.append(float(meta.get("quality_score", 0.5)))

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            "total_edits": total,
            "total_retrieved": total_retrieved,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "acceptance_rate": total_accepted / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else None,
            "average_quality_score": round(avg_quality, 3),
            "edits_with_feedback": sum(1 for m in results["metadatas"] if int(m.get("times_accepted", 0)) + int(m.get("times_rejected", 0)) > 0),
        }

    def count(self) -> int:
        """Get total number of stored edits."""
        return self.collection.count()


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Format edit examples for injection into a prompt.

    Args:
        examples: List of edit examples with before, after, reason

    Returns:
        Formatted string for prompt
    """
    if not examples:
        return ""

    lines = [
        "Learn from these past improvements:",
        "",
    ]

    for i, example in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        # Truncate long examples
        before = example["before"]
        after = example["after"]
        if len(before) > 200:
            before = before[:200] + "..."
        if len(after) > 200:
            after = after[:200] + "..."

        lines.append(f'  Before: "{before}"')
        lines.append(f'  After:  "{after}"')
        if example.get("reason"):
            lines.append(f'  Reason: {example["reason"]}')
        lines.append("")

    lines.append("Apply similar improvements in your response.")

    return "\n".join(lines)
