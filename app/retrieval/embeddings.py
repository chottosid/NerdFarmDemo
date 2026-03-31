"""Embedding client using OpenRouter API."""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
MAX_BATCH_SIZE = 100  # Max texts per embedding API call


class EmbeddingClient:
    """Client for generating embeddings via OpenRouter."""

    def __init__(self):
        """Initialize embedding client with OpenRouter configuration."""
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=30.0,
        )
        self.model = settings.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts with batching and retries.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If all retries are exhausted
        """
        if not texts:
            return []

        # Batch texts to avoid exceeding API limits
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts with retry logic.

        Args:
            texts: Batch of text strings to embed

        Returns:
            List of embedding vectors
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [item.embedding for item in response.data]

            except RateLimitError as e:
                last_error = e
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Embedding rate limited on attempt %d/%d. Retrying in %.1fs...",
                    attempt, MAX_RETRIES, wait_time,
                )
                await asyncio.sleep(wait_time)

            except (APITimeoutError, APIError) as e:
                last_error = e
                logger.error(
                    "Embedding API error on attempt %d/%d: %s",
                    attempt, MAX_RETRIES, str(e),
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)

        raise RuntimeError(
            f"Embedding generation failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

