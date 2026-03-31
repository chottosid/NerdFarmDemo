"""LLM client using OpenRouter API."""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from app.config import get_settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class LLMClient:
    """Client for LLM interactions via OpenRouter."""

    def __init__(self):
        """Initialize LLM client with OpenRouter configuration."""
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=30.0,
        )
        self.model = settings.llm_model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text completion with retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            RuntimeError: If all retries are exhausted
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                last_error = e
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Rate limited on attempt %d/%d. Retrying in %.1fs...",
                    attempt, MAX_RETRIES, wait_time,
                )
                await asyncio.sleep(wait_time)

            except APITimeoutError as e:
                last_error = e
                logger.warning(
                    "LLM request timed out on attempt %d/%d.",
                    attempt, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0)

            except APIError as e:
                last_error = e
                logger.error(
                    "LLM API error on attempt %d/%d: %s",
                    attempt, MAX_RETRIES, str(e),
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)

        raise RuntimeError(
            f"LLM generation failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str,
        few_shot_examples: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text with retrieved context.

        Args:
            query: User query
            context: Retrieved context chunks
            system_prompt: System prompt
            few_shot_examples: Optional few-shot examples
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        # Build prompt with context
        prompt_parts = []

        if few_shot_examples:
            prompt_parts.append(f"Learned improvements to apply:\n{few_shot_examples}\n")

        prompt_parts.append(f"Context from documents:\n{context}\n")
        prompt_parts.append(f"Query: {query}")

        prompt = "\n".join(prompt_parts)

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

