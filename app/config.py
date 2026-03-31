"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"
    llm_model: str = "openai/gpt-4o"

    # Storage paths
    chroma_persist_dir: str = "./chroma_data"
    sqlite_db_path: str = "./edits.db"

    # Upload settings
    upload_dir: str = "./uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    max_content_length: int = 100000  # Max chars to extract from document

    # Database configuration
    database_backend: str = "sqlite"  # sqlite (future: postgresql)

    # Search configuration
    use_hybrid_search: bool = True
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_alpha: float = 0.5  # weight for semantic vs keyword (0.0-1.0)

    # Vision processing configuration
    vision_model: str = "openai/gpt-4o-mini"  # Cheaper than gpt-4o, still accurate
    vision_max_tokens: int = 4096
    use_vision_for_images: bool = True  # Use vision model for images/scanned PDFs


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
