"""Factory for creating repository instances.

This module provides a factory pattern for creating repository instances
based on configuration, allowing easy switching between database backends.
"""

import logging
from typing import Optional

from app.config import get_settings

from .base import DatabaseConfig, EditRepository

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating repository instances based on configuration."""

    @staticmethod
    def create_edit_repository(
        config: Optional[DatabaseConfig] = None,
    ) -> EditRepository:
        """Create an edit repository based on configuration.

        Args:
            config: Optional database configuration. Uses settings if not provided.

        Returns:
            EditRepository instance

        Raises:
            ValueError: If backend is not supported
        """
        if config is None:
            settings = get_settings()
            config = DatabaseConfig(
                backend=getattr(settings, "database_backend", "sqlite"),
                path=settings.sqlite_db_path,
            )

        if config.backend == "sqlite":
            # Import here to avoid circular imports
            from app.learning import SimpleEditStore

            logger.info("Creating ChromaDB edit repository")
            return SimpleEditStore()

        # Future: elif config.backend == "postgresql":
        #     from .postgres_edit_repo import PostgresEditRepository
        #     logger.info("Creating PostgreSQL edit repository")
        #     return PostgresEditRepository(config)

        raise ValueError(f"Unsupported database backend: {config.backend}")

    @staticmethod
    def get_available_backends() -> list[str]:
        """Get list of available database backends.

        Returns:
            List of supported backend names
        """
        return ["sqlite"]  # Future: ["sqlite", "postgresql"]
