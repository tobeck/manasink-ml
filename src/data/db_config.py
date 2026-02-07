"""
Database configuration for Manasink ML.

Supports both SQLite (development) and PostgreSQL (production) via
environment variables.

Configuration:
    DATABASE_URL: Full database URL (overrides other settings)
        - SQLite: sqlite:///data/cards.db
        - PostgreSQL: postgresql://user:pass@host:5432/manasink

    Or use individual settings:
        DB_TYPE: 'sqlite' or 'postgresql'
        DB_HOST: Database host (PostgreSQL only)
        DB_PORT: Database port (default: 5432)
        DB_NAME: Database name
        DB_USER: Database user (PostgreSQL only)
        DB_PASSWORD: Database password (PostgreSQL only)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    url: str
    is_sqlite: bool
    echo: bool = False  # Log SQL queries

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        # Check for full URL first
        database_url = os.environ.get("DATABASE_URL")

        if database_url:
            is_sqlite = database_url.startswith("sqlite")
            return cls(url=database_url, is_sqlite=is_sqlite)

        # Build URL from components
        db_type = os.environ.get("DB_TYPE", "sqlite").lower()

        if db_type == "sqlite":
            db_path = os.environ.get("DB_PATH", "data/cards.db")
            # Ensure parent directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path}"
            return cls(url=url, is_sqlite=True)

        elif db_type in ("postgresql", "postgres"):
            host = os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            name = os.environ.get("DB_NAME", "manasink")
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "")

            url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
            return cls(url=url, is_sqlite=False)

        else:
            raise ValueError(f"Unsupported DB_TYPE: {db_type}")

    @classmethod
    def sqlite(cls, path: str = "data/cards.db") -> "DatabaseConfig":
        """Create SQLite config."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cls(url=f"sqlite:///{path}", is_sqlite=True)

    @classmethod
    def postgresql(
        cls,
        host: str = "localhost",
        port: int = 5432,
        name: str = "manasink",
        user: str = "postgres",
        password: str = "",
    ) -> "DatabaseConfig":
        """Create PostgreSQL config."""
        url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        return cls(url=url, is_sqlite=False)


class DatabaseManager:
    """
    Manages database connections and sessions.

    Usage:
        db = DatabaseManager()
        with db.session() as session:
            cards = session.query(CardModel).all()

        # Or get engine directly for bulk operations
        engine = db.engine
    """

    _instance: Optional["DatabaseManager"] = None
    _config: DatabaseConfig | None = None

    def __new__(cls, config: DatabaseConfig | None = None):
        """Singleton pattern - one database manager per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: DatabaseConfig | None = None):
        if self._initialized and config is None:
            return

        self._config = config or DatabaseConfig.from_env()
        self._setup_engine()
        self._initialized = True

    def _setup_engine(self):
        """Create SQLAlchemy engine with appropriate settings."""
        if self._config.is_sqlite:
            # SQLite-specific settings
            self._engine = create_engine(
                self._config.url,
                echo=self._config.echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,  # Single connection for SQLite
            )
        else:
            # PostgreSQL settings
            self._engine = create_engine(
                self._config.url,
                echo=self._config.echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
            )

        self._SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine,
        )

    @property
    def engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine

    @property
    def config(self) -> DatabaseConfig:
        """Get the database configuration."""
        return self._config

    def session(self) -> Session:
        """
        Get a new database session.

        Usage:
            with db.session() as session:
                # Use session
                pass
        """
        return self._SessionLocal()

    def create_tables(self):
        """Create all tables defined in models."""
        from .db_models import Base

        Base.metadata.create_all(bind=self._engine)

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        from .db_models import Base

        Base.metadata.drop_all(bind=self._engine)

    def reset(self):
        """Reset the singleton instance (for testing)."""
        DatabaseManager._instance = None
        DatabaseManager._config = None


# Convenience function for getting a session
def get_db_session() -> Session:
    """Get a database session from the default manager."""
    return DatabaseManager().session()


def get_engine():
    """Get the database engine from the default manager."""
    return DatabaseManager().engine
