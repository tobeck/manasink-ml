"""
Database interface for card storage and retrieval.

This module provides a CardDatabase class that supports both SQLite and PostgreSQL
backends via SQLAlchemy. The interface remains the same regardless of backend.

Configuration:
    Set DATABASE_URL environment variable, or use individual DB_* variables.
    See db_config.py for full configuration options.

Example:
    db = CardDatabase()
    card = db.get_card("Sol Ring")
    commanders = db.get_commanders(colors="UG")
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func, or_, and_, text
from sqlalchemy.orm import Session

from src.game.card import Card, CardType, Color

from .db_config import DatabaseConfig, DatabaseManager, get_db_session
from .db_models import (
    Base,
    CardModel,
    SyncMetadata as SyncMetadataModel,
    CommanderModel,
    CommanderRecommendation,
    AverageDeckCard,
    CardSaltScore,
    EDHRecSyncMetadata,
    CardFeatureModel,
    CardCategory,
)

# Default database path (for backwards compatibility)
DEFAULT_DB_PATH = Path("data/cards.db")


@dataclass
class SyncMetadata:
    """Metadata about the last database sync."""

    last_updated: Optional[datetime]
    card_count: int
    scryfall_updated_at: Optional[str]


class CardDatabase:
    """
    Database interface for card queries.

    Supports both SQLite and PostgreSQL backends via SQLAlchemy.
    This class is read-heavy and expects writes to happen via the ingest module.

    Example:
        db = CardDatabase()
        card = db.get_card("Sol Ring")
        commanders = db.get_commanders(colors="UG")
        creatures = db.search(card_types=["creature"], max_cmc=3)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the database connection.

        Args:
            db_path: Path to SQLite database (legacy, for backwards compatibility)
            config: DatabaseConfig object (preferred for new code)
        """
        if config:
            self._manager = DatabaseManager(config)
        elif db_path:
            # Legacy: create SQLite config from path
            self._manager = DatabaseManager(DatabaseConfig.sqlite(str(db_path)))
        else:
            # Use environment configuration
            self._manager = DatabaseManager()

        self._session: Optional[Session] = None

    def _get_session(self) -> Session:
        """Get or create database session."""
        if self._session is None:
            self._session = self._manager.session()
        return self._session

    def close(self) -> None:
        """Close the database session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> "CardDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_card(self, name: str) -> Optional[Card]:
        """
        Get a card by exact name.

        Args:
            name: Exact card name (case-insensitive)

        Returns:
            Card object or None if not found
        """
        session = self._get_session()
        result = session.query(CardModel).filter(CardModel.name_lower == name.lower()).first()
        if result:
            return Card.from_scryfall(json.loads(result.scryfall_json))
        return None

    def get_card_by_id(self, scryfall_id: str) -> Optional[Card]:
        """Get a card by Scryfall ID."""
        session = self._get_session()
        result = session.query(CardModel).filter(CardModel.scryfall_id == scryfall_id).first()
        if result:
            return Card.from_scryfall(json.loads(result.scryfall_json))
        return None

    def search(
        self,
        *,
        name_contains: Optional[str] = None,
        card_types: Optional[list[str]] = None,
        colors: Optional[str] = None,
        color_identity: Optional[str] = None,
        min_cmc: Optional[int] = None,
        max_cmc: Optional[int] = None,
        is_commander: Optional[bool] = None,
        is_legal_commander: Optional[bool] = None,
        text_contains: Optional[str] = None,
        limit: int = 100,
    ) -> list[Card]:
        """
        Search for cards matching criteria.

        Args:
            name_contains: Substring match on card name
            card_types: List of types to match (creature, instant, etc.)
            colors: Exact color match (e.g., "UG" for blue-green)
            color_identity: Color identity subset (e.g., "UG" finds cards that fit in Simic)
            min_cmc: Minimum mana value
            max_cmc: Maximum mana value
            is_commander: Filter for legendary creatures
            is_legal_commander: Filter for commander-legal cards
            text_contains: Substring match on oracle text
            limit: Maximum results to return

        Returns:
            List of matching Card objects
        """
        session = self._get_session()
        query = session.query(CardModel)

        if name_contains:
            query = query.filter(CardModel.name_lower.contains(name_contains.lower()))

        if card_types:
            type_filters = [CardModel.type_line.ilike(f"%{ct.lower()}%") for ct in card_types]
            query = query.filter(or_(*type_filters))

        if colors:
            query = query.filter(CardModel.colors == _normalize_colors(colors))

        if color_identity:
            # Cards whose identity is a subset of the given colors
            identity_normalized = _normalize_colors(color_identity)
            # For each color in the card's identity, check if it's in the filter
            query = query.filter(
                _color_identity_fits_filter(CardModel.color_identity, identity_normalized)
            )

        if min_cmc is not None:
            query = query.filter(CardModel.cmc >= min_cmc)

        if max_cmc is not None:
            query = query.filter(CardModel.cmc <= max_cmc)

        if is_commander is not None:
            query = query.filter(CardModel.is_commander == is_commander)

        if is_legal_commander is not None:
            query = query.filter(CardModel.legal_commander == is_legal_commander)

        if text_contains:
            query = query.filter(CardModel.oracle_text_lower.contains(text_contains.lower()))

        results = query.order_by(CardModel.name_lower).limit(limit).all()
        return [Card.from_scryfall(json.loads(r.scryfall_json)) for r in results]

    def get_commanders(
        self,
        colors: Optional[str] = None,
        max_cmc: Optional[int] = None,
        limit: int = 100,
    ) -> list[Card]:
        """
        Get legendary creatures that can be commanders.

        Args:
            colors: Filter by color identity (e.g., "UG" for Simic commanders)
            max_cmc: Maximum mana value
            limit: Maximum results

        Returns:
            List of commander-eligible cards
        """
        return self.search(
            is_commander=True,
            is_legal_commander=True,
            color_identity=colors,
            max_cmc=max_cmc,
            limit=limit,
        )

    def get_cards_for_commander(
        self,
        commander: Card,
        card_types: Optional[list[str]] = None,
        max_cmc: Optional[int] = None,
        text_contains: Optional[str] = None,
        limit: int = 100,
    ) -> list[Card]:
        """
        Get cards legal in a commander's color identity.

        Args:
            commander: The commander card
            card_types: Filter by card types
            max_cmc: Maximum mana value
            text_contains: Filter by oracle text
            limit: Maximum results

        Returns:
            List of cards legal in the commander's deck
        """
        identity = _colors_to_string(commander.color_identity)

        return self.search(
            color_identity=identity or "C",  # Colorless if no colors
            card_types=card_types,
            max_cmc=max_cmc,
            text_contains=text_contains,
            is_legal_commander=True,
            limit=limit,
        )

    def get_random_cards(
        self, count: int = 10, card_types: Optional[list[str]] = None
    ) -> list[Card]:
        """Get random cards, optionally filtered by type."""
        session = self._get_session()
        query = session.query(CardModel).filter(CardModel.legal_commander == True)

        if card_types:
            type_filters = [CardModel.type_line.ilike(f"%{ct.lower()}%") for ct in card_types]
            query = query.filter(or_(*type_filters))

        # Use database-specific random function
        if self._manager.config.is_sqlite:
            query = query.order_by(func.random())
        else:
            query = query.order_by(func.random())

        results = query.limit(count).all()
        return [Card.from_scryfall(json.loads(r.scryfall_json)) for r in results]

    def count(self, **kwargs) -> int:
        """Count cards matching search criteria."""
        if not kwargs:
            session = self._get_session()
            return session.query(func.count(CardModel.id)).scalar() or 0

        # Use search logic but count instead
        cards = self.search(**kwargs, limit=100000)
        return len(cards)

    def get_metadata(self) -> SyncMetadata:
        """Get sync metadata."""
        session = self._get_session()
        result = session.query(SyncMetadataModel).order_by(SyncMetadataModel.id.desc()).first()

        if result:
            return SyncMetadata(
                last_updated=result.last_updated,
                card_count=result.card_count,
                scryfall_updated_at=result.scryfall_updated_at,
            )

        return SyncMetadata(last_updated=None, card_count=0, scryfall_updated_at=None)

    def get_card_count(self) -> int:
        """Get total number of cards in database."""
        session = self._get_session()
        return session.query(func.count(CardModel.id)).scalar() or 0

    def get_scryfall_json(self, name: str) -> Optional[dict]:
        """Get the raw Scryfall JSON for a card (useful for debugging)."""
        session = self._get_session()
        result = (
            session.query(CardModel.scryfall_json)
            .filter(CardModel.name_lower == name.lower())
            .first()
        )
        if result:
            return json.loads(result.scryfall_json)
        return None


# -----------------------------------------------------------------------------
# Schema Creation
# -----------------------------------------------------------------------------


def create_schema(db_path: Optional[Path] = None, config: Optional[DatabaseConfig] = None):
    """
    Create the database schema.

    This is called by the ingest module when setting up a new database.
    Returns the DatabaseManager for further operations.
    """
    if config:
        manager = DatabaseManager(config)
    elif db_path:
        manager = DatabaseManager(DatabaseConfig.sqlite(str(db_path)))
    else:
        manager = DatabaseManager()

    manager.create_tables()
    return manager


def create_edhrec_schema(db_path: Optional[Path] = None, config: Optional[DatabaseConfig] = None):
    """
    Create the EDHREC-related database tables.

    This is now handled by create_schema() since all tables are defined together.
    Kept for backwards compatibility.
    """
    return create_schema(db_path, config)


def create_features_schema(db_path: Optional[Path] = None, config: Optional[DatabaseConfig] = None):
    """
    Create the card features database table.

    This is now handled by create_schema() since all tables are defined together.
    Kept for backwards compatibility.
    """
    return create_schema(db_path, config)


def create_categories_schema(
    db_path: Optional[Path] = None, config: Optional[DatabaseConfig] = None
):
    """
    Create the card categories database table.

    This is now handled by create_schema() since all tables are defined together.
    Kept for backwards compatibility.
    """
    return create_schema(db_path, config)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _normalize_colors(colors: str) -> str:
    """Normalize a color string to sorted uppercase (e.g., 'ug' -> 'GU')."""
    valid_colors = set("WUBRGC")
    normalized = sorted(c.upper() for c in colors if c.upper() in valid_colors)
    return "".join(normalized)


def _colors_to_string(colors: set[Color]) -> str:
    """Convert a set of Color enums to a sorted string."""
    color_chars = []
    for color in colors:
        if color == Color.WHITE:
            color_chars.append("W")
        elif color == Color.BLUE:
            color_chars.append("U")
        elif color == Color.BLACK:
            color_chars.append("B")
        elif color == Color.RED:
            color_chars.append("R")
        elif color == Color.GREEN:
            color_chars.append("G")
    return "".join(sorted(color_chars))


def _color_identity_fits_filter(column, allowed_identity: str):
    """
    Create SQLAlchemy filter for color identity matching.

    Returns a filter that matches cards whose color identity is a subset
    of the allowed colors.
    """
    # Colorless cards fit in any deck
    allowed_set = set(allowed_identity.upper())

    # Build conditions for each possible color
    conditions = []
    for color in "WUBRG":
        if color not in allowed_set:
            # If this color is not allowed, card must not have it
            conditions.append(~column.contains(color))

    if conditions:
        return and_(*conditions)
    return True  # All colors allowed
