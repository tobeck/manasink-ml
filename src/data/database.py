"""
SQLite database for card storage and retrieval.

This module provides a CardDatabase class that stores Scryfall card data
in SQLite for efficient querying. The full Scryfall JSON is preserved,
with indexed columns extracted for common query patterns.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from game.card import Card, CardType, Color


# Default database location
DEFAULT_DB_PATH = Path("data/cards.db")


@dataclass
class SyncMetadata:
    """Metadata about the last database sync."""

    last_updated: Optional[datetime]
    card_count: int
    scryfall_updated_at: Optional[str]


class CardDatabase:
    """
    SQLite-backed card database for efficient queries.

    Stores full Scryfall JSON alongside indexed columns for fast lookups.
    This class is read-heavy and expects writes to happen via the ingest module.

    Example:
        db = CardDatabase()
        card = db.get_card("Sol Ring")
        commanders = db.get_commanders(colors="UG")
        creatures = db.search(card_types=["creature"], max_cmc=3)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Database not found at {self.db_path}. "
                    "Run 'manasink-data sync' to download card data."
                )
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

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
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT scryfall_json FROM cards WHERE name_lower = ?",
            (name.lower(),)
        )
        row = cursor.fetchone()
        if row:
            return Card.from_scryfall(json.loads(row["scryfall_json"]))
        return None

    def get_card_by_id(self, scryfall_id: str) -> Optional[Card]:
        """Get a card by Scryfall ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT scryfall_json FROM cards WHERE scryfall_id = ?",
            (scryfall_id,)
        )
        row = cursor.fetchone()
        if row:
            return Card.from_scryfall(json.loads(row["scryfall_json"]))
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
        conn = self._get_connection()

        conditions = []
        params = []

        if name_contains:
            conditions.append("name_lower LIKE ?")
            params.append(f"%{name_contains.lower()}%")

        if card_types:
            type_conditions = []
            for card_type in card_types:
                type_conditions.append("type_line LIKE ?")
                params.append(f"%{card_type.lower()}%")
            conditions.append(f"({' OR '.join(type_conditions)})")

        if colors:
            conditions.append("colors = ?")
            params.append(_normalize_colors(colors))

        if color_identity:
            # Cards whose identity is a subset of the given colors
            identity_normalized = _normalize_colors(color_identity)
            # Build condition: each color in card's identity must be in the filter
            conditions.append("_color_identity_fits(color_identity, ?)")
            params.append(identity_normalized)

        if min_cmc is not None:
            conditions.append("cmc >= ?")
            params.append(min_cmc)

        if max_cmc is not None:
            conditions.append("cmc <= ?")
            params.append(max_cmc)

        if is_commander is not None:
            conditions.append("is_commander = ?")
            params.append(1 if is_commander else 0)

        if is_legal_commander is not None:
            conditions.append("legal_commander = ?")
            params.append(1 if is_legal_commander else 0)

        if text_contains:
            conditions.append("oracle_text_lower LIKE ?")
            params.append(f"%{text_contains.lower()}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Register custom function for color identity matching
        conn.create_function("_color_identity_fits", 2, _color_identity_fits)

        query = f"""
            SELECT scryfall_json
            FROM cards
            WHERE {where_clause}
            ORDER BY name_lower
            LIMIT ?
        """
        params.append(limit)

        cursor = conn.execute(query, params)
        return [Card.from_scryfall(json.loads(row["scryfall_json"])) for row in cursor]

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
        # Build color identity string
        identity = _colors_to_string(commander.color_identity)

        return self.search(
            color_identity=identity or "C",  # Colorless if no colors
            card_types=card_types,
            max_cmc=max_cmc,
            text_contains=text_contains,
            is_legal_commander=True,
            limit=limit,
        )

    def get_random_cards(self, count: int = 10, card_types: Optional[list[str]] = None) -> list[Card]:
        """Get random cards, optionally filtered by type."""
        conn = self._get_connection()

        conditions = ["legal_commander = 1"]
        params = []

        if card_types:
            type_conditions = []
            for card_type in card_types:
                type_conditions.append("type_line LIKE ?")
                params.append(f"%{card_type.lower()}%")
            conditions.append(f"({' OR '.join(type_conditions)})")

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT scryfall_json
            FROM cards
            WHERE {where_clause}
            ORDER BY RANDOM()
            LIMIT ?
        """
        params.append(count)

        cursor = conn.execute(query, params)
        return [Card.from_scryfall(json.loads(row["scryfall_json"])) for row in cursor]

    def count(self, **kwargs) -> int:
        """Count cards matching search criteria."""
        conn = self._get_connection()

        if not kwargs:
            cursor = conn.execute("SELECT COUNT(*) FROM cards")
            return cursor.fetchone()[0]

        # Use search logic but count instead
        # This is a simplified version - for complex counts, use search + len
        cards = self.search(**kwargs, limit=100000)
        return len(cards)

    def get_metadata(self) -> SyncMetadata:
        """Get sync metadata."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM sync_metadata ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()

        if row:
            last_updated = None
            if row["last_updated"]:
                last_updated = datetime.fromisoformat(row["last_updated"])

            return SyncMetadata(
                last_updated=last_updated,
                card_count=row["card_count"],
                scryfall_updated_at=row["scryfall_updated_at"],
            )

        return SyncMetadata(last_updated=None, card_count=0, scryfall_updated_at=None)

    def get_scryfall_json(self, name: str) -> Optional[dict]:
        """Get the raw Scryfall JSON for a card (useful for debugging)."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT scryfall_json FROM cards WHERE name_lower = ?",
            (name.lower(),)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["scryfall_json"])
        return None


# -----------------------------------------------------------------------------
# Schema Creation (used by ingest module)
# -----------------------------------------------------------------------------

def create_schema(db_path: Path) -> sqlite3.Connection:
    """
    Create the database schema.

    This is called by the ingest module when setting up a new database.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.executescript("""
        -- Main cards table
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scryfall_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            name_lower TEXT NOT NULL,
            mana_cost TEXT,
            cmc REAL,
            type_line TEXT,
            colors TEXT,  -- Sorted color string (e.g., "BU" for blue-black)
            color_identity TEXT,  -- Sorted color identity string
            oracle_text TEXT,
            oracle_text_lower TEXT,
            power TEXT,
            toughness TEXT,
            is_commander INTEGER DEFAULT 0,
            legal_commander INTEGER DEFAULT 0,
            rarity TEXT,
            set_code TEXT,
            scryfall_json TEXT NOT NULL
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_cards_name_lower ON cards(name_lower);
        CREATE INDEX IF NOT EXISTS idx_cards_cmc ON cards(cmc);
        CREATE INDEX IF NOT EXISTS idx_cards_color_identity ON cards(color_identity);
        CREATE INDEX IF NOT EXISTS idx_cards_type_line ON cards(type_line);
        CREATE INDEX IF NOT EXISTS idx_cards_is_commander ON cards(is_commander);
        CREATE INDEX IF NOT EXISTS idx_cards_legal_commander ON cards(legal_commander);

        -- Sync metadata table
        CREATE TABLE IF NOT EXISTS sync_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_updated TEXT,
            card_count INTEGER,
            scryfall_updated_at TEXT
        );
    """)

    conn.commit()
    return conn


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


def _color_identity_fits(card_identity: str, allowed_identity: str) -> bool:
    """
    Check if a card's color identity fits within allowed colors.

    Used as a SQLite custom function for color identity filtering.
    """
    if not card_identity:
        return True  # Colorless fits in any deck

    allowed_set = set(allowed_identity.upper())
    card_set = set(card_identity.upper())

    return card_set.issubset(allowed_set)
