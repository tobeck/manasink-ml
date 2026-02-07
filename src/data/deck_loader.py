"""
EDHREC deck loading for simulation.

This module loads real decks from EDHREC average deck data into Card objects
suitable for the game simulator. It bridges the gap between the data layer
and the game simulation layer.

Features:
- Load average decks for any synced commander
- Resolve card names to Card objects from the database
- Track missing cards that couldn't be resolved
- Load synergy data alongside decks for training
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.game.card import Card, CardType, Color

from .database import DEFAULT_DB_PATH, CardDatabase


@dataclass
class DeckLoadResult:
    """Result of loading a deck from EDHREC data."""

    commander: Card
    deck: list[Card]  # The 99 (or fewer if cards missing)
    missing_cards: list[str]  # Card names that couldn't be resolved
    color_identity: set[Color]

    @property
    def is_complete(self) -> bool:
        """Return True if deck has all 99 cards."""
        return len(self.deck) == 99

    @property
    def deck_size(self) -> int:
        """Return current deck size."""
        return len(self.deck)


@dataclass
class SynergyData:
    """Synergy data for cards in a deck."""

    commander_name: str
    card_synergies: dict[str, float]  # card_name -> synergy_score
    card_inclusions: dict[str, float]  # card_name -> inclusion_rate
    card_categories: dict[str, str]  # card_name -> category

    def get_synergy(self, card_name: str) -> float:
        """Get synergy score for a card (0 if not found)."""
        return self.card_synergies.get(card_name, 0.0)

    def get_inclusion(self, card_name: str) -> float:
        """Get inclusion rate for a card (0 if not found)."""
        return self.card_inclusions.get(card_name, 0.0)

    def get_category(self, card_name: str) -> str:
        """Get category for a card (empty string if not found)."""
        return self.card_categories.get(card_name, "")


def load_deck_from_edhrec(
    commander_name: str,
    db_path: Path | None = None,
) -> DeckLoadResult | None:
    """
    Load a deck from EDHREC average deck data.

    Args:
        commander_name: Name of the commander
        db_path: Path to SQLite database

    Returns:
        DeckLoadResult with commander, deck cards, and missing cards
        Returns None if commander not found or no average deck data
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find commander in database
    cursor = conn.execute(
        "SELECT id, name, color_identity FROM commanders WHERE name = ? OR name LIKE ?",
        (commander_name, f"%{commander_name}%"),
    )
    cmd_row = cursor.fetchone()

    if not cmd_row:
        conn.close()
        return None

    commander_id = cmd_row["id"]
    commander_name_resolved = cmd_row["name"]
    color_identity_str = cmd_row["color_identity"] or ""

    # Parse color identity
    color_map = {
        "W": Color.WHITE,
        "U": Color.BLUE,
        "B": Color.BLACK,
        "R": Color.RED,
        "G": Color.GREEN,
    }
    color_identity = {color_map[c] for c in color_identity_str if c in color_map}

    # Get the commander Card object
    db = CardDatabase(db_path)
    commander_card = db.get_card(commander_name_resolved)

    if not commander_card:
        conn.close()
        db.close()
        return None

    # Get average deck card names
    cursor = conn.execute(
        "SELECT card_name FROM average_decks WHERE commander_id = ? ORDER BY slot_number",
        (commander_id,),
    )
    card_names = [row["card_name"] for row in cursor]
    conn.close()

    if not card_names:
        db.close()
        return None

    # Resolve card names to Card objects
    deck = []
    missing = []

    for name in card_names:
        card = db.get_card(name)
        if card:
            deck.append(card)
        else:
            missing.append(name)

    db.close()

    return DeckLoadResult(
        commander=commander_card,
        deck=deck,
        missing_cards=missing,
        color_identity=color_identity or commander_card.color_identity,
    )


def load_synergy_data(
    commander_name: str,
    db_path: Path | None = None,
) -> SynergyData | None:
    """
    Load synergy data for a commander's recommended cards.

    Args:
        commander_name: Name of the commander
        db_path: Path to SQLite database

    Returns:
        SynergyData with card synergies, inclusions, and categories
        Returns None if commander not found
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find commander
    cursor = conn.execute(
        "SELECT id, name FROM commanders WHERE name = ? OR name LIKE ?",
        (commander_name, f"%{commander_name}%"),
    )
    cmd_row = cursor.fetchone()

    if not cmd_row:
        conn.close()
        return None

    commander_id = cmd_row["id"]
    commander_name_resolved = cmd_row["name"]

    # Get all recommendations
    cursor = conn.execute(
        """
        SELECT card_name, synergy_score, inclusion_rate, category
        FROM commander_recommendations
        WHERE commander_id = ?
    """,
        (commander_id,),
    )

    synergies = {}
    inclusions = {}
    categories = {}

    for row in cursor:
        card_name = row["card_name"]
        synergies[card_name] = row["synergy_score"] or 0.0
        inclusions[card_name] = row["inclusion_rate"] or 0.0
        categories[card_name] = row["category"] or ""

    conn.close()

    return SynergyData(
        commander_name=commander_name_resolved,
        card_synergies=synergies,
        card_inclusions=inclusions,
        card_categories=categories,
    )


def load_deck_with_synergy_data(
    commander_name: str,
    db_path: Path | None = None,
) -> tuple[DeckLoadResult | None, SynergyData | None]:
    """
    Load both deck and synergy data for a commander.

    Args:
        commander_name: Name of the commander
        db_path: Path to SQLite database

    Returns:
        Tuple of (DeckLoadResult, SynergyData), either may be None
    """
    deck_result = load_deck_from_edhrec(commander_name, db_path)
    synergy_data = load_synergy_data(commander_name, db_path)

    return deck_result, synergy_data


def list_available_commanders(
    db_path: Path | None = None,
    min_decks: int = 100,
    color_identity: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    List commanders that have average deck data available.

    Args:
        db_path: Path to SQLite database
        min_decks: Minimum number of decks for the commander
        color_identity: Filter by color identity (e.g., "UG")
        limit: Maximum results

    Returns:
        List of commander dicts with name, num_decks, color_identity
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build query
    conditions = ["num_decks >= ?"]
    params = [min_decks]

    if color_identity:
        # Filter by color identity (subset match)
        conditions.append("color_identity = ?")
        params.append("".join(sorted(color_identity.upper())))

    # Only include commanders with average deck data
    conditions.append("id IN (SELECT DISTINCT commander_id FROM average_decks)")

    where_clause = " AND ".join(conditions)

    cursor = conn.execute(
        f"""
        SELECT name, num_decks, color_identity, edhrec_rank, salt_score
        FROM commanders
        WHERE {where_clause}
        ORDER BY num_decks DESC
        LIMIT ?
    """,
        params + [limit],
    )

    results = [
        {
            "name": row["name"],
            "num_decks": row["num_decks"],
            "color_identity": row["color_identity"],
            "edhrec_rank": row["edhrec_rank"],
            "salt_score": row["salt_score"],
        }
        for row in cursor
    ]

    conn.close()
    return results


def get_deck_stats(deck_result: DeckLoadResult) -> dict:
    """
    Get statistics about a loaded deck.

    Args:
        deck_result: Result from load_deck_from_edhrec

    Returns:
        Dict with deck statistics
    """
    stats = {
        "commander": deck_result.commander.name,
        "deck_size": deck_result.deck_size,
        "missing_count": len(deck_result.missing_cards),
        "is_complete": deck_result.is_complete,
        "color_identity": "".join(c.value for c in deck_result.color_identity if c.value != "C"),
    }

    # Count by type
    creatures = sum(1 for c in deck_result.deck if c.is_creature)
    lands = sum(1 for c in deck_result.deck if c.is_land)
    instants = sum(1 for c in deck_result.deck if c.is_instant)
    sorceries = sum(1 for c in deck_result.deck if CardType.SORCERY in c.card_types)
    artifacts = sum(1 for c in deck_result.deck if CardType.ARTIFACT in c.card_types)
    enchantments = sum(1 for c in deck_result.deck if CardType.ENCHANTMENT in c.card_types)

    stats["creatures"] = creatures
    stats["lands"] = lands
    stats["instants"] = instants
    stats["sorceries"] = sorceries
    stats["artifacts"] = artifacts
    stats["enchantments"] = enchantments
    stats["other"] = (
        deck_result.deck_size - creatures - lands - instants - sorceries - artifacts - enchantments
    )

    # Mana curve (non-lands)
    non_lands = [c for c in deck_result.deck if not c.is_land]
    cmc_distribution = {}
    for card in non_lands:
        cmc = min(card.cmc, 7)  # Group 7+ together
        cmc_distribution[cmc] = cmc_distribution.get(cmc, 0) + 1
    stats["cmc_distribution"] = cmc_distribution

    # Average CMC
    if non_lands:
        stats["avg_cmc"] = sum(c.cmc for c in non_lands) / len(non_lands)
    else:
        stats["avg_cmc"] = 0

    return stats
