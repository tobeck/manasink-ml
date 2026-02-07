"""
Category aggregation for card functional roles.

This module aggregates EDHREC category data across all commanders to build
a comprehensive view of each card's functional roles. Categories like
"Ramp", "Card Draw", "Removal" etc. help identify what role a card plays
in a deck.

The aggregated data is useful for:
- Computing role scores for cards
- Understanding card utility across different commanders
- Building deck composition heuristics
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .database import DEFAULT_DB_PATH, create_categories_schema

# Standard EDHREC categories and their aliases
CATEGORY_MAPPING = {
    # Ramp
    "Ramp": "Ramp",
    "Mana Rocks": "Ramp",
    "Mana Dorks": "Ramp",
    "Land Ramp": "Ramp",
    # Card Draw
    "Card Draw": "Card Draw",
    "Draw": "Card Draw",
    "Card Advantage": "Card Draw",
    # Removal
    "Removal": "Removal",
    "Spot Removal": "Removal",
    "Single Target Removal": "Removal",
    "Targeted Removal": "Removal",
    # Board Wipes
    "Board Wipes": "Board Wipe",
    "Sweepers": "Board Wipe",
    "Wrath": "Board Wipe",
    "Mass Removal": "Board Wipe",
    # Protection
    "Protection": "Protection",
    "Counterspells": "Protection",
    "Counter": "Protection",
    # Finishers
    "Finishers": "Finisher",
    "Win Conditions": "Finisher",
    "Combo": "Finisher",
    # Creatures
    "Creatures": "Creatures",
    # Lands
    "Lands": "Lands",
    # Artifacts
    "Artifacts": "Artifacts",
    # Enchantments
    "Enchantments": "Enchantments",
    # Instants
    "Instants": "Instants",
    # Sorceries
    "Sorceries": "Sorceries",
}


# Patterns for role detection when categories aren't available
ROLE_PATTERNS = {
    "Ramp": [
        r"add \{[WUBRGC]\}",
        r"add \d+ mana",
        r"search your library for .* land",
        r"put .* land .* onto the battlefield",
        r"treasure token",
    ],
    "Card Draw": [
        r"draw .* card",
        r"draw a card",
        r"scry \d+",
        r"look at the top",
    ],
    "Removal": [
        r"destroy target",
        r"exile target",
        r"deals? \d+ damage to",
        r"counter target",
        r"return target .* to its owner's hand",
    ],
    "Board Wipe": [
        r"destroy all",
        r"exile all",
        r"deals? \d+ damage to each",
    ],
    "Protection": [
        r"hexproof",
        r"indestructible",
        r"protection from",
        r"can't be .* target",
    ],
    "Finisher": [
        r"you win the game",
        r"opponent loses the game",
        r"infect",
        r"annihilator",
    ],
}


@dataclass
class CardCategoryInfo:
    """Aggregated category information for a card."""

    card_name: str
    categories: dict[str, dict]  # category -> {occurrences, commanders, avg_synergy, avg_inclusion}

    @property
    def primary_category(self) -> Optional[str]:
        """Return the most common category for this card."""
        if not self.categories:
            return None
        return max(self.categories.items(), key=lambda x: x[1].get("commander_count", 0))[0]

    @property
    def role_scores(self) -> dict[str, float]:
        """Convert category data to role scores (0-1)."""
        scores = {}
        for cat, info in self.categories.items():
            # Normalize by max expected commander count (100)
            score = min(info.get("commander_count", 0) / 50.0, 1.0)
            scores[cat] = score
        return scores


def normalize_category(category: str) -> str:
    """Normalize category name to standard form."""
    if not category:
        return "Other"

    # Direct mapping
    if category in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[category]

    # Check if it contains any known category
    category_lower = category.lower()
    for key, value in CATEGORY_MAPPING.items():
        if key.lower() in category_lower:
            return value

    return category


def populate_card_categories(
    db_path: Optional[Path] = None,
    show_progress: bool = True,
) -> int:
    """
    Aggregate card categories from commander_recommendations table.

    This creates a summary of how each card is categorized across
    all commanders, useful for understanding card roles.

    Args:
        db_path: Path to SQLite database
        show_progress: Show progress indicator

    Returns:
        Number of cards processed
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Create categories table if needed
    conn = create_categories_schema(db_path)
    conn.row_factory = sqlite3.Row

    # Check if recommendations table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='commander_recommendations'"
    )
    if not cursor.fetchone():
        print("commander_recommendations table not found. Run edhrec-sync first.")
        conn.close()
        return 0

    print("Aggregating card categories from EDHREC data...")

    # Get all unique card/category pairs with aggregated stats
    cursor = conn.execute("""
        SELECT
            card_name,
            category,
            COUNT(*) as total_occurrences,
            COUNT(DISTINCT commander_id) as commander_count,
            AVG(synergy_score) as avg_synergy,
            AVG(inclusion_rate) as avg_inclusion
        FROM commander_recommendations
        WHERE category IS NOT NULL AND category != ''
        GROUP BY card_name, category
    """)

    rows = cursor.fetchall()
    print(f"Found {len(rows)} card/category pairs")

    if show_progress and HAS_TQDM:
        rows = tqdm(rows, desc="Processing categories")

    processed = 0
    for row in rows:
        card_name = row["card_name"]
        category = normalize_category(row["category"])

        conn.execute(
            """
            INSERT OR REPLACE INTO card_categories
            (card_name, category, total_occurrences, commander_count,
             avg_synergy_score, avg_inclusion_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                card_name,
                category,
                row["total_occurrences"],
                row["commander_count"],
                row["avg_synergy"] or 0,
                row["avg_inclusion"] or 0,
            ),
        )
        processed += 1

    conn.commit()

    # Get summary stats
    cursor = conn.execute("SELECT COUNT(DISTINCT card_name) as cards FROM card_categories")
    unique_cards = cursor.fetchone()["cards"]

    cursor = conn.execute("SELECT COUNT(DISTINCT category) as cats FROM card_categories")
    unique_cats = cursor.fetchone()["cats"]

    conn.close()

    print(f"Aggregated {unique_cats} categories for {unique_cards:,} cards")
    return processed


def compute_role_scores(
    card_name: str,
    oracle_text: str,
    categories: Optional[list[dict]] = None,
    db_path: Optional[Path] = None,
) -> dict[str, float]:
    """
    Compute role scores for a card based on categories and oracle text.

    Args:
        card_name: Name of the card
        oracle_text: Oracle text of the card
        categories: Optional list of category dicts from get_card_categories
        db_path: Path to database (used if categories not provided)

    Returns:
        Dict mapping role names to scores (0.0 to 1.0)
    """
    scores = {
        "Ramp": 0.0,
        "Card Draw": 0.0,
        "Removal": 0.0,
        "Board Wipe": 0.0,
        "Protection": 0.0,
        "Finisher": 0.0,
    }

    # Get categories from DB if not provided
    if categories is None and db_path:
        categories = get_card_categories(card_name, db_path)

    # Score from categories
    if categories:
        for cat_info in categories:
            cat = cat_info.get("category", "")
            if cat in scores:
                # Higher commander count = more confident role assignment
                cmd_count = cat_info.get("commander_count", 0)
                synergy = cat_info.get("avg_synergy_score", 0) or 0
                # Score based on how many commanders use it in this role + synergy
                scores[cat] = max(scores[cat], min((cmd_count / 30.0) + (synergy * 0.5), 1.0))

    # Supplement with text pattern matching
    if oracle_text:
        text_lower = oracle_text.lower()
        for role, patterns in ROLE_PATTERNS.items():
            if role in scores:
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        scores[role] = max(scores[role], 0.5)  # Text match = 0.5 minimum
                        break

    return scores


def get_card_categories(
    card_name: str,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Get all categories for a card.

    Args:
        card_name: Exact card name
        db_path: Path to SQLite database

    Returns:
        List of category dicts with counts and scores
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """
        SELECT category, total_occurrences, commander_count,
               avg_synergy_score, avg_inclusion_rate
        FROM card_categories
        WHERE card_name = ?
        ORDER BY commander_count DESC
    """,
        (card_name,),
    )

    results = [
        {
            "category": row["category"],
            "total_occurrences": row["total_occurrences"],
            "commander_count": row["commander_count"],
            "avg_synergy_score": row["avg_synergy_score"],
            "avg_inclusion_rate": row["avg_inclusion_rate"],
        }
        for row in cursor
    ]

    conn.close()
    return results


def get_cards_by_category(
    category: str,
    db_path: Optional[Path] = None,
    limit: int = 100,
    min_commander_count: int = 1,
) -> list[dict]:
    """
    Get cards that belong to a specific category.

    Args:
        category: Category name (e.g., "Ramp", "Card Draw")
        db_path: Path to SQLite database
        limit: Maximum results
        min_commander_count: Minimum number of commanders using the card in this role

    Returns:
        List of card dicts sorted by commander count
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Normalize the category for matching
    normalized = normalize_category(category)

    cursor = conn.execute(
        """
        SELECT card_name, total_occurrences, commander_count,
               avg_synergy_score, avg_inclusion_rate
        FROM card_categories
        WHERE category = ? AND commander_count >= ?
        ORDER BY commander_count DESC, avg_synergy_score DESC
        LIMIT ?
    """,
        (normalized, min_commander_count, limit),
    )

    results = [
        {
            "card_name": row["card_name"],
            "category": normalized,
            "total_occurrences": row["total_occurrences"],
            "commander_count": row["commander_count"],
            "avg_synergy_score": row["avg_synergy_score"],
            "avg_inclusion_rate": row["avg_inclusion_rate"],
        }
        for row in cursor
    ]

    conn.close()
    return results


def get_top_cards_per_role(
    db_path: Optional[Path] = None,
    top_n: int = 10,
) -> dict[str, list[dict]]:
    """
    Get the top cards for each functional role.

    Args:
        db_path: Path to SQLite database
        top_n: Number of top cards per role

    Returns:
        Dict mapping role names to lists of top cards
    """
    roles = ["Ramp", "Card Draw", "Removal", "Board Wipe", "Protection", "Finisher"]

    result = {}
    for role in roles:
        result[role] = get_cards_by_category(
            role,
            db_path=db_path,
            limit=top_n,
            min_commander_count=5,
        )

    return result


def get_categories_stats(db_path: Optional[Path] = None) -> dict:
    """Get statistics about the card_categories table."""
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='card_categories'"
    )
    if not cursor.fetchone():
        conn.close()
        return {"exists": True, "categories_initialized": False}

    stats = {"exists": True, "categories_initialized": True}

    # Total entries
    cursor = conn.execute("SELECT COUNT(*) as count FROM card_categories")
    stats["total_entries"] = cursor.fetchone()["count"]

    # Unique cards
    cursor = conn.execute("SELECT COUNT(DISTINCT card_name) as count FROM card_categories")
    stats["unique_cards"] = cursor.fetchone()["count"]

    # Unique categories
    cursor = conn.execute("SELECT COUNT(DISTINCT category) as count FROM card_categories")
    stats["unique_categories"] = cursor.fetchone()["count"]

    # Top categories by card count
    cursor = conn.execute("""
        SELECT category, COUNT(DISTINCT card_name) as card_count
        FROM card_categories
        GROUP BY category
        ORDER BY card_count DESC
        LIMIT 10
    """)
    stats["top_categories"] = [
        {"category": row["category"], "card_count": row["card_count"]} for row in cursor
    ]

    conn.close()
    return stats
