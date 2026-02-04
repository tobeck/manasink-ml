"""
EDHREC data ingestion module.

This module handles:
- Fetching top commanders and their recommendations from EDHREC
- Storing data in SQLite alongside Scryfall card data
- Linking EDHREC card names to Scryfall IDs
- Power level estimation using salt scores

Usage:
    from src.data.edhrec_ingest import sync_edhrec_data
    sync_edhrec_data(limit=100)

Or via CLI:
    manasink-data edhrec-sync --limit 100
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .database import DEFAULT_DB_PATH, create_edhrec_schema
from .edhrec import EDHRecClient


@dataclass
class PowerLevelEstimate:
    """Power level estimation based on salt scores."""
    salt_sum: float
    bracket: int  # 1-4
    power_score: int  # 1-10
    description: str


# Power level thresholds based on salt sum
POWER_BRACKETS = [
    (25, 1, 3, "Casual"),
    (35, 2, 5, "Low"),
    (50, 3, 7, "Mid"),
    (float("inf"), 4, 9, "High"),
]


def estimate_deck_power(
    cards: list[str],
    salt_scores: dict[str, float],
) -> PowerLevelEstimate:
    """
    Estimate deck power level from card salt scores.

    Uses salt score sum as a proxy for power level:
    - Salt < 25: Bracket 1 (Casual), Power 1-4
    - Salt 25-35: Bracket 2 (Low), Power 4-6
    - Salt 35-50: Bracket 3 (Mid), Power 6-8
    - Salt > 50: Bracket 4 (High), Power 8-10

    Args:
        cards: List of card names in the deck
        salt_scores: Dict mapping card names to salt scores

    Returns:
        PowerLevelEstimate with salt_sum, bracket, power_score, and description
    """
    salt_sum = sum(salt_scores.get(card, 0) for card in cards)

    for threshold, bracket, power, desc in POWER_BRACKETS:
        if salt_sum < threshold:
            return PowerLevelEstimate(
                salt_sum=salt_sum,
                bracket=bracket,
                power_score=power,
                description=desc,
            )

    # Fallback (shouldn't reach here due to inf threshold)
    return PowerLevelEstimate(
        salt_sum=salt_sum,
        bracket=4,
        power_score=10,
        description="High",
    )


def sync_edhrec_data(
    db_path: Optional[Path] = None,
    limit: int = 100,
    force: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Sync EDHREC data to the database.

    Fetches top commanders, their card recommendations, average decks,
    and global salt scores. Links card names to Scryfall IDs when possible.

    Args:
        db_path: Path to SQLite database
        limit: Number of top commanders to sync
        force: Force re-fetch even if recently synced
        show_progress: Show progress bars

    Returns:
        Dict with sync results
    """
    db_path = db_path or DEFAULT_DB_PATH

    result = {
        "commanders_synced": 0,
        "recommendations_synced": 0,
        "average_deck_cards": 0,
        "salt_scores_synced": 0,
        "skipped": False,
    }

    # Ensure database exists with EDHREC tables
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Run 'manasink-data sync' first to create the card database.")
        return result

    # Create EDHREC schema (tables created if not exist)
    conn = create_edhrec_schema(db_path)
    conn.row_factory = sqlite3.Row

    # Check if we need to sync
    if not force:
        cursor = conn.execute(
            "SELECT last_updated FROM edhrec_sync_metadata ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row and row["last_updated"]:
            # Check if synced within last 24 hours
            last_sync = datetime.fromisoformat(row["last_updated"])
            hours_since = (datetime.now() - last_sync).total_seconds() / 3600
            if hours_since < 24:
                print(f"EDHREC data synced {hours_since:.1f} hours ago. Use --force to re-sync.")
                result["skipped"] = True
                conn.close()
                return result

    client = EDHRecClient()

    # Build a card name -> scryfall_id lookup from existing cards table
    print("Building card name lookup...")
    cursor = conn.execute("SELECT name, scryfall_id FROM cards")
    card_lookup = {row["name"]: row["scryfall_id"] for row in cursor}
    print(f"  Found {len(card_lookup):,} cards in database")

    # Step 1: Fetch top commanders
    print(f"\nFetching top {limit} commanders from EDHREC...")
    commanders = client.get_top_commanders(limit=limit, use_cache=not force)
    print(f"  Found {len(commanders)} commanders")

    # Step 2: Sync each commander
    print("\nSyncing commander data...")

    if show_progress and HAS_TQDM:
        iterator = tqdm(commanders, desc="Commanders")
    else:
        iterator = commanders

    for cmd in iterator:
        cmd_name = cmd.get("name", "")
        cmd_slug = client.name_to_slug(cmd_name)

        # Get full commander data
        cmd_data = client.get_commander_data(cmd_name, use_cache=not force)
        if not cmd_data:
            continue

        # Insert/update commander record
        scryfall_id = card_lookup.get(cmd_name)
        color_identity = "".join(sorted(cmd_data.get("color_identity", [])))

        conn.execute(
            """
            INSERT OR REPLACE INTO commanders
            (name, name_slug, scryfall_id, edhrec_rank, num_decks,
             salt_score, color_identity, last_synced, edhrec_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cmd_name,
                cmd_slug,
                scryfall_id,
                cmd_data.get("edhrec_rank"),
                cmd_data.get("num_decks", 0),
                cmd_data.get("salt_score"),
                color_identity,
                datetime.now().isoformat(),
                json.dumps(cmd_data.get("raw_data")),
            ),
        )

        # Get commander ID
        cursor = conn.execute(
            "SELECT id FROM commanders WHERE name = ?", (cmd_name,)
        )
        commander_id = cursor.fetchone()["id"]

        # Insert recommendations
        recommendations = cmd_data.get("recommendations", [])
        for rec in recommendations:
            card_name = rec.get("card_name", "")
            if not card_name:
                continue

            rec_scryfall_id = card_lookup.get(card_name)

            conn.execute(
                """
                INSERT OR REPLACE INTO commander_recommendations
                (commander_id, card_name, scryfall_id, inclusion_rate,
                 synergy_score, num_decks, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commander_id,
                    card_name,
                    rec_scryfall_id,
                    rec.get("inclusion_rate", 0),
                    rec.get("synergy_score", 0),
                    rec.get("num_decks", 0),
                    rec.get("category", ""),
                ),
            )
            result["recommendations_synced"] += 1

        # Fetch and insert average deck
        avg_deck = client.get_average_deck(cmd_name, use_cache=not force)
        for slot_num, card_name in enumerate(avg_deck):
            avg_scryfall_id = card_lookup.get(card_name)

            conn.execute(
                """
                INSERT OR REPLACE INTO average_decks
                (commander_id, card_name, scryfall_id, slot_number)
                VALUES (?, ?, ?, ?)
                """,
                (commander_id, card_name, avg_scryfall_id, slot_num),
            )
            result["average_deck_cards"] += 1

        result["commanders_synced"] += 1
        conn.commit()

    # Step 3: Fetch global salt scores
    print("\nFetching salt scores...")
    salt_scores = client.get_salt_scores(use_cache=not force)
    print(f"  Found {len(salt_scores)} salt scores")

    # Insert salt scores with rank
    sorted_salt = sorted(salt_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (card_name, salt_score) in enumerate(sorted_salt, 1):
        salt_scryfall_id = card_lookup.get(card_name)

        conn.execute(
            """
            INSERT OR REPLACE INTO card_salt_scores
            (card_name, scryfall_id, salt_score, salt_rank)
            VALUES (?, ?, ?, ?)
            """,
            (card_name, salt_scryfall_id, salt_score, rank),
        )
        result["salt_scores_synced"] += 1

    # Record sync metadata
    conn.execute(
        """
        INSERT INTO edhrec_sync_metadata
        (last_updated, commanders_synced, cards_synced, salt_cards_synced)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            result["commanders_synced"],
            result["recommendations_synced"],
            result["salt_scores_synced"],
        ),
    )

    conn.commit()
    conn.close()

    print(f"\nEDHREC sync complete!")
    print(f"  Commanders: {result['commanders_synced']}")
    print(f"  Recommendations: {result['recommendations_synced']:,}")
    print(f"  Average deck cards: {result['average_deck_cards']:,}")
    print(f"  Salt scores: {result['salt_scores_synced']:,}")

    return result


def get_edhrec_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get statistics about the EDHREC data in the database.

    Returns:
        Dict with EDHREC statistics
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {"exists": True, "path": str(db_path)}

    # Check if EDHREC tables exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='commanders'"
    )
    if not cursor.fetchone():
        stats["edhrec_initialized"] = False
        conn.close()
        return stats

    stats["edhrec_initialized"] = True

    # Commander count
    cursor = conn.execute("SELECT COUNT(*) as count FROM commanders")
    stats["commanders"] = cursor.fetchone()["count"]

    # Recommendations count
    cursor = conn.execute("SELECT COUNT(*) as count FROM commander_recommendations")
    stats["recommendations"] = cursor.fetchone()["count"]

    # Average deck cards
    cursor = conn.execute("SELECT COUNT(*) as count FROM average_decks")
    stats["average_deck_cards"] = cursor.fetchone()["count"]

    # Salt scores count
    cursor = conn.execute("SELECT COUNT(*) as count FROM card_salt_scores")
    stats["salt_scores"] = cursor.fetchone()["count"]

    # Top 5 commanders by deck count
    cursor = conn.execute(
        """
        SELECT name, num_decks, edhrec_rank
        FROM commanders
        ORDER BY num_decks DESC
        LIMIT 5
        """
    )
    stats["top_commanders"] = [
        {"name": row["name"], "num_decks": row["num_decks"], "rank": row["edhrec_rank"]}
        for row in cursor
    ]

    # Highest salt cards
    cursor = conn.execute(
        """
        SELECT card_name, salt_score, salt_rank
        FROM card_salt_scores
        ORDER BY salt_score DESC
        LIMIT 5
        """
    )
    stats["highest_salt"] = [
        {"name": row["card_name"], "salt": row["salt_score"], "rank": row["salt_rank"]}
        for row in cursor
    ]

    # Sync metadata
    cursor = conn.execute(
        "SELECT * FROM edhrec_sync_metadata ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row:
        stats["last_updated"] = row["last_updated"]
        stats["last_commanders_synced"] = row["commanders_synced"]

    conn.close()
    return stats


def get_commander_recommendations(
    commander_name: str,
    db_path: Optional[Path] = None,
    limit: int = 50,
    min_synergy: Optional[float] = None,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Get card recommendations for a commander from the database.

    Args:
        commander_name: Name of the commander
        db_path: Path to SQLite database
        limit: Maximum results to return
        min_synergy: Minimum synergy score filter
        category: Filter by category (e.g., "Creatures", "Ramp")

    Returns:
        List of recommendation dicts
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find commander
    cursor = conn.execute(
        "SELECT id FROM commanders WHERE name = ? OR name LIKE ?",
        (commander_name, f"%{commander_name}%"),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return []

    commander_id = row["id"]

    # Build query
    conditions = ["commander_id = ?"]
    params = [commander_id]

    if min_synergy is not None:
        conditions.append("synergy_score >= ?")
        params.append(min_synergy)

    if category:
        conditions.append("category LIKE ?")
        params.append(f"%{category}%")

    where_clause = " AND ".join(conditions)

    cursor = conn.execute(
        f"""
        SELECT card_name, inclusion_rate, synergy_score, num_decks, category
        FROM commander_recommendations
        WHERE {where_clause}
        ORDER BY synergy_score DESC
        LIMIT ?
        """,
        params + [limit],
    )

    recommendations = [
        {
            "card_name": row["card_name"],
            "inclusion_rate": row["inclusion_rate"],
            "synergy_score": row["synergy_score"],
            "num_decks": row["num_decks"],
            "category": row["category"],
        }
        for row in cursor
    ]

    conn.close()
    return recommendations


def get_salt_scores_from_db(
    db_path: Optional[Path] = None,
) -> dict[str, float]:
    """
    Get all salt scores from the database.

    Returns:
        Dict mapping card names to salt scores
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("SELECT card_name, salt_score FROM card_salt_scores")
    scores = {row["card_name"]: row["salt_score"] for row in cursor}

    conn.close()
    return scores
