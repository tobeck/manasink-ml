"""
EDHREC data ingestion module.

This module handles:
- Fetching top commanders and their recommendations from EDHREC
- Storing data in the database alongside Scryfall card data
- Linking EDHREC card names to Scryfall IDs
- Power level estimation using salt scores

Usage:
    from src.data.edhrec_ingest import sync_edhrec_data
    sync_edhrec_data(limit=100)

Or via CLI:
    manasink-data edhrec-sync --limit 100
"""

import json
from dataclasses import dataclass
from datetime import datetime

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .db_config import DatabaseConfig, DatabaseManager
from .db_models import (
    AverageDeckCard,
    CardModel,
    CardSaltScore,
    CommanderModel,
    CommanderRecommendation,
    EDHRecSyncMetadata,
)
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
    config: DatabaseConfig | None = None,
    limit: int = 100,
    force: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Sync EDHREC data to the database.

    Fetches top commanders, their card recommendations, average decks,
    and global salt scores. Links card names to Scryfall IDs when possible.

    Args:
        config: Database configuration
        limit: Number of top commanders to sync
        force: Force re-fetch even if recently synced
        show_progress: Show progress bars

    Returns:
        Dict with sync results
    """
    result = {
        "commanders_synced": 0,
        "recommendations_synced": 0,
        "average_deck_cards": 0,
        "salt_scores_synced": 0,
        "skipped": False,
    }

    # Set up database
    manager = DatabaseManager(config) if config else DatabaseManager()
    manager.create_tables()
    session = manager.session()

    # Check if we need to sync
    if not force:
        metadata = session.query(EDHRecSyncMetadata).order_by(EDHRecSyncMetadata.id.desc()).first()
        if metadata and metadata.last_updated:
            hours_since = (datetime.utcnow() - metadata.last_updated).total_seconds() / 3600
            if hours_since < 24:
                print(f"EDHREC data synced {hours_since:.1f} hours ago. Use --force to re-sync.")
                result["skipped"] = True
                session.close()
                return result

    client = EDHRecClient()

    # Build a card name -> scryfall_id lookup from existing cards table
    print("Building card name lookup...")
    card_lookup = {
        row.name: row.scryfall_id for row in session.query(CardModel.name, CardModel.scryfall_id)
    }
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

        scryfall_id = card_lookup.get(cmd_name)
        color_identity = "".join(sorted(cmd_data.get("color_identity", [])))

        # Upsert commander record
        existing = session.query(CommanderModel).filter_by(name=cmd_name).first()
        if existing:
            existing.name_slug = cmd_slug
            existing.scryfall_id = scryfall_id
            existing.edhrec_rank = cmd_data.get("edhrec_rank")
            existing.num_decks = cmd_data.get("num_decks", 0)
            existing.salt_score = cmd_data.get("salt_score")
            existing.color_identity = color_identity
            existing.last_synced = datetime.utcnow()
            existing.edhrec_json = json.dumps(cmd_data.get("raw_data"))
            commander_model = existing
        else:
            commander_model = CommanderModel(
                name=cmd_name,
                name_slug=cmd_slug,
                scryfall_id=scryfall_id,
                edhrec_rank=cmd_data.get("edhrec_rank"),
                num_decks=cmd_data.get("num_decks", 0),
                salt_score=cmd_data.get("salt_score"),
                color_identity=color_identity,
                last_synced=datetime.utcnow(),
                edhrec_json=json.dumps(cmd_data.get("raw_data")),
            )
            session.add(commander_model)

        session.flush()  # Get commander ID

        # Clear existing recommendations for this commander
        session.query(CommanderRecommendation).filter_by(commander_id=commander_model.id).delete()
        session.query(AverageDeckCard).filter_by(commander_id=commander_model.id).delete()

        # Insert recommendations
        recommendations = cmd_data.get("recommendations", [])
        for rec in recommendations:
            card_name = rec.get("card_name", "")
            if not card_name:
                continue

            rec_model = CommanderRecommendation(
                commander_id=commander_model.id,
                card_name=card_name,
                scryfall_id=card_lookup.get(card_name),
                inclusion_rate=rec.get("inclusion_rate", 0),
                synergy_score=rec.get("synergy_score", 0),
                num_decks=rec.get("num_decks", 0),
                category=rec.get("category", ""),
            )
            session.add(rec_model)
            result["recommendations_synced"] += 1

        # Fetch and insert average deck
        avg_deck = client.get_average_deck(cmd_name, use_cache=not force)
        for slot_num, card_name in enumerate(avg_deck):
            avg_model = AverageDeckCard(
                commander_id=commander_model.id,
                card_name=card_name,
                scryfall_id=card_lookup.get(card_name),
                slot_number=slot_num,
            )
            session.add(avg_model)
            result["average_deck_cards"] += 1

        result["commanders_synced"] += 1
        session.commit()

    # Step 3: Fetch global salt scores
    print("\nFetching salt scores...")
    salt_scores = client.get_salt_scores(use_cache=not force)
    print(f"  Found {len(salt_scores)} salt scores")

    # Clear existing salt scores
    session.query(CardSaltScore).delete()

    # Insert salt scores with rank
    sorted_salt = sorted(salt_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (card_name, salt_score) in enumerate(sorted_salt, 1):
        salt_model = CardSaltScore(
            card_name=card_name,
            scryfall_id=card_lookup.get(card_name),
            salt_score=salt_score,
            salt_rank=rank,
        )
        session.add(salt_model)
        result["salt_scores_synced"] += 1

    # Record sync metadata
    metadata = EDHRecSyncMetadata(
        last_updated=datetime.utcnow(),
        commanders_synced=result["commanders_synced"],
        cards_synced=result["recommendations_synced"],
        salt_cards_synced=result["salt_scores_synced"],
    )
    session.add(metadata)

    session.commit()
    session.close()

    print("\nEDHREC sync complete!")
    print(f"  Commanders: {result['commanders_synced']}")
    print(f"  Recommendations: {result['recommendations_synced']:,}")
    print(f"  Average deck cards: {result['average_deck_cards']:,}")
    print(f"  Salt scores: {result['salt_scores_synced']:,}")

    return result


def get_edhrec_stats(config: DatabaseConfig | None = None) -> dict:
    """
    Get statistics about the EDHREC data in the database.

    Returns:
        Dict with EDHREC statistics
    """
    try:
        manager = DatabaseManager(config) if config else DatabaseManager()
        session = manager.session()

        stats = {"exists": True, "edhrec_initialized": True}

        # Commander count
        stats["commanders"] = session.query(CommanderModel).count()

        # Recommendations count
        stats["recommendations"] = session.query(CommanderRecommendation).count()

        # Average deck cards
        stats["average_deck_cards"] = session.query(AverageDeckCard).count()

        # Salt scores count
        stats["salt_scores"] = session.query(CardSaltScore).count()

        # Top 5 commanders by deck count
        top_commanders = (
            session.query(CommanderModel).order_by(CommanderModel.num_decks.desc()).limit(5).all()
        )
        stats["top_commanders"] = [
            {"name": c.name, "num_decks": c.num_decks, "rank": c.edhrec_rank}
            for c in top_commanders
        ]

        # Highest salt cards
        top_salt = (
            session.query(CardSaltScore).order_by(CardSaltScore.salt_score.desc()).limit(5).all()
        )
        stats["highest_salt"] = [
            {"name": s.card_name, "salt": s.salt_score, "rank": s.salt_rank} for s in top_salt
        ]

        # Sync metadata
        metadata = session.query(EDHRecSyncMetadata).order_by(EDHRecSyncMetadata.id.desc()).first()
        if metadata:
            stats["last_updated"] = (
                metadata.last_updated.isoformat() if metadata.last_updated else None
            )
            stats["last_commanders_synced"] = metadata.commanders_synced

        session.close()
        return stats

    except Exception as e:
        return {"exists": False, "edhrec_initialized": False, "error": str(e)}


def get_commander_recommendations(
    commander_name: str,
    config: DatabaseConfig | None = None,
    limit: int = 50,
    min_synergy: float | None = None,
    category: str | None = None,
) -> list[dict]:
    """
    Get card recommendations for a commander from the database.

    Args:
        commander_name: Name of the commander
        config: Database configuration
        limit: Maximum results to return
        min_synergy: Minimum synergy score filter
        category: Filter by category (e.g., "Creatures", "Ramp")

    Returns:
        List of recommendation dicts
    """
    try:
        manager = DatabaseManager(config) if config else DatabaseManager()
        session = manager.session()

        # Find commander
        commander = (
            session.query(CommanderModel)
            .filter(
                (CommanderModel.name == commander_name)
                | (CommanderModel.name.ilike(f"%{commander_name}%"))
            )
            .first()
        )

        if not commander:
            session.close()
            return []

        # Build query
        query = session.query(CommanderRecommendation).filter_by(commander_id=commander.id)

        if min_synergy is not None:
            query = query.filter(CommanderRecommendation.synergy_score >= min_synergy)

        if category:
            query = query.filter(CommanderRecommendation.category.ilike(f"%{category}%"))

        results = query.order_by(CommanderRecommendation.synergy_score.desc()).limit(limit).all()

        recommendations = [
            {
                "card_name": r.card_name,
                "inclusion_rate": r.inclusion_rate,
                "synergy_score": r.synergy_score,
                "num_decks": r.num_decks,
                "category": r.category,
            }
            for r in results
        ]

        session.close()
        return recommendations

    except Exception:
        return []


def get_salt_scores_from_db(config: DatabaseConfig | None = None) -> dict[str, float]:
    """
    Get all salt scores from the database.

    Returns:
        Dict mapping card names to salt scores
    """
    try:
        manager = DatabaseManager(config) if config else DatabaseManager()
        session = manager.session()

        results = session.query(CardSaltScore).all()
        scores = {r.card_name: r.salt_score for r in results}

        session.close()
        return scores

    except Exception:
        return {}
