"""
Data ingestion module for downloading and processing Scryfall bulk data.

This module handles:
- Downloading bulk card data from Scryfall
- Checking for updates
- Populating the database (SQLite or PostgreSQL)

Usage:
    from src.data.ingest import sync_database
    sync_database()  # Downloads data and populates DB

Or via CLI:
    manasink-data sync
"""

import json
from datetime import datetime
from pathlib import Path

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


from .db_config import DatabaseConfig, DatabaseManager
from .db_models import CardModel, SyncMetadata

SCRYFALL_API_BASE = "https://api.scryfall.com"
DEFAULT_BULK_PATH = Path("data/raw/scryfall_bulk.json")


def get_bulk_data_info() -> dict:
    """
    Fetch metadata about Scryfall's bulk data files.

    Returns:
        Dict with bulk data info including download_uri and updated_at
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required for Scryfall API access")

    response = requests.get(f"{SCRYFALL_API_BASE}/bulk-data")
    response.raise_for_status()

    bulk_data = response.json()

    # Find the oracle_cards bulk data (one entry per unique card name)
    for item in bulk_data["data"]:
        if item["type"] == "oracle_cards":
            return item

    raise ValueError("Could not find oracle_cards bulk data")


def needs_update(
    bulk_path: Path,
    config: DatabaseConfig | None = None,
) -> tuple[bool, str | None]:
    """
    Check if we need to download new data.

    Args:
        bulk_path: Path to the bulk JSON file
        config: Database configuration

    Returns:
        Tuple of (needs_update, scryfall_updated_at)
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required")

    # Check Scryfall's last update time
    info = get_bulk_data_info()
    scryfall_updated = info.get("updated_at")

    # Check if we have any data in the database
    try:
        manager = DatabaseManager(config) if config else DatabaseManager()
        session = manager.session()

        result = session.query(SyncMetadata).order_by(SyncMetadata.id.desc()).first()

        session.close()

        if not result or not result.scryfall_updated_at:
            return True, scryfall_updated

        # Compare timestamps
        return scryfall_updated != result.scryfall_updated_at, scryfall_updated

    except Exception:
        # Database doesn't exist or tables not created
        return True, scryfall_updated


def download_bulk_data(
    output_path: Path | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Download Scryfall's bulk data file.

    Args:
        output_path: Where to save the file
        show_progress: Show download progress bar

    Returns:
        Path to the downloaded file
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required")

    output_path = output_path or DEFAULT_BULK_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = get_bulk_data_info()
    download_url = info["download_uri"]
    file_size = info.get("size", 0)

    print("Downloading Scryfall oracle cards data...")
    print(f"  URL: {download_url}")
    print(f"  Size: {file_size / 1024 / 1024:.1f} MB")

    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    # Get actual file size from headers if available
    total_size = int(response.headers.get("content-length", file_size))

    if show_progress and HAS_TQDM:
        progress = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        )
    else:
        progress = None

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            if progress:
                progress.update(len(chunk))

    if progress:
        progress.close()

    print(f"Downloaded to {output_path}")
    return output_path


def populate_database(
    bulk_path: Path | None = None,
    config: DatabaseConfig | None = None,
    scryfall_updated_at: str | None = None,
    show_progress: bool = True,
    drop_existing: bool = True,
) -> int:
    """
    Populate the database from bulk JSON data.

    Args:
        bulk_path: Path to the bulk JSON file
        config: Database configuration
        scryfall_updated_at: Scryfall's update timestamp
        show_progress: Show progress bar
        drop_existing: Drop existing data before inserting (default True)

    Returns:
        Number of cards inserted
    """
    bulk_path = bulk_path or DEFAULT_BULK_PATH

    if not bulk_path.exists():
        raise FileNotFoundError(f"Bulk data not found at {bulk_path}")

    print(f"Loading bulk data from {bulk_path}...")

    with open(bulk_path) as f:
        cards_data = json.load(f)

    print(f"Found {len(cards_data)} cards")

    # Set up database
    manager = DatabaseManager(config) if config else DatabaseManager()

    if drop_existing:
        print("Dropping existing card data...")
        # Create tables if they don't exist
        manager.create_tables()
        # Clear existing cards
        session = manager.session()
        session.query(CardModel).delete()
        session.query(SyncMetadata).delete()
        session.commit()
        session.close()

    print("Populating database...")

    if show_progress and HAS_TQDM:
        iterator = tqdm(cards_data, desc="Inserting cards")
    else:
        iterator = cards_data

    session = manager.session()
    inserted = 0
    batch = []
    batch_size = 1000

    for card_data in iterator:
        card_model = _extract_card_model(card_data)
        if card_model:
            batch.append(card_model)
            inserted += 1

            if len(batch) >= batch_size:
                session.bulk_save_objects(batch)
                session.commit()
                batch = []

    # Insert remaining
    if batch:
        session.bulk_save_objects(batch)
        session.commit()

    # Record sync metadata
    metadata = SyncMetadata(
        last_updated=datetime.utcnow(),
        card_count=inserted,
        scryfall_updated_at=scryfall_updated_at,
    )
    session.add(metadata)
    session.commit()
    session.close()

    print(f"Inserted {inserted} cards")
    return inserted


def _extract_card_model(card_data: dict) -> CardModel | None:
    """Extract CardModel from Scryfall card data."""
    try:
        scryfall_id = card_data.get("id")
        if not scryfall_id:
            return None

        name = card_data.get("name", "Unknown")
        mana_cost = card_data.get("mana_cost", "")
        cmc = card_data.get("cmc", 0)
        type_line = card_data.get("type_line", "")
        oracle_text = card_data.get("oracle_text", "")
        power = card_data.get("power")
        toughness = card_data.get("toughness")
        rarity = card_data.get("rarity")
        set_code = card_data.get("set")

        # Colors (sorted string)
        colors = "".join(sorted(card_data.get("colors", [])))

        # Color identity (sorted string)
        color_identity = "".join(sorted(card_data.get("color_identity", [])))

        # Is this a potential commander?
        type_lower = type_line.lower()
        is_commander = "legendary" in type_lower and "creature" in type_lower

        # Also check for cards that say "can be your commander"
        if "can be your commander" in oracle_text.lower():
            is_commander = True

        # Check commander legality
        legalities = card_data.get("legalities", {})
        legal_commander = legalities.get("commander") == "legal"

        return CardModel(
            scryfall_id=scryfall_id,
            name=name,
            name_lower=name.lower(),
            mana_cost=mana_cost,
            cmc=cmc,
            type_line=type_line,
            colors=colors,
            color_identity=color_identity,
            oracle_text=oracle_text,
            oracle_text_lower=oracle_text.lower(),
            power=power,
            toughness=toughness,
            is_commander=is_commander,
            legal_commander=legal_commander,
            rarity=rarity,
            set_code=set_code,
            scryfall_json=json.dumps(card_data),
        )
    except Exception:
        return None


def sync_database(
    bulk_path: Path | None = None,
    config: DatabaseConfig | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Full sync: check for updates, download if needed, populate database.

    Args:
        bulk_path: Path for bulk JSON file
        config: Database configuration
        force: Force re-download even if up to date
        show_progress: Show progress bars

    Returns:
        Dict with sync results
    """
    bulk_path = bulk_path or DEFAULT_BULK_PATH

    result = {
        "downloaded": False,
        "cards_inserted": 0,
        "skipped": False,
        "scryfall_updated_at": None,
    }

    # Check if update needed
    if not force:
        needs, scryfall_updated = needs_update(bulk_path, config)
        result["scryfall_updated_at"] = scryfall_updated

        if not needs:
            print("Database is up to date with Scryfall.")
            result["skipped"] = True
            return result
    else:
        info = get_bulk_data_info()
        result["scryfall_updated_at"] = info.get("updated_at")

    # Download
    download_bulk_data(bulk_path, show_progress=show_progress)
    result["downloaded"] = True

    # Populate
    cards = populate_database(
        bulk_path,
        config,
        scryfall_updated_at=result["scryfall_updated_at"],
        show_progress=show_progress,
    )
    result["cards_inserted"] = cards

    print("\nSync complete!")
    print(f"  Cards: {cards}")

    return result


def get_database_stats(config: DatabaseConfig | None = None) -> dict:
    """
    Get statistics about the card database.

    Returns:
        Dict with database statistics
    """
    try:
        manager = DatabaseManager(config) if config else DatabaseManager()
        session = manager.session()

        stats = {"exists": True}

        # Total cards
        stats["total_cards"] = session.query(CardModel).count()

        # Commander-legal cards
        stats["commander_legal"] = (
            session.query(CardModel).filter(CardModel.legal_commander.is_(True)).count()
        )

        # Potential commanders
        stats["commanders"] = (
            session.query(CardModel)
            .filter(CardModel.is_commander.is_(True), CardModel.legal_commander.is_(True))
            .count()
        )

        # By card type
        for card_type in [
            "creature",
            "instant",
            "sorcery",
            "artifact",
            "enchantment",
            "land",
            "planeswalker",
        ]:
            stats[f"type_{card_type}"] = (
                session.query(CardModel).filter(CardModel.type_line.ilike(f"%{card_type}%")).count()
            )

        # Sync metadata
        result = session.query(SyncMetadata).order_by(SyncMetadata.id.desc()).first()
        if result:
            stats["last_updated"] = result.last_updated.isoformat() if result.last_updated else None
            stats["scryfall_updated_at"] = result.scryfall_updated_at

        session.close()
        return stats

    except Exception as e:
        return {"exists": False, "error": str(e)}
