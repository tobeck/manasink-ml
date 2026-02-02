"""
Data ingestion module for downloading and processing Scryfall bulk data.

This module handles:
- Downloading bulk card data from Scryfall
- Checking for updates
- Populating the SQLite database

Usage:
    from src.data.ingest import sync_database
    sync_database()  # Downloads data and populates DB

Or via CLI:
    manasink-data sync
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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

from .database import DEFAULT_DB_PATH, create_schema


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


def needs_update(bulk_path: Path, db_path: Path) -> tuple[bool, Optional[str]]:
    """
    Check if we need to download new data.

    Args:
        bulk_path: Path to the bulk JSON file
        db_path: Path to the SQLite database

    Returns:
        Tuple of (needs_update, scryfall_updated_at)
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required")

    # If no database exists, we need to sync
    if not db_path.exists():
        info = get_bulk_data_info()
        return True, info.get("updated_at")

    # Check Scryfall's last update time
    info = get_bulk_data_info()
    scryfall_updated = info.get("updated_at")

    # Check our last sync time
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT scryfall_updated_at FROM sync_metadata ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row or not row["scryfall_updated_at"]:
        return True, scryfall_updated

    our_updated = row["scryfall_updated_at"]

    # Compare timestamps
    return scryfall_updated != our_updated, scryfall_updated


def download_bulk_data(
    output_path: Optional[Path] = None,
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

    print(f"Downloading Scryfall oracle cards data...")
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
    bulk_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    scryfall_updated_at: Optional[str] = None,
    show_progress: bool = True,
) -> int:
    """
    Populate the SQLite database from bulk JSON data.

    Args:
        bulk_path: Path to the bulk JSON file
        db_path: Path to the SQLite database
        scryfall_updated_at: Scryfall's update timestamp
        show_progress: Show progress bar

    Returns:
        Number of cards inserted
    """
    bulk_path = bulk_path or DEFAULT_BULK_PATH
    db_path = db_path or DEFAULT_DB_PATH

    if not bulk_path.exists():
        raise FileNotFoundError(f"Bulk data not found at {bulk_path}")

    print(f"Loading bulk data from {bulk_path}...")

    with open(bulk_path) as f:
        cards_data = json.load(f)

    print(f"Found {len(cards_data)} cards")

    # Create fresh database (drop existing if any)
    if db_path.exists():
        db_path.unlink()

    conn = create_schema(db_path)

    print("Populating database...")

    if show_progress and HAS_TQDM:
        iterator = tqdm(cards_data, desc="Inserting cards")
    else:
        iterator = cards_data

    inserted = 0
    batch = []
    batch_size = 1000

    for card_data in iterator:
        row = _extract_card_row(card_data)
        if row:
            batch.append(row)
            inserted += 1

            if len(batch) >= batch_size:
                _insert_batch(conn, batch)
                batch = []

    # Insert remaining
    if batch:
        _insert_batch(conn, batch)

    # Record sync metadata
    conn.execute(
        """
        INSERT INTO sync_metadata (last_updated, card_count, scryfall_updated_at)
        VALUES (?, ?, ?)
        """,
        (datetime.now().isoformat(), inserted, scryfall_updated_at),
    )

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} cards into {db_path}")
    return inserted


def _extract_card_row(card_data: dict) -> Optional[tuple]:
    """Extract database row from Scryfall card data."""
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
        is_commander = 1 if ("legendary" in type_lower and "creature" in type_lower) else 0

        # Also check for cards that say "can be your commander"
        if "can be your commander" in oracle_text.lower():
            is_commander = 1

        # Check commander legality
        legalities = card_data.get("legalities", {})
        legal_commander = 1 if legalities.get("commander") == "legal" else 0

        return (
            scryfall_id,
            name,
            name.lower(),
            mana_cost,
            cmc,
            type_line,
            colors,
            color_identity,
            oracle_text,
            oracle_text.lower(),
            power,
            toughness,
            is_commander,
            legal_commander,
            rarity,
            set_code,
            json.dumps(card_data),
        )
    except Exception:
        return None


def _insert_batch(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    """Insert a batch of card rows."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO cards (
            scryfall_id, name, name_lower, mana_cost, cmc, type_line,
            colors, color_identity, oracle_text, oracle_text_lower,
            power, toughness, is_commander, legal_commander,
            rarity, set_code, scryfall_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def sync_database(
    bulk_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    force: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Full sync: check for updates, download if needed, populate database.

    Args:
        bulk_path: Path for bulk JSON file
        db_path: Path for SQLite database
        force: Force re-download even if up to date
        show_progress: Show progress bars

    Returns:
        Dict with sync results
    """
    bulk_path = bulk_path or DEFAULT_BULK_PATH
    db_path = db_path or DEFAULT_DB_PATH

    result = {
        "downloaded": False,
        "cards_inserted": 0,
        "skipped": False,
        "scryfall_updated_at": None,
    }

    # Check if update needed
    if not force:
        needs, scryfall_updated = needs_update(bulk_path, db_path)
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
        db_path,
        scryfall_updated_at=result["scryfall_updated_at"],
        show_progress=show_progress,
    )
    result["cards_inserted"] = cards

    print(f"\nSync complete!")
    print(f"  Cards: {cards}")
    print(f"  Database: {db_path}")

    return result


def get_database_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get statistics about the card database.

    Returns:
        Dict with database statistics
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {"exists": True, "path": str(db_path)}

    # Total cards
    cursor = conn.execute("SELECT COUNT(*) as count FROM cards")
    stats["total_cards"] = cursor.fetchone()["count"]

    # Commander-legal cards
    cursor = conn.execute("SELECT COUNT(*) as count FROM cards WHERE legal_commander = 1")
    stats["commander_legal"] = cursor.fetchone()["count"]

    # Potential commanders
    cursor = conn.execute(
        "SELECT COUNT(*) as count FROM cards WHERE is_commander = 1 AND legal_commander = 1"
    )
    stats["commanders"] = cursor.fetchone()["count"]

    # By card type
    for card_type in ["creature", "instant", "sorcery", "artifact", "enchantment", "land", "planeswalker"]:
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM cards WHERE type_line LIKE ?",
            (f"%{card_type}%",)
        )
        stats[f"type_{card_type}"] = cursor.fetchone()["count"]

    # Sync metadata
    cursor = conn.execute("SELECT * FROM sync_metadata ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        stats["last_updated"] = row["last_updated"]
        stats["scryfall_updated_at"] = row["scryfall_updated_at"]

    # File size
    stats["file_size_mb"] = db_path.stat().st_size / 1024 / 1024

    conn.close()
    return stats
