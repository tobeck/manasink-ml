"""
Data ingestion and processing package.

This package provides:
- ScryfallClient: Direct API access to Scryfall (for ad-hoc queries)
- CardDatabase: SQLite-backed card storage (for bulk operations)
- sync_database: Download and populate the card database

Quick start:
    # First, sync the database (downloads ~30MB)
    from src.data import sync_database
    sync_database()

    # Then query cards
    from src.data import CardDatabase
    db = CardDatabase()
    card = db.get_card("Sol Ring")
    commanders = db.get_commanders(colors="UG")
"""

from .scryfall import (
    ScryfallClient,
    fetch_card,
    fetch_commander,
)
from .database import CardDatabase
from .ingest import sync_database, get_database_stats

__all__ = [
    # Scryfall API client
    "ScryfallClient",
    "fetch_card",
    "fetch_commander",
    # SQLite database
    "CardDatabase",
    # Ingestion
    "sync_database",
    "get_database_stats",
]
