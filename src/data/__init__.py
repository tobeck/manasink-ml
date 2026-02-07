"""
Data ingestion and processing package.

This package provides:
- ScryfallClient: Direct API access to Scryfall (for ad-hoc queries)
- CardDatabase: Database-backed card storage (SQLite or PostgreSQL)
- sync_database: Download and populate the card database
- EDHRecClient: EDHREC API client for commander recommendations
- sync_edhrec_data: Fetch EDHREC data and populate database

Database Configuration:
    Set DATABASE_URL environment variable to use PostgreSQL:
        DATABASE_URL=postgresql://user:pass@host:5432/manasink

    Or use individual settings:
        DB_TYPE=postgresql
        DB_HOST=localhost
        DB_PORT=5432
        DB_NAME=manasink
        DB_USER=postgres
        DB_PASSWORD=secret

    Default is SQLite at data/cards.db

Quick start:
    # First, sync the database (downloads ~30MB)
    from src.data import sync_database
    sync_database()

    # Then query cards
    from src.data import CardDatabase
    db = CardDatabase()
    card = db.get_card("Sol Ring")
    commanders = db.get_commanders(colors="UG")

    # Sync EDHREC data for recommendations
    from src.data import sync_edhrec_data
    sync_edhrec_data(limit=100)

    # Get commander recommendations
    from src.data import get_commander_recommendations
    recs = get_commander_recommendations("Atraxa, Praetors' Voice")
"""

from .db_config import DatabaseConfig, DatabaseManager, get_db_session
from .scryfall import (
    ScryfallClient,
    fetch_card,
    fetch_commander,
)
from .database import CardDatabase
from .ingest import sync_database, get_database_stats
from .edhrec import (
    EDHRecClient,
    fetch_commander_recommendations,
    fetch_salt_scores,
)
from .edhrec_ingest import (
    sync_edhrec_data,
    get_edhrec_stats,
    get_commander_recommendations,
    get_salt_scores_from_db,
    estimate_deck_power,
    PowerLevelEstimate,
)
from .features import (
    CardFeatures,
    extract_features_from_scryfall,
    populate_card_features,
    get_feature_vector,
    get_batch_features,
    get_features_stats,
)
from .categories import (
    populate_card_categories,
    compute_role_scores,
    get_card_categories,
    get_cards_by_category,
    get_top_cards_per_role,
    get_categories_stats,
)
from .deck_loader import (
    DeckLoadResult,
    SynergyData,
    load_deck_from_edhrec,
    load_synergy_data,
    load_deck_with_synergy_data,
    list_available_commanders,
    get_deck_stats,
)

__all__ = [
    # Database configuration
    "DatabaseConfig",
    "DatabaseManager",
    "get_db_session",
    # Scryfall API client
    "ScryfallClient",
    "fetch_card",
    "fetch_commander",
    # Database (SQLite or PostgreSQL)
    "CardDatabase",
    # Scryfall ingestion
    "sync_database",
    "get_database_stats",
    # EDHREC API client
    "EDHRecClient",
    "fetch_commander_recommendations",
    "fetch_salt_scores",
    # EDHREC ingestion
    "sync_edhrec_data",
    "get_edhrec_stats",
    "get_commander_recommendations",
    "get_salt_scores_from_db",
    # Power level estimation
    "estimate_deck_power",
    "PowerLevelEstimate",
    # Features
    "CardFeatures",
    "extract_features_from_scryfall",
    "populate_card_features",
    "get_feature_vector",
    "get_batch_features",
    "get_features_stats",
    # Categories
    "populate_card_categories",
    "compute_role_scores",
    "get_card_categories",
    "get_cards_by_category",
    "get_top_cards_per_role",
    "get_categories_stats",
    # Deck loading
    "DeckLoadResult",
    "SynergyData",
    "load_deck_from_edhrec",
    "load_synergy_data",
    "load_deck_with_synergy_data",
    "list_available_commanders",
    "get_deck_stats",
]
