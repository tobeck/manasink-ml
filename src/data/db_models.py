"""
SQLAlchemy ORM models for Manasink ML database.

These models support both SQLite and PostgreSQL backends.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# =============================================================================
# Card Data Models
# =============================================================================


class CardModel(Base):
    """
    Main cards table storing Scryfall card data.

    Stores full Scryfall JSON alongside indexed columns for efficient queries.
    """

    __tablename__ = "cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scryfall_id = Column(String(36), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    name_lower = Column(String(255), nullable=False, index=True)
    mana_cost = Column(String(100))
    cmc = Column(Float, default=0, index=True)
    type_line = Column(String(255), index=True)
    colors = Column(String(10))  # Sorted color string (e.g., "BU")
    color_identity = Column(String(10), index=True)  # Sorted color identity
    oracle_text = Column(Text)
    oracle_text_lower = Column(Text)
    power = Column(String(10))
    toughness = Column(String(10))
    is_commander = Column(Boolean, default=False, index=True)
    legal_commander = Column(Boolean, default=False, index=True)
    rarity = Column(String(20))
    set_code = Column(String(10))
    scryfall_json = Column(Text, nullable=False)  # Full JSON for flexibility

    def __repr__(self):
        return f"<Card {self.name}>"


class SyncMetadata(Base):
    """Metadata about Scryfall data syncs."""

    __tablename__ = "sync_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    card_count = Column(Integer, default=0)
    scryfall_updated_at = Column(String(50))


# =============================================================================
# EDHREC Data Models
# =============================================================================


class CommanderModel(Base):
    """
    Commanders table with EDHREC popularity data.

    Links to the cards table via scryfall_id and stores EDHREC-specific
    metrics like deck count, rank, and salt score.
    """

    __tablename__ = "commanders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    name_slug = Column(String(255), unique=True, nullable=False, index=True)
    scryfall_id = Column(String(36), ForeignKey("cards.scryfall_id"))
    edhrec_rank = Column(Integer, index=True)
    num_decks = Column(Integer, default=0)
    salt_score = Column(Float)
    color_identity = Column(String(10))
    last_synced = Column(DateTime)
    edhrec_json = Column(Text)

    # Relationships
    recommendations = relationship(
        "CommanderRecommendation",
        back_populates="commander",
        cascade="all, delete-orphan",
    )
    average_deck_cards = relationship(
        "AverageDeckCard",
        back_populates="commander",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Commander {self.name}>"


class CommanderRecommendation(Base):
    """
    Card recommendations per commander from EDHREC.

    Stores synergy scores, inclusion rates, and categories for each
    card recommendation.
    """

    __tablename__ = "commander_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    commander_id = Column(Integer, ForeignKey("commanders.id"), nullable=False, index=True)
    card_name = Column(String(255), nullable=False)
    scryfall_id = Column(String(36))
    inclusion_rate = Column(Float, default=0)
    synergy_score = Column(Float, default=0, index=True)
    num_decks = Column(Integer, default=0)
    category = Column(String(100))

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("commander_id", "card_name", name="uq_commander_card"),
        Index("idx_recs_inclusion", "inclusion_rate"),
    )

    # Relationships
    commander = relationship("CommanderModel", back_populates="recommendations")

    def __repr__(self):
        return f"<Recommendation {self.card_name} for commander_id={self.commander_id}>"


class AverageDeckCard(Base):
    """
    Average decklist cards (consensus 99) from EDHREC.

    Stores the "average deck" for each commander - the most commonly
    played cards in that commander's decks.
    """

    __tablename__ = "average_decks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    commander_id = Column(Integer, ForeignKey("commanders.id"), nullable=False, index=True)
    card_name = Column(String(255), nullable=False)
    scryfall_id = Column(String(36))
    slot_number = Column(Integer)

    # Unique constraint
    __table_args__ = (UniqueConstraint("commander_id", "card_name", name="uq_avgdeck_card"),)

    # Relationships
    commander = relationship("CommanderModel", back_populates="average_deck_cards")

    def __repr__(self):
        return f"<AverageDeckCard {self.card_name}>"


class CardSaltScore(Base):
    """
    Global card salt scores from EDHREC.

    Salt scores indicate how "annoying" or powerful a card is perceived
    by the community - useful for power level estimation.
    """

    __tablename__ = "card_salt_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_name = Column(String(255), unique=True, nullable=False, index=True)
    scryfall_id = Column(String(36))
    salt_score = Column(Float, nullable=False, index=True)
    salt_rank = Column(Integer)

    def __repr__(self):
        return f"<CardSaltScore {self.card_name}: {self.salt_score}>"


class EDHRecSyncMetadata(Base):
    """Metadata about EDHREC data syncs."""

    __tablename__ = "edhrec_sync_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    commanders_synced = Column(Integer, default=0)
    cards_synced = Column(Integer, default=0)
    salt_cards_synced = Column(Integer, default=0)


# =============================================================================
# ML Feature Models
# =============================================================================


class CardFeatureModel(Base):
    """
    ML-ready features extracted from Scryfall card data.

    Stores the 29-dimensional feature vector components for each card,
    used for training models.
    """

    __tablename__ = "card_features"

    scryfall_id = Column(String(36), primary_key=True)
    card_name = Column(String(255), nullable=False, index=True)

    # Mana vector (WUBRGC counts)
    mana_w = Column(Integer, default=0)
    mana_u = Column(Integer, default=0)
    mana_b = Column(Integer, default=0)
    mana_r = Column(Integer, default=0)
    mana_g = Column(Integer, default=0)
    mana_c = Column(Integer, default=0)
    cmc = Column(Float, default=0, index=True)

    # Type flags
    is_creature = Column(Boolean, default=False)
    is_instant = Column(Boolean, default=False)
    is_sorcery = Column(Boolean, default=False)
    is_artifact = Column(Boolean, default=False)
    is_enchantment = Column(Boolean, default=False)
    is_planeswalker = Column(Boolean, default=False)
    is_land = Column(Boolean, default=False)
    is_legendary = Column(Boolean, default=False)

    # Keywords as bitmap
    keyword_bitmap = Column(Integer, default=0)

    # Creature stats
    power = Column(Integer)
    toughness = Column(Integer)

    # Color identity as 5-bit WUBRG bitmap
    color_identity_bitmap = Column(Integer, default=0)

    # Role scores (0.0 to 1.0)
    role_ramp = Column(Float, default=0)
    role_card_draw = Column(Float, default=0)
    role_removal = Column(Float, default=0)
    role_board_wipe = Column(Float, default=0)
    role_protection = Column(Float, default=0)
    role_finisher = Column(Float, default=0)
    role_utility = Column(Float, default=0)

    def __repr__(self):
        return f"<CardFeature {self.card_name}>"


class CardCategory(Base):
    """
    Card categories aggregated from EDHREC recommendations.

    Stores how often each card appears in each category across all
    commanders, useful for determining functional roles.
    """

    __tablename__ = "card_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    total_occurrences = Column(Integer, default=0)
    commander_count = Column(Integer, default=0)
    avg_synergy_score = Column(Float, default=0)
    avg_inclusion_rate = Column(Float, default=0)

    # Unique constraint
    __table_args__ = (UniqueConstraint("card_name", "category", name="uq_card_category"),)

    def __repr__(self):
        return f"<CardCategory {self.card_name}: {self.category}>"
