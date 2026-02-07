"""
Pydantic models for API requests and responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================


class RecommendCardsRequest(BaseModel):
    """Request for card recommendations."""

    commander: str = Field(..., description="Commander name")
    count: int = Field(default=20, ge=1, le=100, description="Number of cards to recommend")
    exclude: list[str] = Field(default_factory=list, description="Cards to exclude")
    budget: Optional[str] = Field(
        default=None,
        description="Budget tier: 'budget', 'medium', 'high', or None for all",
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Filter by categories: 'Ramp', 'Card Draw', 'Removal', etc.",
    )


class AnalyzeDeckRequest(BaseModel):
    """Request for deck analysis."""

    commander: str = Field(..., description="Commander name")
    decklist: list[str] = Field(..., min_length=1, description="List of card names in deck")
    num_simulations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of goldfish simulations to run",
    )
    max_turns: int = Field(default=10, ge=1, le=20, description="Max turns per simulation")


class SynergyRequest(BaseModel):
    """Request for synergy analysis."""

    commander: str = Field(..., description="Commander name")
    cards: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Cards to analyze for synergy",
    )


class SimulateRequest(BaseModel):
    """Request for goldfish simulation."""

    commander: Optional[str] = Field(default=None, description="Commander name (optional)")
    decklist: list[str] = Field(..., min_length=40, description="List of card names")
    num_games: int = Field(default=10, ge=1, le=100, description="Number of games to simulate")
    max_turns: int = Field(default=15, ge=1, le=30, description="Max turns per game")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


# ============================================================================
# Response Models
# ============================================================================


class CardRecommendation(BaseModel):
    """A single card recommendation."""

    name: str
    synergy_score: float = Field(..., description="Synergy with commander (0-1)")
    inclusion_rate: float = Field(..., description="% of decks that include this card")
    category: Optional[str] = Field(default=None, description="Card category/role")
    cmc: float = Field(..., description="Mana value")
    type_line: Optional[str] = Field(default=None, description="Card type")
    reason: Optional[str] = Field(default=None, description="Why this card is recommended")


class RecommendCardsResponse(BaseModel):
    """Response with card recommendations."""

    commander: str
    recommendations: list[CardRecommendation]
    total_available: int = Field(..., description="Total cards available for this commander")


class CardPerformance(BaseModel):
    """Performance metrics for a single card."""

    name: str
    times_drawn: int
    times_played: int
    play_rate: float = Field(..., description="% of times played when drawn")
    avg_turn_played: Optional[float] = Field(default=None, description="Average turn played")
    synergy_score: float = Field(default=0.0)
    category: Optional[str] = None


class DeckAnalysis(BaseModel):
    """Analysis results for a deck."""

    # Basic stats
    total_cards: int
    avg_cmc: float
    land_count: int
    creature_count: int

    # Mana curve
    cmc_distribution: dict[str, int] = Field(
        ..., description="Cards at each CMC (0-7+)"
    )

    # Role distribution
    role_distribution: dict[str, int] = Field(
        ..., description="Cards in each role category"
    )

    # Simulation results
    avg_damage: float = Field(..., description="Average damage dealt in goldfish")
    avg_turns_to_kill: float = Field(..., description="Average turns to deal 40 damage")
    avg_cards_played: float
    avg_lands_played: float


class DeckSuggestion(BaseModel):
    """A suggestion for deck improvement."""

    action: str = Field(..., description="'cut' or 'add'")
    card: str
    reason: str
    priority: str = Field(..., description="'high', 'medium', or 'low'")


class AnalyzeDeckResponse(BaseModel):
    """Response with deck analysis."""

    commander: str
    analysis: DeckAnalysis
    card_performance: list[CardPerformance]
    suggestions: list[DeckSuggestion]
    missing_cards: list[str] = Field(
        default_factory=list,
        description="Cards in decklist not found in database",
    )


class CardSynergy(BaseModel):
    """Synergy between two cards."""

    card1: str
    card2: str
    synergy_score: float = Field(..., description="Combined synergy score")
    reason: Optional[str] = None


class SynergyResponse(BaseModel):
    """Response with synergy analysis."""

    commander: str
    cards: list[str]
    average_synergy: float
    card_synergies: dict[str, float] = Field(
        ..., description="Individual synergy scores with commander"
    )
    pair_synergies: list[CardSynergy] = Field(
        default_factory=list,
        description="Synergy between card pairs",
    )


class SimulationResult(BaseModel):
    """Result of a single simulation."""

    game_id: int
    turns: int
    total_damage: int
    cards_played: int
    lands_played: int
    won: bool = Field(..., description="Dealt 40+ damage")


class SimulateResponse(BaseModel):
    """Response with simulation results."""

    num_games: int
    results: list[SimulationResult]

    # Aggregates
    avg_damage: float
    avg_turns: float
    avg_cards_played: float
    win_rate: float = Field(..., description="% of games dealing 40+ damage")

    # Per-turn breakdown
    damage_by_turn: list[float] = Field(
        ..., description="Average damage dealt on each turn"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database_connected: bool
    cards_loaded: int
    commanders_available: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
