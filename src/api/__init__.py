"""
API package for serving Manasink ML predictions.

Provides FastAPI endpoints for:
- Card recommendations based on commander
- Deck analysis with simulation
- Synergy scoring between cards
- Goldfish simulation

Quick start:
    # Run the API server
    uvicorn src.api.app:app --reload

    # Or from command line
    python -m src.api.app

Endpoints:
    GET  /health           - Health check
    GET  /cache/stats      - Cache statistics
    POST /cache/clear      - Clear cache
    GET  /commanders       - List available commanders
    POST /recommend/cards  - Get card recommendations
    POST /analyze/deck     - Analyze a deck
    POST /analyze/synergy  - Get synergy scores
    POST /simulate/goldfish - Run goldfish simulation

Caching:
    Recommendations and synergy scores are cached in Redis when available.
    Set REDIS_URL environment variable to enable caching.
"""

from .app import app, create_app
from .cache import (
    cache,
    get_cache_stats,
    invalidate_all_cache,
    invalidate_commander_cache,
)
from .models import (
    AnalyzeDeckRequest,
    AnalyzeDeckResponse,
    CardPerformance,
    CardRecommendation,
    CardSynergy,
    DeckAnalysis,
    DeckSuggestion,
    ErrorResponse,
    HealthResponse,
    # Requests
    RecommendCardsRequest,
    # Responses
    RecommendCardsResponse,
    SimulateRequest,
    SimulateResponse,
    SimulationResult,
    SynergyRequest,
    SynergyResponse,
)
from .services import (
    analyze_deck,
    get_card_recommendations,
    get_health_status,
    get_synergy_scores,
    list_commanders,
    run_simulation,
)

__all__ = [
    # App
    "app",
    "create_app",
    # Request models
    "RecommendCardsRequest",
    "AnalyzeDeckRequest",
    "SynergyRequest",
    "SimulateRequest",
    # Response models
    "RecommendCardsResponse",
    "CardRecommendation",
    "AnalyzeDeckResponse",
    "DeckAnalysis",
    "CardPerformance",
    "DeckSuggestion",
    "SynergyResponse",
    "CardSynergy",
    "SimulateResponse",
    "SimulationResult",
    "HealthResponse",
    "ErrorResponse",
    # Services
    "get_card_recommendations",
    "analyze_deck",
    "get_synergy_scores",
    "run_simulation",
    "get_health_status",
    "list_commanders",
    # Caching
    "cache",
    "invalidate_commander_cache",
    "invalidate_all_cache",
    "get_cache_stats",
]
