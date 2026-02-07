"""
FastAPI application for Manasink ML API.

Provides endpoints for:
- Card recommendations based on commander
- Deck analysis with simulation
- Synergy scoring between cards
- Goldfish simulation

API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    AnalyzeDeckRequest,
    AnalyzeDeckResponse,
    ErrorResponse,
    HealthResponse,
    RecommendCardsRequest,
    RecommendCardsResponse,
    SimulateRequest,
    SimulateResponse,
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAPI tag descriptions for Swagger UI organization
tags_metadata = [
    {
        "name": "Health",
        "description": "API health checks and status information.",
    },
    {
        "name": "Info",
        "description": "Browse available commanders and card data.",
    },
    {
        "name": "Recommendations",
        "description": "Get card recommendations for your commander deck.",
    },
    {
        "name": "Analysis",
        "description": "Analyze deck composition, synergies, and performance.",
    },
    {
        "name": "Simulation",
        "description": "Run goldfish simulations to test deck performance.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Manasink ML API...")
    logger.info(f"Registered routes: {[r.path for r in app.routes]}")
    # Startup: could preload models here
    yield
    # Shutdown: cleanup
    logger.info("Shutting down Manasink ML API...")


# Create FastAPI app
app = FastAPI(
    title="Manasink ML API",
    description="""
## MTG Commander Card Recommendations

Machine learning-powered card recommendations for Magic: The Gathering Commander decks.

### Features

- **Card Recommendations**: Get personalized card suggestions based on your commander
- **Deck Analysis**: Analyze your deck's mana curve, role distribution, and performance
- **Synergy Scoring**: Evaluate how well cards work together
- **Goldfish Simulation**: Test deck consistency and damage output

### Getting Started

1. Check `/commanders` to see available commanders with data
2. Use `/recommend/cards` to get recommendations for your commander
3. Analyze your deck with `/analyze/deck` to get improvement suggestions

### Data Sources

- Card data from [Scryfall](https://scryfall.com/)
- Commander recommendations from [EDHREC](https://edhrec.com/)
- Synergy scores derived from deck co-occurrence and simulation
""",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    contact={
        "name": "Manasink",
        "url": "https://github.com/tobeck/manasink-ml",
    },
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and database connectivity."""
    return get_health_status()


@app.get("/cache/stats", tags=["Health"])
async def cache_stats():
    """Get cache statistics."""
    from .cache import get_cache_stats

    return get_cache_stats()


@app.post("/cache/clear", tags=["Health"])
async def cache_clear():
    """Clear all cache entries."""
    from .cache import invalidate_all_cache

    deleted = invalidate_all_cache()
    return {"message": f"Cleared {deleted} cache entries"}


@app.get("/commanders", tags=["Info"])
async def get_commanders(
    color_identity: str | None = Query(
        default=None,
        description="Filter by color identity (e.g., 'UG', 'WBR')",
    ),
    min_decks: int = Query(default=100, description="Minimum deck count"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List available commanders with deck data."""
    try:
        commanders = list_commanders(
            color_identity=color_identity,
            min_decks=min_decks,
            limit=limit,
        )
        return {"commanders": commanders, "count": len(commanders)}
    except Exception as e:
        logger.error(f"Error listing commanders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recommendation Endpoints
# ============================================================================


@app.post(
    "/recommend/cards",
    response_model=RecommendCardsResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
async def recommend_cards(request: RecommendCardsRequest):
    """
    Get card recommendations for a commander.

    Returns cards ranked by synergy score and inclusion rate,
    filtered by optional criteria (budget, categories).
    """
    try:
        result = get_card_recommendations(
            commander=request.commander,
            count=request.count,
            exclude=request.exclude,
            budget=request.budget,
            categories=request.categories,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Commander '{request.commander}' not found or no data available",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analysis Endpoints
# ============================================================================


@app.post(
    "/analyze/deck",
    response_model=AnalyzeDeckResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze_deck_endpoint(request: AnalyzeDeckRequest):
    """
    Analyze a deck and get improvement suggestions.

    Runs goldfish simulations to evaluate card performance
    and suggests cuts/additions based on synergy data.
    """
    try:
        result = analyze_deck(
            commander=request.commander,
            decklist=request.decklist,
            num_simulations=request.num_simulations,
            max_turns=request.max_turns,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Commander '{request.commander}' not found",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing deck: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/analyze/synergy",
    response_model=SynergyResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze_synergy(request: SynergyRequest):
    """
    Analyze synergy between cards for a commander.

    Returns individual synergy scores with the commander
    and pair synergies between the specified cards.
    """
    try:
        result = get_synergy_scores(
            commander=request.commander,
            cards=request.cards,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Commander '{request.commander}' not found",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing synergy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Simulation Endpoints
# ============================================================================


@app.post(
    "/simulate/goldfish",
    response_model=SimulateResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Simulation"],
)
async def simulate_goldfish(request: SimulateRequest):
    """
    Run goldfish simulations with a deck.

    Simulates the deck playing against an empty opponent
    to measure damage output and consistency.
    """
    try:
        result = run_simulation(
            decklist=request.decklist,
            commander=request.commander,
            num_games=request.num_games,
            max_turns=request.max_turns,
            seed=request.seed,
        )

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Could not resolve enough cards from decklist",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/simulate/commander/{commander_name}",
    response_model=SimulateResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Simulation"],
)
async def simulate_commander_deck(
    commander_name: str,
    num_games: int = Query(default=10, ge=1, le=100),
    max_turns: int = Query(default=15, ge=1, le=30),
    seed: int | None = Query(default=None),
):
    """
    Run goldfish simulation with a commander's average EDHREC deck.

    This is the easiest way to test - just provide a commander name
    and it loads the average deck from EDHREC data automatically.

    Example: GET /simulate/commander/Atraxa%2C%20Praetors%27%20Voice?num_games=5
    """
    from .services import run_commander_simulation

    try:
        result = run_commander_simulation(
            commander=commander_name,
            num_games=num_games,
            max_turns=max_turns,
            seed=seed,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Commander '{commander_name}' not found or no average deck available",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running commander simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main entry point
# ============================================================================


def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
