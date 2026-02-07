"""
FastAPI application for Manasink ML API.

Provides endpoints for:
- Card recommendations based on commander
- Deck analysis with simulation
- Synergy scoring between cards
- Goldfish simulation
"""

from contextlib import asynccontextmanager
from typing import Optional
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    RecommendCardsRequest,
    RecommendCardsResponse,
    AnalyzeDeckRequest,
    AnalyzeDeckResponse,
    SynergyRequest,
    SynergyResponse,
    SimulateRequest,
    SimulateResponse,
    HealthResponse,
    ErrorResponse,
)
from .services import (
    get_card_recommendations,
    analyze_deck,
    get_synergy_scores,
    run_simulation,
    get_health_status,
    list_commanders,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Manasink ML API...")
    # Startup: could preload models here
    yield
    # Shutdown: cleanup
    logger.info("Shutting down Manasink ML API...")


# Create FastAPI app
app = FastAPI(
    title="Manasink ML API",
    description="Machine learning-powered MTG Commander card recommendations",
    version="0.1.0",
    lifespan=lifespan,
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


@app.get("/commanders", tags=["Info"])
async def get_commanders(
    color_identity: Optional[str] = Query(
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


# ============================================================================
# Main entry point
# ============================================================================


def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
