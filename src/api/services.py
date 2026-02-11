"""
Service layer for API endpoints.

Contains the business logic for:
- Card recommendations
- Deck analysis
- Synergy scoring
- Goldfish simulation

Caching:
    Results are cached in Redis when available.
    Cache keys are based on function arguments.
    TTL defaults to 1 hour for recommendations, 6 hours for synergy.
"""

import logging
from collections import defaultdict

from .cache import cache
from .models import (
    AnalyzeDeckResponse,
    CardPerformance,
    CardRecommendation,
    CardSynergy,
    DeckAnalysis,
    DeckSuggestion,
    HealthResponse,
    RecommendCardsResponse,
    SimulateResponse,
    SimulationResult,
    SynergyResponse,
)

logger = logging.getLogger(__name__)

# Cache TTLs (in seconds)
RECOMMENDATIONS_TTL = 3600  # 1 hour
SYNERGY_TTL = 21600  # 6 hours
COMMANDERS_TTL = 3600  # 1 hour


# ============================================================================
# Health & Info Services
# ============================================================================


def get_health_status() -> HealthResponse:
    """Get health status of the API."""
    from sqlalchemy import text

    from src.data.db_config import DatabaseManager
    from src.data.db_models import AverageDeckCard, CardModel

    db_connected = False
    cards_loaded = 0
    commanders_available = 0

    try:
        db = DatabaseManager()
        with db.session() as session:
            # Check cards table
            cards_loaded = session.query(CardModel).count()
            db_connected = True

            # Check commanders with deck data
            commanders_available = (
                session.query(AverageDeckCard.commander_id)
                .distinct()
                .count()
            )
    except Exception as e:
        logger.error(f"Health check error: {e}")

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        database_connected=db_connected,
        cards_loaded=cards_loaded,
        commanders_available=commanders_available,
    )


def _list_commanders_impl(
    color_identity: str | None,
    min_decks: int,
    limit: int,
) -> list[dict]:
    """Internal implementation of list_commanders."""
    from src.data.deck_loader import list_available_commanders

    return list_available_commanders(
        color_identity=color_identity,
        min_decks=min_decks,
        limit=limit,
    )


def list_commanders(
    color_identity: str | None = None,
    min_decks: int = 100,
    limit: int = 50,
) -> list[dict]:
    """List available commanders (cached)."""
    # Create cache key
    cache_key = f"commanders:{color_identity or 'all'}:{min_decks}:{limit}"

    # Try cache first
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Get fresh data
    result = _list_commanders_impl(color_identity, min_decks, limit)

    # Cache it
    cache.set(cache_key, result, ttl=COMMANDERS_TTL)

    return result


# ============================================================================
# Recommendation Services
# ============================================================================


def _get_recommendations_impl(
    commander: str,
    count: int,
    exclude_tuple: tuple,  # Tuple for hashability
    budget: str | None,
    categories_tuple: tuple | None,  # Tuple for hashability
) -> dict | None:
    """Internal implementation of get_card_recommendations."""
    from src.data.db_config import DatabaseManager
    from src.data.db_models import CardModel, CommanderModel, CommanderRecommendation
    from src.data.deck_loader import load_synergy_data

    exclude_set = set(e.lower() for e in exclude_tuple)
    categories = list(categories_tuple) if categories_tuple else None

    # Load synergy data for commander
    synergy_data = load_synergy_data(commander)
    if synergy_data is None:
        return None

    manager = DatabaseManager()

    with manager.session() as session:
        # Find commander
        cmd = (
            session.query(CommanderModel)
            .filter(
                (CommanderModel.name == synergy_data.commander_name)
                | (CommanderModel.name.ilike(f"%{commander}%"))
            )
            .first()
        )
        if not cmd:
            return None

        # Get recommendations
        recs = (
            session.query(CommanderRecommendation)
            .filter(CommanderRecommendation.commander_id == cmd.id)
            .order_by(
                CommanderRecommendation.synergy_score.desc(),
                CommanderRecommendation.inclusion_rate.desc(),
            )
            .all()
        )

        recommendations = []
        total_available = len(recs)

        for rec in recs:
            card_name = rec.card_name

            # Skip excluded cards
            if card_name.lower() in exclude_set:
                continue

            # Filter by category if specified
            if categories:
                card_category = rec.category or ""
                if not any(cat.lower() in card_category.lower() for cat in categories):
                    continue

            # Get additional card info
            card_info = _get_card_info(session, card_name)

            recommendations.append(
                {
                    "name": card_name,
                    "synergy_score": rec.synergy_score or 0.0,
                    "inclusion_rate": rec.inclusion_rate or 0.0,
                    "category": rec.category,
                    "cmc": card_info.get("cmc", 0),
                    "type_line": card_info.get("type_line"),
                    "reason": _generate_recommendation_reason(
                        rec.synergy_score,
                        rec.inclusion_rate,
                        rec.category,
                    ),
                }
            )

            if len(recommendations) >= count:
                break

        return {
            "commander": synergy_data.commander_name,
            "recommendations": recommendations,
            "total_available": total_available,
        }


def get_card_recommendations(
    commander: str,
    count: int = 20,
    exclude: list[str] | None = None,
    budget: str | None = None,
    categories: list[str] | None = None,
) -> RecommendCardsResponse | None:
    """
    Get card recommendations for a commander (cached).

    Args:
        commander: Commander name
        count: Number of recommendations
        exclude: Cards to exclude
        budget: Budget tier filter
        categories: Category filters

    Returns:
        RecommendCardsResponse or None if commander not found
    """
    # Convert to tuples for cache key hashing
    exclude_tuple = tuple(sorted(exclude)) if exclude else ()
    categories_tuple = tuple(sorted(categories)) if categories else None

    # Create cache key (exclude affects results, so include in key)
    cat_hash = hash(categories_tuple) if categories_tuple else "none"
    exc_hash = hash(exclude_tuple)
    cache_key = f"recommendations:{commander.lower()}:{count}:{exc_hash}:{budget}:{cat_hash}"

    # Try cache first (only for requests without exclusions)
    if not exclude_tuple:
        cached = cache.get(cache_key)
        if cached is not None:
            return RecommendCardsResponse(
                commander=cached["commander"],
                recommendations=[CardRecommendation(**r) for r in cached["recommendations"]],
                total_available=cached["total_available"],
            )

    # Get fresh data
    result = _get_recommendations_impl(commander, count, exclude_tuple, budget, categories_tuple)

    if result is None:
        return None

    # Cache it (only if no exclusions - exclusions are user-specific)
    if not exclude_tuple:
        cache.set(cache_key, result, ttl=RECOMMENDATIONS_TTL)

    return RecommendCardsResponse(
        commander=result["commander"],
        recommendations=[CardRecommendation(**r) for r in result["recommendations"]],
        total_available=result["total_available"],
    )


def _get_card_info(session, card_name: str) -> dict:
    """Get basic card info from database."""
    from src.data.db_models import CardModel

    card = session.query(CardModel).filter(CardModel.name == card_name).first()
    if card:
        return {"cmc": card.cmc or 0, "type_line": card.type_line}
    return {"cmc": 0, "type_line": None}


def _generate_recommendation_reason(
    synergy: float,
    inclusion: float,
    category: str | None,
) -> str:
    """Generate a human-readable reason for recommendation."""
    reasons = []

    if synergy and synergy >= 0.5:
        reasons.append("high synergy with commander")
    elif synergy and synergy >= 0.3:
        reasons.append("good synergy")

    if inclusion and inclusion >= 50:
        reasons.append(f"included in {inclusion:.0f}% of decks")
    elif inclusion and inclusion >= 30:
        reasons.append("popular choice")

    if category:
        reasons.append(f"provides {category.lower()}")

    return "; ".join(reasons) if reasons else "recommended for this commander"


# ============================================================================
# Deck Analysis Services
# ============================================================================


def analyze_deck(
    commander: str,
    decklist: list[str],
    num_simulations: int = 10,
    max_turns: int = 10,
) -> AnalyzeDeckResponse | None:
    """
    Analyze a deck with simulations.

    Args:
        commander: Commander name
        decklist: List of card names
        num_simulations: Number of goldfish simulations
        max_turns: Max turns per simulation

    Returns:
        AnalyzeDeckResponse or None if commander not found
    """
    from src.data.database import CardDatabase
    from src.data.deck_loader import load_synergy_data
    from src.game import GreedyPolicy, Simulator

    # Load synergy data
    synergy_data = load_synergy_data(commander)
    if synergy_data is None:
        return None

    # Resolve cards from database
    db = CardDatabase()
    resolved_cards = []
    missing_cards = []
    card_objects = {}

    for card_name in decklist:
        card = db.get_card(card_name)
        if card:
            resolved_cards.append(card)
            card_objects[card_name] = card
        else:
            missing_cards.append(card_name)

    db.close()

    if len(resolved_cards) < 40:
        # Not enough cards to simulate
        return AnalyzeDeckResponse(
            commander=synergy_data.commander_name,
            analysis=DeckAnalysis(
                total_cards=len(resolved_cards),
                avg_cmc=0,
                land_count=0,
                creature_count=0,
                cmc_distribution={},
                role_distribution={},
                avg_damage=0,
                avg_turns_to_kill=max_turns,
                avg_cards_played=0,
                avg_lands_played=0,
            ),
            card_performance=[],
            suggestions=[],
            missing_cards=missing_cards,
        )

    # Calculate deck statistics
    analysis = _calculate_deck_stats(resolved_cards, max_turns)

    # Run simulations
    sim = Simulator(max_turns=max_turns)
    policy = GreedyPolicy()

    total_damage = 0
    total_turns_to_kill = 0
    total_cards_played = 0
    total_lands_played = 0

    for i in range(num_simulations):
        result = sim.run_goldfish(resolved_cards, policy, seed=i)
        total_damage += result.total_damage
        total_turns_to_kill += result.turns_to_kill
        total_cards_played += result.cards_played
        total_lands_played += len(result.turn_mana_available)

    # Update analysis with simulation results
    analysis.avg_damage = total_damage / num_simulations
    analysis.avg_turns_to_kill = total_turns_to_kill / num_simulations
    analysis.avg_cards_played = total_cards_played / num_simulations
    analysis.avg_lands_played = total_lands_played / num_simulations

    # Generate card performance (simplified - based on synergy scores)
    card_performance = []
    for card_name in decklist:
        if card_name in missing_cards:
            continue

        synergy = synergy_data.get_synergy(card_name)
        category = synergy_data.get_category(card_name)

        card_performance.append(
            CardPerformance(
                name=card_name,
                times_drawn=num_simulations,  # Simplified
                times_played=int(num_simulations * 0.7),  # Estimate
                play_rate=0.7,
                synergy_score=synergy,
                category=category or None,
            )
        )

    # Generate suggestions
    suggestions = _generate_suggestions(
        resolved_cards,
        synergy_data,
        analysis,
        card_performance,
    )

    return AnalyzeDeckResponse(
        commander=synergy_data.commander_name,
        analysis=analysis,
        card_performance=card_performance,
        suggestions=suggestions,
        missing_cards=missing_cards,
    )


def _calculate_deck_stats(cards: list, max_turns: int) -> DeckAnalysis:
    """Calculate basic deck statistics."""

    land_count = sum(1 for c in cards if c.is_land)
    creature_count = sum(1 for c in cards if c.is_creature)
    non_lands = [c for c in cards if not c.is_land]

    # CMC distribution
    cmc_dist = defaultdict(int)
    for card in non_lands:
        cmc = min(int(card.cmc), 7)  # Group 7+
        cmc_dist[str(cmc)] += 1

    # Role distribution (simplified)
    role_dist = {
        "Creatures": creature_count,
        "Lands": land_count,
        "Spells": len(cards) - creature_count - land_count,
    }

    avg_cmc = sum(c.cmc for c in non_lands) / len(non_lands) if non_lands else 0

    return DeckAnalysis(
        total_cards=len(cards),
        avg_cmc=round(avg_cmc, 2),
        land_count=land_count,
        creature_count=creature_count,
        cmc_distribution=dict(cmc_dist),
        role_distribution=role_dist,
        avg_damage=0,
        avg_turns_to_kill=max_turns,
        avg_cards_played=0,
        avg_lands_played=0,
    )


def _generate_suggestions(
    cards: list,
    synergy_data,
    analysis: DeckAnalysis,
    card_performance: list[CardPerformance],
) -> list[DeckSuggestion]:
    """Generate deck improvement suggestions."""
    suggestions = []

    # Check land count
    if analysis.land_count < 35:
        suggestions.append(
            DeckSuggestion(
                action="add",
                card="Basic Lands",
                reason=f"Only {analysis.land_count} lands - consider adding more for consistency",
                priority="high",
            )
        )
    elif analysis.land_count > 40:
        suggestions.append(
            DeckSuggestion(
                action="cut",
                card="Excess Lands",
                reason=f"{analysis.land_count} lands may be too many",
                priority="low",
            )
        )

    # Check mana curve
    if analysis.avg_cmc > 3.5:
        suggestions.append(
            DeckSuggestion(
                action="add",
                card="Lower CMC cards",
                reason=f"Average CMC of {analysis.avg_cmc:.1f} is high - add cheaper spells",
                priority="medium",
            )
        )

    # Suggest cuts for low synergy cards
    low_synergy = [cp for cp in card_performance if cp.synergy_score < 0.1 and cp.category is None]
    for cp in low_synergy[:3]:
        suggestions.append(
            DeckSuggestion(
                action="cut",
                card=cp.name,
                reason="Low synergy with commander",
                priority="low",
            )
        )

    return suggestions


# ============================================================================
# Synergy Services
# ============================================================================


def _get_synergy_impl(commander: str, cards_tuple: tuple) -> dict | None:
    """Internal implementation of get_synergy_scores."""
    from src.data.deck_loader import load_synergy_data

    synergy_data = load_synergy_data(commander)
    if synergy_data is None:
        return None

    cards = list(cards_tuple)

    # Get individual synergies
    card_synergies = {}
    for card in cards:
        card_synergies[card] = synergy_data.get_synergy(card)

    # Calculate pair synergies (simplified: average of individual scores)
    pair_synergies = []
    for i, card1 in enumerate(cards):
        for card2 in cards[i + 1 :]:
            syn1 = card_synergies.get(card1, 0)
            syn2 = card_synergies.get(card2, 0)
            combined = (syn1 + syn2) / 2

            pair_synergies.append(
                {
                    "card1": card1,
                    "card2": card2,
                    "synergy_score": combined,
                    "reason": _get_pair_reason(syn1, syn2),
                }
            )

    avg_synergy = sum(card_synergies.values()) / len(card_synergies) if card_synergies else 0

    return {
        "commander": synergy_data.commander_name,
        "cards": cards,
        "average_synergy": round(avg_synergy, 3),
        "card_synergies": card_synergies,
        "pair_synergies": pair_synergies,
    }


def get_synergy_scores(
    commander: str,
    cards: list[str],
) -> SynergyResponse | None:
    """
    Get synergy scores for cards with a commander (cached).

    Args:
        commander: Commander name
        cards: List of card names to analyze

    Returns:
        SynergyResponse or None if commander not found
    """
    # Convert to tuple for cache key
    cards_tuple = tuple(sorted(cards))

    # Create cache key
    cache_key = f"synergy:{commander.lower()}:{hash(cards_tuple)}"

    # Try cache first
    cached = cache.get(cache_key)
    if cached is not None:
        return SynergyResponse(
            commander=cached["commander"],
            cards=cached["cards"],
            average_synergy=cached["average_synergy"],
            card_synergies=cached["card_synergies"],
            pair_synergies=[CardSynergy(**p) for p in cached["pair_synergies"]],
        )

    # Get fresh data
    result = _get_synergy_impl(commander, cards_tuple)

    if result is None:
        return None

    # Cache it
    cache.set(cache_key, result, ttl=SYNERGY_TTL)

    return SynergyResponse(
        commander=result["commander"],
        cards=result["cards"],
        average_synergy=result["average_synergy"],
        card_synergies=result["card_synergies"],
        pair_synergies=[CardSynergy(**p) for p in result["pair_synergies"]],
    )


def _get_pair_reason(syn1: float, syn2: float) -> str:
    """Get reason for pair synergy."""
    if syn1 >= 0.3 and syn2 >= 0.3:
        return "Both cards have high synergy with commander"
    elif syn1 >= 0.3 or syn2 >= 0.3:
        return "One card has strong commander synergy"
    return "Low individual synergies"


# ============================================================================
# Simulation Services
# ============================================================================


def run_commander_simulation(
    commander: str,
    num_games: int = 10,
    max_turns: int = 15,
    seed: int | None = None,
) -> SimulateResponse | None:
    """
    Run goldfish simulation using a commander's average EDHREC deck.

    Args:
        commander: Commander name
        num_games: Number of games to simulate
        max_turns: Max turns per game
        seed: Random seed for reproducibility

    Returns:
        SimulateResponse or None if commander/deck not found
    """
    from src.data.deck_loader import load_deck_from_edhrec
    from src.game import GreedyPolicy, Simulator

    # Load the average deck for this commander
    deck_result = load_deck_from_edhrec(commander)
    if deck_result is None:
        return None

    if len(deck_result.deck) < 40:
        logger.warning(
            f"Commander '{commander}' deck too small: {len(deck_result.deck)} cards"
        )
        return None

    # Run simulations
    sim = Simulator(max_turns=max_turns)
    policy = GreedyPolicy()

    results = []
    damage_by_turn = defaultdict(list)

    base_seed = seed if seed is not None else 0

    for i in range(num_games):
        game_seed = base_seed + i if seed is not None else None
        result = sim.run_goldfish(
            deck_result.deck,
            policy,
            commander=deck_result.commander,
            seed=game_seed,
        )

        results.append(
            SimulationResult(
                game_id=i,
                turns=result.turns_to_kill,
                total_damage=result.total_damage,
                cards_played=result.cards_played,
                lands_played=len([d for d in result.turn_damage if d >= 0]),
                won=result.total_damage >= 40,
            )
        )

        # Track damage by turn
        for turn, damage in enumerate(result.turn_damage):
            damage_by_turn[turn].append(damage)

    # Calculate aggregates
    avg_damage = sum(r.total_damage for r in results) / len(results)
    avg_turns = sum(r.turns for r in results) / len(results)
    avg_cards = sum(r.cards_played for r in results) / len(results)
    win_rate = sum(1 for r in results if r.won) / len(results) * 100

    # Average damage per turn
    max_turn = max(damage_by_turn.keys()) if damage_by_turn else 0
    avg_damage_by_turn = [
        sum(damage_by_turn.get(t, [0])) / max(len(damage_by_turn.get(t, [1])), 1)
        for t in range(max_turn + 1)
    ]

    return SimulateResponse(
        num_games=num_games,
        results=results,
        avg_damage=round(avg_damage, 1),
        avg_turns=round(avg_turns, 1),
        avg_cards_played=round(avg_cards, 1),
        win_rate=round(win_rate, 1),
        damage_by_turn=avg_damage_by_turn,
    )


def run_simulation(
    decklist: list[str],
    commander: str | None = None,
    num_games: int = 10,
    max_turns: int = 15,
    seed: int | None = None,
) -> SimulateResponse | None:
    """
    Run goldfish simulations.

    Args:
        decklist: List of card names
        commander: Optional commander name
        num_games: Number of games to simulate
        max_turns: Max turns per game
        seed: Random seed for reproducibility

    Returns:
        SimulateResponse or None if cards can't be resolved
    """
    from src.data.database import CardDatabase
    from src.game import GreedyPolicy, Simulator

    # Resolve cards
    db = CardDatabase()
    resolved_cards = []
    commander_card = None

    if commander:
        commander_card = db.get_card(commander)

    for card_name in decklist:
        card = db.get_card(card_name)
        if card:
            resolved_cards.append(card)

    db.close()

    if len(resolved_cards) < 40:
        return None

    # Run simulations
    sim = Simulator(max_turns=max_turns)
    policy = GreedyPolicy()

    results = []
    damage_by_turn = defaultdict(list)

    base_seed = seed if seed is not None else 0

    for i in range(num_games):
        game_seed = base_seed + i if seed is not None else None
        result = sim.run_goldfish(
            resolved_cards,
            policy,
            commander=commander_card,
            seed=game_seed,
        )

        results.append(
            SimulationResult(
                game_id=i,
                turns=result.turns_to_kill,
                total_damage=result.total_damage,
                cards_played=result.cards_played,
                lands_played=len([d for d in result.turn_damage if d >= 0]),
                won=result.total_damage >= 40,
            )
        )

        # Track damage by turn
        for turn, damage in enumerate(result.turn_damage):
            damage_by_turn[turn].append(damage)

    # Calculate aggregates
    avg_damage = sum(r.total_damage for r in results) / len(results)
    avg_turns = sum(r.turns for r in results) / len(results)
    avg_cards = sum(r.cards_played for r in results) / len(results)
    win_rate = sum(1 for r in results if r.won) / len(results) * 100

    # Average damage per turn
    max_turn = max(damage_by_turn.keys()) if damage_by_turn else 0
    avg_damage_by_turn = [
        sum(damage_by_turn.get(t, [0])) / max(len(damage_by_turn.get(t, [1])), 1)
        for t in range(max_turn + 1)
    ]

    return SimulateResponse(
        num_games=num_games,
        results=results,
        avg_damage=round(avg_damage, 1),
        avg_turns=round(avg_turns, 1),
        avg_cards_played=round(avg_cards, 1),
        win_rate=round(win_rate, 1),
        damage_by_turn=avg_damage_by_turn,
    )
