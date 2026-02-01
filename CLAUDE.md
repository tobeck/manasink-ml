# CLAUDE.md - Manasink ML

## Project Overview

Manasink-ML is the machine learning backend for the Manasink app - a Tinder-style Magic: The Gathering commander discovery application. This repo handles card recommendations based on gameplay simulation and reinforcement learning.

## Architecture

```
manasink-ml/
├── src/
│   ├── data/          # Scryfall data ingestion & card representations
│   ├── game/          # MTG game simulator (zones, actions, state)
│   ├── models/        # ML models (GNN, RL agents)
│   └── api/           # FastAPI endpoints for serving predictions
├── tests/
├── notebooks/         # Exploration & prototyping
└── data/
    ├── raw/           # Scryfall dumps, EDHREC data
    └── processed/     # Cleaned datasets, embeddings
```

## Core Concepts

### Game Simulator
We're building a simplified MTG simulator focused on Commander format. The goal is NOT to implement every rule, but to simulate enough gameplay that an RL agent can learn card synergies through self-play.

**Phase 1: Goldfish Simulator**
- Deck plays against itself (no opponent)
- Tracks mana curve, card draw, ramp effectiveness
- Measures "does the deck do its thing?"

**Phase 2: Simple Opponent**
- Basic opponent that plays creatures on curve
- Attacks/blocks with simple heuristics
- Enough to evaluate removal, interaction, board presence

**Phase 3: Self-Play RL**
- Pit decks against each other
- Learn optimal play patterns
- Extract card synergy signals from trained policies

### Card Representation
Cards are represented using data from Scryfall API. Key attributes:
- Mana cost / CMC
- Card types (creature, instant, sorcery, etc.)
- Oracle text (for keyword extraction)
- Power/toughness (creatures)
- Color identity (critical for Commander)

### Reward Signals
- **Goldfish**: Turns to achieve board state milestones
- **vs Opponent**: Win rate, life differential, board control metrics
- **Synergy**: Cards that consistently appear in winning game states together

## Key Files

- `src/game/card.py` - Card dataclass and Scryfall integration
- `src/game/state.py` - GameState with zones (hand, battlefield, graveyard, etc.)
- `src/game/actions.py` - Legal action generation and execution
- `src/game/simulator.py` - Main game loop and episode runner
- `src/data/scryfall.py` - Scryfall API client and caching

## Development Commands

```bash
# Install in dev mode
pip install -e ".[dev,ml]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type check
mypy src/
```

## Design Decisions

1. **Simplified Rules**: We intentionally skip complex interactions (stack, priority, triggered abilities with choices). The simulator approximates gameplay well enough for ML signal extraction.

2. **Commander Focus**: 100-card singleton format with commander in command zone. Color identity restrictions enforced.

3. **Deterministic Option**: Simulator supports seeded randomness for reproducible training runs.

4. **Fast Iteration**: Prioritize simulation speed over rule completeness. We can refine rules later as we identify where simplifications hurt model quality.

## Integration with Manasink App

The main Manasink app (Next.js + Supabase) will call this service via API:
- `POST /recommend/cards` - Given a commander, return recommended cards
- `POST /analyze/deck` - Given a decklist, suggest cuts/additions
- `POST /analyze/synergy` - Score card pairs for synergy

## Related Resources

- [Scryfall API Docs](https://scryfall.com/docs/api)
- [EDHREC](https://edhrec.com/) - Commander popularity data
- [open-mtg](https://github.com/hlynurd/open-mtg) - Reference Python MTG implementation
- [RLCard](https://github.com/datamllab/rlcard) - RL toolkit for card games (architecture reference)
