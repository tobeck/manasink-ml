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

### ML Features (`CardFeatures`)
Cards are encoded as 29-dimensional feature vectors for ML:
- **Mana** (7): W, U, B, R, G, colorless, CMC
- **Types** (8): creature, instant, sorcery, artifact, enchantment, planeswalker, land, legendary
- **Stats** (2): power, toughness
- **Color identity** (5): WUBRG bitmap
- **Role scores** (7): ramp, card_draw, removal, board_wipe, protection, finisher, utility

```python
from src.data import get_feature_vector
features = get_feature_vector("Sol Ring")
vector = features.to_vector()  # 29-dim list
```

### Reward Signals
- **Goldfish**: Turns to achieve board state milestones
- **vs Opponent**: Win rate, life differential, board control metrics
- **Synergy**: Cards that consistently appear in winning game states together

### RL Training Infrastructure
The `src/models/` package provides PPO-based training:

**State Encoding** (`StateEncoder`):
- Hand, battlefield, opponent battlefield → padded card tensors
- Global features: turn, life totals, mana, phase (17 dims)
- Cards encoded as 29-dim feature vectors

**Policy Network** (`PolicyNetwork`):
- Card set encoders with mean pooling
- Shared trunk → policy head (action logits) + value head
- Action masking for legal actions only

**Training** (`Trainer`):
- PPO with clipped surrogate objective
- GAE for advantage estimation
- Reward shaping: damage dealt + synergy bonuses
- Checkpoint save/load support

```python
from src.models import Trainer, TrainingConfig
from src.game import create_test_deck, Color

deck = create_test_deck({Color.GREEN})
config = TrainingConfig(num_episodes=1000, learning_rate=3e-4)
trainer = Trainer(config)
trainer.train(deck, num_episodes=1000)
policy = trainer.get_policy()  # Use for inference
```

## Key Files

### Game Simulator (`src/game/`)
- `card.py` - Card dataclass and Scryfall integration
- `state.py` - GameState with zones (hand, battlefield, graveyard, etc.)
- `actions.py` - Legal action generation and execution
- `simulator.py` - Main game loop and episode runner
- `synergy_policy.py` - Synergy-aware policy using EDHREC data

### Data Pipeline (`src/data/`)
- `scryfall.py` - Scryfall API client and caching
- `database.py` - SQLite CardDatabase for bulk storage
- `edhrec.py` - EDHREC API client
- `edhrec_ingest.py` - Sync EDHREC recommendations to database
- `features.py` - ML feature extraction (CardFeatures, 29-dim vectors)
- `categories.py` - Card role categorization (ramp, removal, etc.)
- `deck_loader.py` - Load average decks for simulation

### RL Training (`src/models/`)
- `state_encoder.py` - Converts GameState to tensors for neural networks
- `policy_network.py` - PolicyNetwork (policy + value heads), NeuralPolicy wrapper
- `training.py` - PPO Trainer with experience collection and GAE

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

## Data CLI

The `manasink-data` CLI provides data pipeline management:

```bash
# Sync Scryfall card database (~30MB download)
manasink-data sync-scryfall

# Sync EDHREC data for top commanders
manasink-data sync-edhrec --limit 100

# Extract ML features for all cards
manasink-data extract-features

# Show database statistics
manasink-data stats
```

Or programmatically:

```python
from src.data import sync_database, sync_edhrec_data, populate_card_features

# One-time setup
sync_database()              # Scryfall cards
sync_edhrec_data(limit=100)  # EDHREC recommendations
populate_card_features()     # ML features

# Load deck for simulation
from src.data import load_deck_from_edhrec
result = load_deck_from_edhrec("Atraxa, Praetors' Voice")
```

## Design Decisions

1. **Simplified Rules**: We intentionally skip complex interactions (stack, priority, triggered abilities with choices). The simulator approximates gameplay well enough for ML signal extraction.

2. **Commander Focus**: 100-card singleton format with commander in command zone. Color identity restrictions enforced.

3. **Deterministic Option**: Simulator supports seeded randomness for reproducible training runs.

4. **Fast Iteration**: Prioritize simulation speed over rule completeness. We can refine rules later as we identify where simplifications hurt model quality.

## API Server

The FastAPI server provides the following endpoints:

```bash
# Start the server
uvicorn src.api.app:app --reload --port 8000

# Or using Python directly
python -m src.api.app
```

**Endpoints:**
- `GET  /health` - Health check with database status
- `GET  /commanders` - List available commanders
- `POST /recommend/cards` - Card recommendations for a commander
- `POST /analyze/deck` - Deck analysis with simulation
- `POST /analyze/synergy` - Synergy scores between cards
- `POST /simulate/goldfish` - Run goldfish simulations

**Example:**
```python
import requests

# Get recommendations
resp = requests.post("http://localhost:8000/recommend/cards", json={
    "commander": "Atraxa, Praetors' Voice",
    "count": 20,
    "categories": ["Ramp", "Card Draw"]
})

# Analyze a deck
resp = requests.post("http://localhost:8000/analyze/deck", json={
    "commander": "Korvold, Fae-Cursed King",
    "decklist": ["Sol Ring", "Arcane Signet", ...],
    "num_simulations": 10
})
```

## Integration with Manasink App

The main Manasink app (Next.js + Supabase) will call this service via API.

## Related Resources

- [Scryfall API Docs](https://scryfall.com/docs/api)
- [EDHREC](https://edhrec.com/) - Commander popularity data
- [open-mtg](https://github.com/hlynurd/open-mtg) - Reference Python MTG implementation
- [RLCard](https://github.com/datamllab/rlcard) - RL toolkit for card games (architecture reference)
