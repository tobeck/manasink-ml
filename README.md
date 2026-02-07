# Manasink ML

Machine learning backend for the Manasink app - a Tinder-style Magic: The Gathering commander discovery application.

## Overview

This project uses reinforcement learning and game simulation to provide intelligent card recommendations for Commander decks. Instead of relying purely on co-occurrence statistics (like EDHREC), we simulate actual gameplay to learn which cards work well together.

## Features

- **Game Simulator**: Simplified MTG rules engine optimized for fast simulation
- **Goldfish Mode**: Test deck performance without an opponent
- **Scryfall Integration**: Fetch real card data from the Scryfall API
- **EDHREC Integration**: Import commander recommendations and average decklists
- **Card Features**: ML-ready feature extraction (mana costs, types, keywords, roles)
- **RL Training**: PPO-based policy learning with synergy reward shaping
- **Extensible Policies**: Plug in different decision-making strategies (random, greedy, synergy-aware, neural)

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/manasink-ml.git
cd manasink-ml

# Install in development mode
pip install -e ".[dev]"

# Or with ML dependencies
pip install -e ".[dev,ml]"
```

## Quick Start

### Basic Simulation

```python
from src.game import Simulator, GreedyPolicy, create_test_deck, Color

# Create a test deck
deck = create_test_deck({Color.GREEN, Color.BLUE})

# Run a goldfish simulation
sim = Simulator(max_turns=10, verbose=True)
policy = GreedyPolicy()

result = sim.run_goldfish(deck, policy, seed=42)

print(f"Turns to deal 40 damage: {result.turns_to_kill}")
print(f"Total damage dealt: {result.total_damage}")
```

### Load Real Decks from EDHREC

```python
from src.data import load_deck_from_edhrec, sync_edhrec_data

# First, sync EDHREC data (one-time setup)
sync_edhrec_data(limit=100)

# Load an average decklist
result = load_deck_from_edhrec("Atraxa, Praetors' Voice")
print(f"Loaded {result.deck_size} cards for {result.commander.name}")
```

### Train an RL Policy

```python
from src.game import create_test_deck, Color
from src.models import Trainer, TrainingConfig

# Create or load a deck
deck = create_test_deck({Color.GREEN})

# Configure training
config = TrainingConfig(
    num_episodes=1000,
    learning_rate=3e-4,
    synergy_reward_weight=0.1,
)

# Train
trainer = Trainer(config)
metrics = trainer.train(deck, num_episodes=1000)

# Use trained policy
policy = trainer.get_policy()
sim = Simulator(max_turns=15)
result = sim.run_goldfish(deck, policy)
```

### Run the API

```bash
# Start the API server
uvicorn src.api.app:app --reload

# API is available at http://localhost:8000
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

```python
import requests

# Get card recommendations
response = requests.post("http://localhost:8000/recommend/cards", json={
    "commander": "Atraxa, Praetors' Voice",
    "count": 10
})
print(response.json()["recommendations"])

# Analyze synergy between cards
response = requests.post("http://localhost:8000/analyze/synergy", json={
    "commander": "Atraxa, Praetors' Voice",
    "cards": ["Deepglow Skate", "Doubling Season"]
})
print(response.json()["average_synergy"])
```

## Project Structure

```
manasink-ml/
├── src/
│   ├── data/              # Data ingestion and processing
│   │   ├── scryfall.py    # Scryfall API client
│   │   ├── database.py    # SQLite card database
│   │   ├── edhrec.py      # EDHREC API client
│   │   ├── features.py    # ML feature extraction
│   │   ├── categories.py  # Card role categorization
│   │   └── deck_loader.py # Load decks for simulation
│   ├── game/              # MTG game simulator
│   │   ├── card.py        # Card representation
│   │   ├── state.py       # Game state and zones
│   │   ├── actions.py     # Legal actions and execution
│   │   ├── simulator.py   # Main simulation loop
│   │   └── synergy_policy.py # Synergy-aware policy
│   ├── models/            # RL training infrastructure
│   │   ├── state_encoder.py   # GameState → tensor
│   │   ├── policy_network.py  # Neural network policy
│   │   └── training.py        # PPO training loop
│   └── api/               # FastAPI REST API
│       ├── app.py         # FastAPI application
│       ├── models.py      # Request/response schemas
│       └── services.py    # Business logic
├── tests/
├── notebooks/             # Jupyter notebooks for exploration
├── data/
│   ├── raw/               # Scryfall data, caches
│   └── processed/         # Embeddings, training data
└── CLAUDE.md              # Context for Claude Code CLI
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Set up pre-commit hooks (recommended)
pip install pre-commit
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Docker Deployment

The application can be deployed using Docker Compose with PostgreSQL.

### Quick Start

```bash
# Copy environment file and customize
cp .env.example .env
# Edit .env with your settings

# Start PostgreSQL
docker-compose up -d postgres

# Run initial data sync (downloads ~30MB, takes a few minutes)
docker-compose --profile sync run --rm sync

# Start the API
docker-compose up -d api

# View logs
docker-compose logs -f api
```

The API will be available at http://localhost:8000

### Services

| Service | Description | Port |
|---------|-------------|------|
| `postgres` | PostgreSQL database | 5432 |
| `redis` | Redis cache (optional) | 6379 |
| `api` | FastAPI application | 8000 |
| `sync` | Data sync worker | - |

### Configuration

Environment variables (set in `.env`):

```bash
# Database
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=manasink

# Or use a full URL
DATABASE_URL=postgresql://user:pass@host:5432/manasink
```

### With Traefik (for SSL)

The compose file includes Traefik labels. Set your domain:

```bash
DOMAIN=yourdomain.com
```

## Roadmap

### Phase 1: Goldfish Simulator ✅
- [x] Card representation with Scryfall integration
- [x] Game state with zones
- [x] Basic action generation and execution
- [x] Goldfish simulation mode

### Phase 1.5: Data Pipeline ✅
- [x] SQLite/PostgreSQL card database with bulk ingestion
- [x] EDHREC integration (recommendations, average decks)
- [x] ML feature extraction (29-dim card vectors)
- [x] Card role categorization
- [x] Deck loading for simulation
- [x] Docker deployment support

### Phase 2: Opponent Modeling (In Progress)
- [x] Combat resolution (basic)
- [ ] Simple AI opponent with heuristics
- [ ] Basic interaction (removal, counterspells)

### Phase 3: Reinforcement Learning ✅
- [x] State encoding for neural networks
- [x] PPO agent implementation
- [x] Goldfish training loop with reward shaping
- [x] Synergy-aware policy
- [ ] Self-play training (two agents)
- [ ] Card synergy extraction from trained policies

### Phase 4: API & Integration ✅
- [x] FastAPI endpoints (health, recommend, analyze, simulate)
- [x] Card recommendation API
- [x] Deck analysis with simulation
- [x] Synergy scoring API
- [ ] Integration with main Manasink app
- [ ] Recommendation caching
- [ ] Model serving for trained policies

## License

MIT
