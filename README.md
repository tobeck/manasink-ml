# Manasink ML

Machine learning backend for the Manasink app - a Tinder-style Magic: The Gathering commander discovery application.

## Overview

This project uses reinforcement learning and game simulation to provide intelligent card recommendations for Commander decks. Instead of relying purely on co-occurrence statistics (like EDHREC), we simulate actual gameplay to learn which cards work well together.

## Features

- **Game Simulator**: Simplified MTG rules engine optimized for fast simulation
- **Goldfish Mode**: Test deck performance without an opponent
- **Scryfall Integration**: Fetch real card data from the Scryfall API
- **Extensible Policies**: Plug in different decision-making strategies

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

```python
from src.game import (
    Simulator, 
    GreedyPolicy, 
    create_test_deck, 
    Color
)

# Create a test deck
deck = create_test_deck({Color.GREEN, Color.BLUE})

# Run a goldfish simulation
sim = Simulator(max_turns=10, verbose=True)
policy = GreedyPolicy()

result = sim.run_goldfish(deck, policy, seed=42)

print(f"Turns to deal 40 damage: {result.turns_to_kill}")
print(f"Total damage dealt: {result.total_damage}")
```

## Project Structure

```
manasink-ml/
├── src/
│   ├── data/          # Scryfall API client, data processing
│   ├── game/          # MTG game simulator
│   │   ├── card.py    # Card representation
│   │   ├── state.py   # Game state and zones
│   │   ├── actions.py # Legal actions and execution
│   │   └── simulator.py # Main simulation loop
│   ├── models/        # ML models (GNN, RL agents)
│   └── api/           # FastAPI service
├── tests/
├── notebooks/         # Jupyter notebooks for exploration
├── data/
│   ├── raw/           # Scryfall data, caches
│   └── processed/     # Embeddings, training data
└── CLAUDE.md          # Context for Claude Code CLI
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
```

## Roadmap

### Phase 1: Goldfish Simulator ✅
- [x] Card representation with Scryfall integration
- [x] Game state with zones
- [x] Basic action generation and execution
- [x] Goldfish simulation mode

### Phase 2: Opponent Modeling
- [ ] Simple AI opponent with heuristics
- [ ] Combat resolution
- [ ] Basic interaction (removal, counterspells)

### Phase 3: Reinforcement Learning
- [ ] State encoding for neural networks
- [ ] PPO/DQN agent implementation
- [ ] Self-play training loop
- [ ] Card synergy extraction from trained policies

### Phase 4: API & Integration
- [ ] FastAPI endpoints
- [ ] Integration with main Manasink app
- [ ] Recommendation caching

## License

MIT
