"""
ML Models package for Manasink.

This package provides reinforcement learning infrastructure for training
policies that learn to play MTG through self-play and goldfish simulation.

Components:
- StateEncoder: Converts GameState to tensor representation
- PolicyNetwork: Neural network with policy and value heads
- NeuralPolicy: Policy wrapper for use with game Simulator
- Trainer: PPO training loop with experience collection

Quick start:
    from src.game import create_test_deck, Color
    from src.models import Trainer, TrainingConfig

    # Create a deck
    deck = create_test_deck({Color.GREEN})

    # Configure training
    config = TrainingConfig(
        num_episodes=100,
        learning_rate=3e-4,
    )

    # Train
    trainer = Trainer(config)
    metrics = trainer.train(deck, num_episodes=100)

    # Get trained policy for inference
    policy = trainer.get_policy()
"""

from .policy_network import (
    CardSetEncoder,
    NeuralPolicy,
    PolicyNetwork,
)
from .state_encoder import (
    CARD_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    EncodedState,
    StateEncoder,
    batch_encode_states,
)
from .training import (
    Experience,
    ExperienceBuffer,
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    create_trainer,
)

__all__ = [
    # State encoding
    "StateEncoder",
    "EncodedState",
    "batch_encode_states",
    "CARD_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    # Neural network
    "PolicyNetwork",
    "NeuralPolicy",
    "CardSetEncoder",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "Experience",
    "ExperienceBuffer",
    "create_trainer",
]
