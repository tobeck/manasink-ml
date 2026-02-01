"""
MTG Game Simulator package.
"""

from .card import Card, ManaCost, ManaPool, CardType, Color, BASIC_LANDS
from .state import GameState, Player, Permanent, Phase, Zone, create_game
from .actions import Action, ActionType, get_legal_actions, execute_action
from .simulator import (
    Simulator, 
    EpisodeResult, 
    GoldfishResult,
    Policy,
    RandomPolicy,
    GreedyPolicy,
    create_test_deck,
)

__all__ = [
    # Card
    "Card",
    "ManaCost", 
    "ManaPool",
    "CardType",
    "Color",
    "BASIC_LANDS",
    # State
    "GameState",
    "Player",
    "Permanent",
    "Phase",
    "Zone",
    "create_game",
    # Actions
    "Action",
    "ActionType",
    "get_legal_actions",
    "execute_action",
    # Simulator
    "Simulator",
    "EpisodeResult",
    "GoldfishResult",
    "Policy",
    "RandomPolicy",
    "GreedyPolicy",
    "create_test_deck",
]
