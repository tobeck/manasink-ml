"""
MTG Game Simulator package.
"""

from .actions import Action, ActionType, execute_action, get_legal_actions
from .card import BASIC_LANDS, Card, CardType, Color, ManaCost, ManaPool
from .simulator import (
    EpisodeResult,
    GoldfishResult,
    GreedyPolicy,
    Policy,
    RandomPolicy,
    Simulator,
    create_test_deck,
)
from .state import GameState, Permanent, Phase, Player, Zone, create_game
from .synergy_policy import (
    SynergyAwarePolicy,
    SynergyBonus,
    SynergyContext,
    create_empty_context,
    load_synergy_context,
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
    # Synergy Policy
    "SynergyAwarePolicy",
    "SynergyBonus",
    "SynergyContext",
    "load_synergy_context",
    "create_empty_context",
]
