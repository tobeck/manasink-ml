"""
Synergy-aware policy for game simulation.

This module provides a policy that uses EDHREC synergy data and card categories
to make smarter decisions during simulation. The synergy-aware policy:

1. Scores cards based on their synergy with the commander
2. Applies category-based bonuses (e.g., prioritize ramp early game)
3. Tracks synergy bonuses earned during a game for reward shaping

This is useful for:
- Training RL agents with reward shaping based on synergies
- Generating more realistic gameplay for simulation data
- Evaluating deck quality through smarter play
"""

from dataclasses import dataclass, field
from pathlib import Path

from .actions import Action, ActionType
from .simulator import Policy
from .state import GameState


@dataclass
class SynergyBonus:
    """Tracks synergy bonuses accumulated during a game."""

    cards_played: list[tuple[str, float]] = field(
        default_factory=list
    )  # (card_name, synergy_score)
    category_bonuses: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (card_name, category, bonus)

    @property
    def total_synergy(self) -> float:
        """Total synergy score from cards played."""
        return sum(score for _, score in self.cards_played)

    @property
    def total_category_bonus(self) -> float:
        """Total category bonus earned."""
        return sum(bonus for _, _, bonus in self.category_bonuses)

    @property
    def total_bonus(self) -> float:
        """Combined synergy and category bonus."""
        return self.total_synergy + self.total_category_bonus


@dataclass
class SynergyContext:
    """Context for synergy-aware decision making."""

    commander_name: str
    card_synergies: dict[str, float]  # card_name -> synergy_score
    card_categories: dict[str, str]  # card_name -> primary category
    card_inclusions: dict[str, float]  # card_name -> inclusion_rate

    def get_synergy(self, card_name: str) -> float:
        """Get synergy score for a card."""
        return self.card_synergies.get(card_name, 0.0)

    def get_category(self, card_name: str) -> str:
        """Get primary category for a card."""
        return self.card_categories.get(card_name, "")

    def get_inclusion(self, card_name: str) -> float:
        """Get inclusion rate for a card."""
        return self.card_inclusions.get(card_name, 0.0)


class SynergyAwarePolicy(Policy):
    """
    Policy that scores cards by synergy + category bonuses.

    Scoring formula:
        score = base_value + (synergy_score * synergy_weight) + category_bonus

    Category bonuses are situational:
        - Ramp: +10 if turn <= 4 and lands < 6
        - Card Draw: +5 if hand_size < 4
        - Removal: +8 if opponent has threats (creatures)
        - Board Wipe: +15 if opponent ahead on board (3+ more creatures)

    The policy tracks all bonuses for later use in reward shaping.
    """

    def __init__(
        self,
        context: SynergyContext,
        synergy_weight: float = 5.0,
        enable_category_bonuses: bool = True,
    ):
        self.context = context
        self.synergy_weight = synergy_weight
        self.enable_category_bonuses = enable_category_bonuses
        self._bonuses = SynergyBonus()

    def reset_bonuses(self) -> None:
        """Reset bonus tracking for a new episode."""
        self._bonuses = SynergyBonus()

    def get_episode_bonuses(self) -> SynergyBonus:
        """Get the accumulated bonuses from the current/last episode."""
        return self._bonuses

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """
        Select the best action based on synergy scoring.

        Args:
            state: Current game state
            legal_actions: List of legal actions

        Returns:
            Selected action
        """
        if not legal_actions:
            return Action(action_type=ActionType.PASS)

        # Score each action
        scored_actions = []
        for action in legal_actions:
            score = self._score_action(action, state)
            scored_actions.append((score, action))

        # Sort by score descending
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Select highest scoring action
        best_score, best_action = scored_actions[0]

        # Track synergy bonus if playing a card
        if best_action.action_type == ActionType.CAST_SPELL and best_action.card:
            card_name = best_action.card.name
            synergy = self.context.get_synergy(card_name)
            if synergy > 0:
                self._bonuses.cards_played.append((card_name, synergy))

            # Track category bonus if applicable
            if self.enable_category_bonuses:
                category = self.context.get_category(card_name)
                cat_bonus = self._get_category_bonus(category, state)
                if cat_bonus > 0:
                    self._bonuses.category_bonuses.append((card_name, category, cat_bonus))

        return best_action

    def _score_action(self, action: Action, state: GameState) -> float:
        """
        Score an action based on synergy and game state.

        Args:
            action: The action to score
            state: Current game state

        Returns:
            Numeric score (higher is better)
        """
        # Pass actions have lowest priority
        if action.action_type == ActionType.PASS:
            return -1000

        # Land plays are always high priority
        if action.action_type == ActionType.PLAY_LAND:
            return 100 + (50 if state.turn_number <= 4 else 0)

        # Attack actions - score based on total power
        if action.action_type == ActionType.ATTACK:
            total_power = sum((a.card.power or 0) for a in getattr(action, "attackers", []))
            return 50 + total_power

        # Cast spell actions
        if action.action_type == ActionType.CAST_SPELL and action.card:
            card = action.card
            card_name = card.name

            # Base value from CMC (higher CMC = generally more impactful)
            base_value = card.cmc * 2

            # Synergy bonus
            synergy = self.context.get_synergy(card_name)
            synergy_bonus = synergy * self.synergy_weight

            # Category bonus
            category_bonus = 0
            if self.enable_category_bonuses:
                category = self.context.get_category(card_name)
                category_bonus = self._get_category_bonus(category, state)

            # Creature bonus (we like creatures)
            creature_bonus = 5 if card.is_creature else 0

            # Inclusion rate bonus (popular cards are often good)
            inclusion = self.context.get_inclusion(card_name)
            inclusion_bonus = inclusion * 0.1  # 0-10 bonus based on 0-100% inclusion

            return base_value + synergy_bonus + category_bonus + creature_bonus + inclusion_bonus

        # Default score for other actions
        return 0

    def _get_category_bonus(self, category: str, state: GameState) -> float:
        """
        Get situational bonus for a card category.

        Args:
            category: Card category (e.g., "Ramp", "Card Draw")
            state: Current game state

        Returns:
            Bonus value
        """
        if not category:
            return 0

        category_lower = category.lower()
        player = state.active_player
        turn = state.turn_number

        # Ramp bonus in early game when mana-light
        if "ramp" in category_lower:
            land_count = len(player.lands)
            if turn <= 4 and land_count < 6:
                return 10
            elif turn <= 6 and land_count < 8:
                return 5
            return 0

        # Card draw bonus when hand is empty
        if "card draw" in category_lower or "draw" in category_lower:
            hand_size = len(player.hand)
            if hand_size < 3:
                return 8
            elif hand_size < 5:
                return 3
            return 0

        # Removal bonus when opponent has threats
        if "removal" in category_lower:
            # Check opponent's board
            opponent_idx = 1 - state.active_player_index
            opponent = state.players[opponent_idx]
            opponent_creatures = len(opponent.creatures)
            if opponent_creatures >= 3:
                return 10
            elif opponent_creatures >= 1:
                return 5
            return 0

        # Board wipe bonus when significantly behind on board
        if "board wipe" in category_lower or "wipe" in category_lower:
            opponent_idx = 1 - state.active_player_index
            opponent = state.players[opponent_idx]
            our_creatures = len(player.creatures)
            their_creatures = len(opponent.creatures)
            if their_creatures >= our_creatures + 3:
                return 15
            elif their_creatures >= our_creatures + 1:
                return 5
            return 0

        # Protection bonus when we have valuable permanents
        if "protection" in category_lower:
            our_creatures = len(player.creatures)
            if our_creatures >= 3:
                return 5
            return 0

        # Finisher bonus in late game
        if "finisher" in category_lower:
            if turn >= 8:
                return 10
            return 0

        return 0


def load_synergy_context(
    commander_name: str,
    db_path: Path | None = None,
) -> SynergyContext | None:
    """
    Load synergy context for a commander from the database.

    Args:
        commander_name: Name of the commander
        db_path: Path to SQLite database

    Returns:
        SynergyContext or None if not found
    """
    # Import here to avoid circular imports
    from src.data.deck_loader import DEFAULT_DB_PATH, load_synergy_data

    db_path = db_path or DEFAULT_DB_PATH

    synergy_data = load_synergy_data(commander_name, db_path)

    if not synergy_data:
        return None

    return SynergyContext(
        commander_name=synergy_data.commander_name,
        card_synergies=synergy_data.card_synergies,
        card_categories=synergy_data.card_categories,
        card_inclusions=synergy_data.card_inclusions,
    )


def create_empty_context(commander_name: str = "Unknown") -> SynergyContext:
    """
    Create an empty synergy context (useful for testing without database).

    Args:
        commander_name: Name to use for the context

    Returns:
        Empty SynergyContext
    """
    return SynergyContext(
        commander_name=commander_name,
        card_synergies={},
        card_categories={},
        card_inclusions={},
    )
