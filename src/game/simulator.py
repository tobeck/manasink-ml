"""
Main game simulator with episode running and goldfish mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING
import random

from .card import Card, BASIC_LANDS, Color
from .state import GameState, Player, Phase, create_game
from .actions import Action, ActionType, get_legal_actions, execute_action

if TYPE_CHECKING:
    from .synergy_policy import SynergyBonus


@dataclass
class EpisodeResult:
    """Results from running a game episode."""

    winner: Optional[int]  # Player index who won, None for draw
    turns: int
    final_state: GameState

    # Metrics for analysis
    total_damage_dealt: list[int] = field(default_factory=lambda: [0, 0])
    cards_played: list[int] = field(default_factory=lambda: [0, 0])
    lands_played: list[int] = field(default_factory=lambda: [0, 0])
    creatures_cast: list[int] = field(default_factory=lambda: [0, 0])

    # Turn-by-turn state for replay/analysis
    history: list[dict] = field(default_factory=list)

    # Synergy bonuses for reward shaping (set by SynergyAwarePolicy)
    synergy_bonuses: Optional["SynergyBonus"] = None


@dataclass
class GoldfishResult:
    """Results from a goldfish simulation (solo deck test)."""

    turns_to_kill: int  # Turns to deal 40 damage (or max turns reached)
    total_damage: int
    mana_spent: int
    cards_played: int
    creatures_cast: int

    # Per-turn breakdown
    turn_damage: list[int] = field(default_factory=list)
    turn_mana_available: list[int] = field(default_factory=list)
    board_state_history: list[dict] = field(default_factory=list)


class Policy:
    """Base class for decision-making policies."""

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Select an action from the legal actions."""
        raise NotImplementedError


class RandomPolicy(Policy):
    """Randomly selects from legal actions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        return self.rng.choice(legal_actions)


class GreedyPolicy(Policy):
    """
    Simple greedy policy that prioritizes:
    1. Playing lands
    2. Casting the highest CMC spell we can afford
    3. Attacking with everything
    4. Passing
    """

    def select_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        # Prioritize land plays
        land_plays = [a for a in legal_actions if a.action_type == ActionType.PLAY_LAND]
        if land_plays:
            return land_plays[0]

        # Then cast highest CMC spell
        cast_actions = [a for a in legal_actions if a.action_type == ActionType.CAST_SPELL]
        if cast_actions:
            # Sort by CMC descending
            cast_actions.sort(key=lambda a: a.card.cmc, reverse=True)
            return cast_actions[0]

        # Attack with all if possible
        attack_actions = [a for a in legal_actions if a.action_type == ActionType.ATTACK]
        if attack_actions:
            # Find the "attack with most creatures" option
            attack_actions.sort(key=lambda a: len(a.attackers), reverse=True)
            return attack_actions[0]

        # Default to pass
        return Action(action_type=ActionType.PASS)


class Simulator:
    """
    Main game simulator for running episodes.
    """

    def __init__(self, max_turns: int = 20, verbose: bool = False):
        self.max_turns = max_turns
        self.verbose = verbose

    def run_episode(
        self,
        deck1: list[Card],
        deck2: list[Card],
        policy1: Policy,
        policy2: Policy,
        commander1: Optional[Card] = None,
        commander2: Optional[Card] = None,
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        """
        Run a full game episode between two decks with given policies.
        """
        state = create_game("Player 1", "Player 2", seed=seed)
        state.setup_game(deck1, deck2, commander1, commander2)

        result = EpisodeResult(
            winner=None,
            turns=0,
            final_state=state,
        )

        policies = [policy1, policy2]

        while not state.is_game_over and state.turn_number <= self.max_turns:
            # Get current player's policy
            policy = policies[state.active_player_index]

            # Get legal actions
            legal_actions = get_legal_actions(state)

            if not legal_actions:
                # No actions available - force pass
                execute_action(state, Action(action_type=ActionType.PASS))
                continue

            # Select and execute action
            action = policy.select_action(state, legal_actions)

            if self.verbose:
                print(
                    f"Turn {state.turn_number} ({state.phase.name}): {state.active_player.name} -> {action}"
                )

            # Track metrics
            self._track_action(result, state.active_player_index, action)

            execute_action(state, action)

            # Record history
            result.history.append(state.get_board_summary())

        result.winner = state.winner
        result.turns = state.turn_number
        result.final_state = state

        return result

    def _track_action(self, result: EpisodeResult, player_idx: int, action: Action) -> None:
        """Track metrics for an action."""
        if action.action_type == ActionType.PLAY_LAND:
            result.lands_played[player_idx] += 1
            result.cards_played[player_idx] += 1
        elif action.action_type == ActionType.CAST_SPELL:
            result.cards_played[player_idx] += 1
            if action.card and action.card.is_creature:
                result.creatures_cast[player_idx] += 1

    def run_goldfish(
        self,
        deck: list[Card],
        policy: Policy,
        commander: Optional[Card] = None,
        seed: Optional[int] = None,
        target_damage: int = 40,
    ) -> GoldfishResult:
        """
        Run a goldfish simulation - deck plays against an empty opponent.
        Measures how quickly the deck can deal damage.
        """
        # Create a game with an empty opponent deck
        state = create_game("Goldfish", "Dummy", seed=seed)

        # Setup with real deck vs empty deck
        dummy_deck = []  # No cards
        state.setup_game(deck, dummy_deck, commander, None)

        # Give dummy player infinite life so we can measure total damage
        state.players[1].life = 1000

        result = GoldfishResult(
            turns_to_kill=self.max_turns,
            total_damage=0,
            mana_spent=0,
            cards_played=0,
            creatures_cast=0,
        )

        initial_dummy_life = state.players[1].life

        while state.turn_number <= self.max_turns:
            turn_start_life = state.players[1].life
            turn_start_lands = len(state.players[0].lands)

            # Only player 0 takes meaningful turns
            if state.active_player_index == 0:
                # Run through the turn
                while state.active_player_index == 0 and state.turn_number <= self.max_turns:
                    legal_actions = get_legal_actions(state)

                    if not legal_actions:
                        execute_action(state, Action(action_type=ActionType.PASS))
                        continue

                    action = policy.select_action(state, legal_actions)

                    # Track cards played
                    if action.action_type == ActionType.CAST_SPELL:
                        result.cards_played += 1
                        if action.card and action.card.is_creature:
                            result.creatures_cast += 1
                    elif action.action_type == ActionType.PLAY_LAND:
                        result.cards_played += 1

                    execute_action(state, action)

                # Record turn metrics
                damage_this_turn = turn_start_life - state.players[1].life
                result.turn_damage.append(damage_this_turn)
                result.turn_mana_available.append(len(state.players[0].lands))

                result.board_state_history.append(
                    {
                        "turn": state.turn_number - 1,
                        "creatures": len(state.players[0].creatures),
                        "lands": len(state.players[0].lands),
                        "hand_size": len(state.players[0].hand),
                        "damage_dealt": damage_this_turn,
                    }
                )
            else:
                # Dummy player just passes
                execute_action(state, Action(action_type=ActionType.PASS))

            # Check if we've dealt enough damage
            result.total_damage = initial_dummy_life - state.players[1].life
            if result.total_damage >= target_damage:
                result.turns_to_kill = state.turn_number - 1
                break

        return result


def create_test_deck(colors: set[Color], creature_count: int = 25) -> list[Card]:
    """
    Create a simple test deck with basic lands and vanilla creatures.
    Useful for testing the simulator.
    """
    from .card import ManaCost, CardType

    deck = []

    # Add basic lands (roughly 40)
    land_count = 40
    color_list = list(colors) if colors else [Color.GREEN]
    lands_per_color = land_count // len(color_list)

    land_map = {
        Color.WHITE: BASIC_LANDS["Plains"],
        Color.BLUE: BASIC_LANDS["Island"],
        Color.BLACK: BASIC_LANDS["Swamp"],
        Color.RED: BASIC_LANDS["Mountain"],
        Color.GREEN: BASIC_LANDS["Forest"],
    }

    for color in color_list:
        for _ in range(lands_per_color):
            deck.append(land_map[color])

    # Add some vanilla creatures at various CMCs
    for i in range(creature_count):
        cmc = (i % 6) + 1  # CMC 1-6
        power = cmc
        toughness = cmc

        # Pick a color for the creature
        color = color_list[i % len(color_list)]

        cost = ManaCost(colorless=max(0, cmc - 1))
        match color:
            case Color.WHITE:
                cost.white = 1
            case Color.BLUE:
                cost.blue = 1
            case Color.BLACK:
                cost.black = 1
            case Color.RED:
                cost.red = 1
            case Color.GREEN:
                cost.green = 1

        creature = Card(
            name=f"Test Creature {i+1}",
            mana_cost=cost,
            card_types={CardType.CREATURE},
            power=power,
            toughness=toughness,
            color_identity={color},
        )
        deck.append(creature)

    # Fill remaining slots with more creatures
    remaining = 100 - len(deck)  # Commander deck size
    for i in range(remaining):
        cmc = (i % 4) + 2
        color = color_list[i % len(color_list)]

        cost = ManaCost(colorless=max(0, cmc - 1))
        match color:
            case Color.WHITE:
                cost.white = 1
            case Color.BLUE:
                cost.blue = 1
            case Color.BLACK:
                cost.black = 1
            case Color.RED:
                cost.red = 1
            case Color.GREEN:
                cost.green = 1

        creature = Card(
            name=f"Filler Creature {i+1}",
            mana_cost=cost,
            card_types={CardType.CREATURE},
            power=cmc,
            toughness=cmc,
            color_identity={color},
        )
        deck.append(creature)

    return deck[:99]  # 99 cards + commander = 100


# Quick test function
def test_goldfish():
    """Run a quick goldfish test."""
    deck = create_test_deck({Color.GREEN})
    policy = GreedyPolicy()
    sim = Simulator(max_turns=10, verbose=True)

    result = sim.run_goldfish(deck, policy, seed=42)

    print(f"\n=== Goldfish Results ===")
    print(f"Turns to kill: {result.turns_to_kill}")
    print(f"Total damage: {result.total_damage}")
    print(f"Cards played: {result.cards_played}")
    print(f"Creatures cast: {result.creatures_cast}")
    print(f"\nPer-turn damage: {result.turn_damage}")

    return result


if __name__ == "__main__":
    test_goldfish()
