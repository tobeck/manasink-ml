"""
Game state representation with zones and turn tracking.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .card import Card, Color, ManaCost, ManaPool


class Phase(Enum):
    """Simplified phases for simulation."""

    UNTAP = auto()
    UPKEEP = auto()
    DRAW = auto()
    MAIN_1 = auto()
    COMBAT = auto()
    MAIN_2 = auto()
    END = auto()


class Zone(Enum):
    """Game zones."""

    LIBRARY = auto()
    HAND = auto()
    BATTLEFIELD = auto()
    GRAVEYARD = auto()
    EXILE = auto()
    COMMAND = auto()  # Commander zone


@dataclass
class Permanent:
    """
    A card on the battlefield with its current state.
    """

    card: Card
    tapped: bool = False
    summoning_sick: bool = True  # Can't attack/tap the turn it enters
    damage_marked: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    # For tracking attackers/blockers
    attacking: bool = False
    blocking: Optional["Permanent"] = None  # What this is blocking
    blocked_by: list["Permanent"] = field(default_factory=list)

    @property
    def current_power(self) -> int:
        """Power including modifications."""
        base = self.card.power or 0
        return base + self.counters.get("+1/+1", 0) - self.counters.get("-1/-1", 0)

    @property
    def current_toughness(self) -> int:
        """Toughness including modifications."""
        base = self.card.toughness or 0
        return base + self.counters.get("+1/+1", 0) - self.counters.get("-1/-1", 0)

    @property
    def is_creature(self) -> bool:
        return self.card.is_creature

    @property
    def is_land(self) -> bool:
        return self.card.is_land

    @property
    def can_attack(self) -> bool:
        """Can this creature attack?"""
        if not self.is_creature:
            return False
        if self.tapped:
            return False
        if self.summoning_sick and "haste" not in self.card.keywords:
            return False
        if "defender" in self.card.keywords:
            return False
        return True

    @property
    def can_block(self) -> bool:
        """Can this creature block?"""
        if not self.is_creature:
            return False
        if self.tapped:
            return False
        return True

    def tap(self) -> bool:
        """Tap this permanent. Returns True if it wasn't already tapped."""
        if self.tapped:
            return False
        self.tapped = True
        return True

    def untap(self) -> None:
        """Untap this permanent."""
        self.tapped = False

    def take_damage(self, amount: int) -> None:
        """Mark damage on this permanent."""
        self.damage_marked += amount

    def is_dead(self) -> bool:
        """Check if this creature has lethal damage."""
        if not self.is_creature:
            return False
        return self.damage_marked >= self.current_toughness

    def clear_damage(self) -> None:
        """Clear damage at end of turn."""
        self.damage_marked = 0
        self.attacking = False
        self.blocking = None
        self.blocked_by = []

    def __repr__(self) -> str:
        state = []
        if self.tapped:
            state.append("tapped")
        if self.summoning_sick:
            state.append("sick")
        state_str = f" ({', '.join(state)})" if state else ""
        return f"Permanent({self.card.name}{state_str})"


@dataclass
class Player:
    """
    Represents a player in the game.
    """

    name: str
    life: int = 40  # Commander starting life

    # Zones
    library: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    battlefield: list[Permanent] = field(default_factory=list)
    graveyard: list[Card] = field(default_factory=list)
    exile: list[Card] = field(default_factory=list)

    # Commander
    commander: Card | None = None
    commander_zone: bool = True  # Is commander in command zone?
    commander_tax: int = 0  # Additional cost for recasting

    # Mana
    mana_pool: ManaPool = field(default_factory=ManaPool)

    # Turn state
    land_played_this_turn: bool = False

    def shuffle_library(self, rng: random.Random | None = None) -> None:
        """Shuffle the library."""
        if rng:
            rng.shuffle(self.library)
        else:
            random.shuffle(self.library)

    def draw(self, count: int = 1) -> list[Card]:
        """Draw cards from library to hand."""
        drawn = []
        for _ in range(count):
            if not self.library:
                break  # TODO: Handle losing from empty library
            card = self.library.pop(0)
            self.hand.append(card)
            drawn.append(card)
        return drawn

    def mill(self, count: int = 1) -> list[Card]:
        """Mill cards from library to graveyard."""
        milled = []
        for _ in range(count):
            if not self.library:
                break
            card = self.library.pop(0)
            self.graveyard.append(card)
            milled.append(card)
        return milled

    @property
    def creatures(self) -> list[Permanent]:
        """All creatures on battlefield."""
        return [p for p in self.battlefield if p.is_creature]

    @property
    def lands(self) -> list[Permanent]:
        """All lands on battlefield."""
        return [p for p in self.battlefield if p.is_land]

    @property
    def untapped_lands(self) -> list[Permanent]:
        """Untapped lands."""
        return [p for p in self.lands if not p.tapped]

    def get_available_mana(self) -> ManaPool:
        """
        Calculate mana available from untapped lands.
        Simplified: each basic land type produces its color.
        """
        pool = ManaPool()
        for land in self.untapped_lands:
            # Simplified land mana production
            card_name = land.card.name.lower()
            if "plains" in card_name or Color.WHITE in land.card.color_identity:
                pool.white += 1
            elif "island" in card_name or Color.BLUE in land.card.color_identity:
                pool.blue += 1
            elif "swamp" in card_name or Color.BLACK in land.card.color_identity:
                pool.black += 1
            elif "mountain" in card_name or Color.RED in land.card.color_identity:
                pool.red += 1
            elif "forest" in card_name or Color.GREEN in land.card.color_identity:
                pool.green += 1
            else:
                # Unknown land - assume colorless
                pool.colorless += 1
        return pool

    def tap_lands_for_mana(self, cost: "ManaCost") -> bool:
        """
        Tap lands to pay a mana cost.
        Returns True if successful.
        Simplified: taps lands greedily.
        """
        available = self.get_available_mana()
        if not cost.can_pay_with(available):
            return False

        # Tap lands to produce the mana we need
        # This is simplified - a real implementation would optimize
        lands_to_tap = []
        pool = ManaPool()

        for land in self.untapped_lands:
            if cost.can_pay_with(pool):
                break

            card_name = land.card.name.lower()
            if "plains" in card_name or Color.WHITE in land.card.color_identity:
                pool.white += 1
            elif "island" in card_name or Color.BLUE in land.card.color_identity:
                pool.blue += 1
            elif "swamp" in card_name or Color.BLACK in land.card.color_identity:
                pool.black += 1
            elif "mountain" in card_name or Color.RED in land.card.color_identity:
                pool.red += 1
            elif "forest" in card_name or Color.GREEN in land.card.color_identity:
                pool.green += 1
            else:
                pool.colorless += 1
            lands_to_tap.append(land)

        if not cost.can_pay_with(pool):
            return False

        for land in lands_to_tap:
            land.tap()

        return True

    def is_dead(self) -> bool:
        """Check if this player has lost."""
        return self.life <= 0

    def __repr__(self) -> str:
        return f"Player({self.name}, {self.life} life, {len(self.hand)} cards)"


@dataclass
class GameState:
    """
    Complete game state for simulation.
    """

    players: list[Player]
    active_player_index: int = 0
    turn_number: int = 1
    phase: Phase = Phase.MAIN_1

    # For reproducibility
    rng: random.Random = field(default_factory=random.Random)

    # Game over state
    winner: int | None = None  # Index of winning player, None if ongoing

    @property
    def active_player(self) -> Player:
        """Current turn's player."""
        return self.players[self.active_player_index]

    @property
    def defending_player(self) -> Player:
        """Non-active player (simplified for 2-player)."""
        return self.players[1 - self.active_player_index]

    @property
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.winner is not None

    def check_state_based_actions(self) -> None:
        """
        Check and resolve state-based actions.
        - Players at 0 or less life lose
        - Creatures with lethal damage die
        """
        # Check player life totals
        for i, player in enumerate(self.players):
            if player.is_dead() and self.winner is None:
                self.winner = 1 - i  # Other player wins
                return

        # Check creature deaths
        for player in self.players:
            dead_creatures = [p for p in player.battlefield if p.is_creature and p.is_dead()]
            for permanent in dead_creatures:
                player.battlefield.remove(permanent)
                player.graveyard.append(permanent.card)

    def untap_step(self) -> None:
        """Untap all permanents for active player."""
        for permanent in self.active_player.battlefield:
            permanent.untap()
            permanent.summoning_sick = False

    def draw_step(self) -> list[Card]:
        """Active player draws a card."""
        # Skip draw on turn 1 for first player (Commander rule)
        if self.turn_number == 1 and self.active_player_index == 0:
            return []
        return self.active_player.draw(1)

    def end_turn(self) -> None:
        """
        End the current turn and advance to next player.
        """
        # Clear damage from creatures
        for player in self.players:
            for permanent in player.battlefield:
                permanent.clear_damage()

        # Reset turn state
        self.active_player.land_played_this_turn = False
        self.active_player.mana_pool.clear()

        # Advance to next player
        self.active_player_index = (self.active_player_index + 1) % len(self.players)

        # Increment turn counter when we get back to first player
        if self.active_player_index == 0:
            self.turn_number += 1

        self.phase = Phase.UNTAP

    def setup_game(
        self,
        deck1: list[Card],
        deck2: list[Card],
        commander1: Card | None = None,
        commander2: Card | None = None,
    ) -> None:
        """
        Initialize a new game with the given decks.
        """
        # Setup player 1
        self.players[0].library = deck1.copy()
        self.players[0].commander = commander1
        self.players[0].shuffle_library(self.rng)
        self.players[0].draw(7)

        # Setup player 2
        self.players[1].library = deck2.copy()
        self.players[1].commander = commander2
        self.players[1].shuffle_library(self.rng)
        self.players[1].draw(7)

        self.turn_number = 1
        self.active_player_index = 0
        self.phase = Phase.MAIN_1
        self.winner = None

    def copy(self) -> "GameState":
        """Create a deep copy of this game state."""
        return deepcopy(self)

    def get_board_summary(self) -> dict:
        """Get a summary of the current board state."""
        return {
            "turn": self.turn_number,
            "phase": self.phase.name,
            "active_player": self.active_player.name,
            "players": [
                {
                    "name": p.name,
                    "life": p.life,
                    "hand_size": len(p.hand),
                    "library_size": len(p.library),
                    "creatures": len(p.creatures),
                    "lands": len(p.lands),
                    "graveyard_size": len(p.graveyard),
                }
                for p in self.players
            ],
        }

    def __repr__(self) -> str:
        phase = self.phase.name
        active = self.active_player.name
        return f"GameState(turn={self.turn_number}, phase={phase}, active={active})"


def create_game(
    player1_name: str = "Player 1",
    player2_name: str = "Player 2",
    seed: int | None = None,
) -> GameState:
    """Create a new game with two players."""
    rng = random.Random(seed) if seed is not None else random.Random()

    return GameState(
        players=[
            Player(name=player1_name),
            Player(name=player2_name),
        ],
        rng=rng,
    )
