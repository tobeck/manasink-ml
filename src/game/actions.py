"""
Action generation and execution for the game simulator.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

from .card import Card, CardType
from .state import GameState, Player, Permanent, Phase


class ActionType(Enum):
    """Types of actions a player can take."""

    PLAY_LAND = auto()
    CAST_SPELL = auto()
    ACTIVATE_ABILITY = auto()
    ATTACK = auto()
    BLOCK = auto()
    PASS = auto()


@dataclass
class Action:
    """
    Represents an action a player can take.
    """

    action_type: ActionType
    card: Optional[Card] = None
    source: Optional[Permanent] = None
    targets: list[Union[Permanent, Player]] = None
    attackers: list[Permanent] = None  # For ATTACK actions
    blocker: Optional[Permanent] = None  # For BLOCK actions
    blocking: Optional[Permanent] = None  # What creature to block

    def __post_init__(self):
        if self.targets is None:
            self.targets = []
        if self.attackers is None:
            self.attackers = []

    def __repr__(self) -> str:
        match self.action_type:
            case ActionType.PLAY_LAND:
                return f"PlayLand({self.card.name})"
            case ActionType.CAST_SPELL:
                return f"Cast({self.card.name})"
            case ActionType.ATTACK:
                names = [a.card.name for a in self.attackers]
                return f"Attack({names})"
            case ActionType.BLOCK:
                return f"Block({self.blocker.card.name} -> {self.blocking.card.name})"
            case ActionType.PASS:
                return "Pass"
            case _:
                return f"Action({self.action_type})"


def get_legal_actions(state: GameState) -> list[Action]:
    """
    Get all legal actions for the active player in the current game state.
    """
    if state.is_game_over:
        return []

    actions = []
    player = state.active_player

    match state.phase:
        case Phase.MAIN_1 | Phase.MAIN_2:
            actions.extend(_get_main_phase_actions(state, player))
        case Phase.COMBAT:
            actions.extend(_get_combat_actions(state, player))
        case _:
            pass  # Other phases handled automatically

    # Can always pass
    actions.append(Action(action_type=ActionType.PASS))

    return actions


def _get_main_phase_actions(state: GameState, player: Player) -> list[Action]:
    """Get legal actions during main phase."""
    actions = []

    # Play a land (if we haven't this turn)
    if not player.land_played_this_turn:
        lands_in_hand = [c for c in player.hand if c.is_land]
        for land in lands_in_hand:
            actions.append(
                Action(
                    action_type=ActionType.PLAY_LAND,
                    card=land,
                )
            )

    # Cast spells
    available_mana = player.get_available_mana()

    for card in player.hand:
        if card.is_land:
            continue

        # Check if we can pay the mana cost
        if card.mana_cost.can_pay_with(available_mana):
            # For now, we don't handle targeting - just add the action
            actions.append(
                Action(
                    action_type=ActionType.CAST_SPELL,
                    card=card,
                )
            )

    # Cast commander from command zone
    if player.commander and player.commander_zone:
        commander = player.commander
        # Add commander tax to the cost check
        effective_cost = commander.mana_cost
        # Simplified: just check base cost for now
        if effective_cost.can_pay_with(available_mana):
            actions.append(
                Action(
                    action_type=ActionType.CAST_SPELL,
                    card=commander,
                )
            )

    return actions


def _get_combat_actions(state: GameState, player: Player) -> list[Action]:
    """Get legal combat actions."""
    actions = []

    # Declare attackers
    attackers = [p for p in player.creatures if p.can_attack]

    if attackers:
        # For simplicity, generate "attack with all" and "attack with each individual"
        # A full implementation would generate all subsets

        # Attack with all
        actions.append(
            Action(
                action_type=ActionType.ATTACK,
                attackers=attackers.copy(),
            )
        )

        # Attack with each individually
        for attacker in attackers:
            actions.append(
                Action(
                    action_type=ActionType.ATTACK,
                    attackers=[attacker],
                )
            )

    return actions


def execute_action(state: GameState, action: Action) -> None:
    """
    Execute an action and modify the game state.
    """
    player = state.active_player

    match action.action_type:
        case ActionType.PLAY_LAND:
            _execute_play_land(state, player, action)
        case ActionType.CAST_SPELL:
            _execute_cast_spell(state, player, action)
        case ActionType.ATTACK:
            _execute_attack(state, player, action)
        case ActionType.BLOCK:
            _execute_block(state, player, action)
        case ActionType.PASS:
            _execute_pass(state)

    # Check state-based actions after any action
    state.check_state_based_actions()


def _execute_play_land(state: GameState, player: Player, action: Action) -> None:
    """Execute playing a land."""
    card = action.card

    # Remove from hand
    player.hand.remove(card)

    # Put onto battlefield
    permanent = Permanent(card=card, summoning_sick=False)  # Lands don't have summoning sickness
    player.battlefield.append(permanent)

    # Mark that we've played a land
    player.land_played_this_turn = True


def _execute_cast_spell(state: GameState, player: Player, action: Action) -> None:
    """Execute casting a spell."""
    card = action.card

    # Pay mana cost
    if not player.tap_lands_for_mana(card.mana_cost):
        raise ValueError(f"Cannot pay mana cost for {card.name}")

    # Remove from hand or command zone
    if card in player.hand:
        player.hand.remove(card)
    elif player.commander and card == player.commander:
        player.commander_zone = False
        player.commander_tax += 2  # Increase tax for next cast

    # Resolve the spell
    if (
        card.is_creature
        or CardType.ARTIFACT in card.card_types
        or CardType.ENCHANTMENT in card.card_types
    ):
        # Permanent spell - put onto battlefield
        permanent = Permanent(card=card)
        player.battlefield.append(permanent)
    else:
        # Non-permanent spell - resolve and go to graveyard
        _resolve_spell_effects(state, player, card)
        player.graveyard.append(card)


def _resolve_spell_effects(state: GameState, player: Player, card: Card) -> None:
    """
    Resolve the effects of a non-permanent spell.
    This is highly simplified - we just look for common patterns.
    """
    text = card.oracle_text.lower()

    # Draw effects
    if "draw" in text:
        # Try to extract number
        import re

        draw_match = re.search(r"draw (\d+|a|two|three)", text)
        if draw_match:
            num_str = draw_match.group(1)
            num = {"a": 1, "two": 2, "three": 3}.get(num_str, 1)
            try:
                num = int(num_str)
            except ValueError:
                pass
            player.draw(num)

    # Direct damage to opponent
    if "damage" in text and "target" in text:
        damage_match = re.search(r"(\d+) damage", text)
        if damage_match:
            damage = int(damage_match.group(1))
            # Apply to opponent (simplified - no targeting)
            state.defending_player.life -= damage

    # Life gain
    if "gain" in text and "life" in text:
        gain_match = re.search(r"gain (\d+) life", text)
        if gain_match:
            player.life += int(gain_match.group(1))


def _execute_attack(state: GameState, player: Player, action: Action) -> None:
    """Execute declaring attackers."""
    for attacker in action.attackers:
        attacker.attacking = True
        attacker.tap()  # Most attackers tap (unless vigilance)
        if "vigilance" in attacker.card.keywords:
            attacker.tapped = False


def _execute_block(state: GameState, player: Player, action: Action) -> None:
    """Execute declaring a blocker."""
    blocker = action.blocker
    blocking = action.blocking

    blocker.blocking = blocking
    blocking.blocked_by.append(blocker)


def _execute_pass(state: GameState) -> None:
    """Execute passing (advancing the phase)."""
    match state.phase:
        case Phase.UNTAP:
            state.untap_step()
            state.phase = Phase.UPKEEP
        case Phase.UPKEEP:
            state.phase = Phase.DRAW
        case Phase.DRAW:
            state.draw_step()
            state.phase = Phase.MAIN_1
        case Phase.MAIN_1:
            state.phase = Phase.COMBAT
        case Phase.COMBAT:
            # Resolve combat damage
            _resolve_combat(state)
            state.phase = Phase.MAIN_2
        case Phase.MAIN_2:
            state.phase = Phase.END
        case Phase.END:
            state.end_turn()


def _resolve_combat(state: GameState) -> None:
    """Resolve combat damage."""
    attacker_player = state.active_player
    defender_player = state.defending_player

    for attacker in attacker_player.creatures:
        if not attacker.attacking:
            continue

        if attacker.blocked_by:
            # Blocked - damage goes to blockers
            # Simplified: just deal damage to first blocker
            blocker = attacker.blocked_by[0]

            # Attacker damages blocker
            blocker.take_damage(attacker.current_power)

            # Blocker damages attacker
            attacker.take_damage(blocker.current_power)

            # Handle first strike, deathtouch, etc.
            if "deathtouch" in attacker.card.keywords and attacker.current_power > 0:
                blocker.damage_marked = blocker.current_toughness  # Lethal
            if "deathtouch" in blocker.card.keywords and blocker.current_power > 0:
                attacker.damage_marked = attacker.current_toughness

            # Handle lifelink
            if "lifelink" in attacker.card.keywords:
                attacker_player.life += attacker.current_power
            if "lifelink" in blocker.card.keywords:
                defender_player.life += blocker.current_power
        else:
            # Unblocked - damage goes to player
            damage = attacker.current_power
            defender_player.life -= damage

            # Handle lifelink
            if "lifelink" in attacker.card.keywords:
                attacker_player.life += damage

    # Reset combat state
    for creature in attacker_player.creatures:
        creature.attacking = False
        creature.blocked_by = []
    for creature in defender_player.creatures:
        creature.blocking = None


def get_action_space_size() -> int:
    """
    Return the maximum action space size for ML purposes.
    This is an approximation for neural network output sizing.
    """
    # Roughly: 7 cards in hand * 2 (play/cast) + 10 attackers * 2^10 subsets + pass
    # We'll use a more practical fixed size
    return 256
