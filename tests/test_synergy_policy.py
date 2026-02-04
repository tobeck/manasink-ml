"""
Tests for synergy-aware policy.
"""

import pytest
from src.game import (
    Card, ManaCost, CardType, Color,
    GameState, create_game, Phase,
    Action, ActionType, BASIC_LANDS,
)
from src.game.synergy_policy import (
    SynergyAwarePolicy,
    SynergyBonus,
    SynergyContext,
    create_empty_context,
)
from src.game.state import Permanent


class TestSynergyBonus:
    def test_empty_bonus(self):
        bonus = SynergyBonus()
        assert bonus.total_synergy == 0
        assert bonus.total_category_bonus == 0
        assert bonus.total_bonus == 0

    def test_with_synergy(self):
        bonus = SynergyBonus(
            cards_played=[("Sol Ring", 0.15), ("Mana Crypt", 0.30)],
            category_bonuses=[],
        )
        assert abs(bonus.total_synergy - 0.45) < 0.001
        assert bonus.total_category_bonus == 0

    def test_with_category_bonus(self):
        bonus = SynergyBonus(
            cards_played=[],
            category_bonuses=[("Cultivate", "Ramp", 10.0), ("Kodama's Reach", "Ramp", 10.0)],
        )
        assert bonus.total_synergy == 0
        assert bonus.total_category_bonus == 20.0

    def test_combined_bonus(self):
        bonus = SynergyBonus(
            cards_played=[("Sol Ring", 0.15)],
            category_bonuses=[("Sol Ring", "Ramp", 10.0)],
        )
        assert bonus.total_bonus == 10.15


class TestSynergyContext:
    def test_get_synergy(self):
        context = SynergyContext(
            commander_name="Test Commander",
            card_synergies={"Sol Ring": 0.15, "Mana Vault": 0.25},
            card_categories={},
            card_inclusions={},
        )

        assert context.get_synergy("Sol Ring") == 0.15
        assert context.get_synergy("Unknown") == 0.0

    def test_get_category(self):
        context = SynergyContext(
            commander_name="Test Commander",
            card_synergies={},
            card_categories={"Sol Ring": "Ramp"},
            card_inclusions={},
        )

        assert context.get_category("Sol Ring") == "Ramp"
        assert context.get_category("Unknown") == ""

    def test_create_empty_context(self):
        context = create_empty_context("My Commander")
        assert context.commander_name == "My Commander"
        assert len(context.card_synergies) == 0


class TestSynergyAwarePolicy:
    def setup_method(self):
        """Set up test fixtures."""
        self.context = SynergyContext(
            commander_name="Test Commander",
            card_synergies={
                "High Synergy Card": 0.50,
                "Medium Synergy Card": 0.25,
                "Low Synergy Card": 0.05,
            },
            card_categories={
                "Ramp Card": "Ramp",
                "Draw Card": "Card Draw",
                "Removal Card": "Removal",
            },
            card_inclusions={
                "High Synergy Card": 80.0,
                "Medium Synergy Card": 60.0,
            },
        )
        self.policy = SynergyAwarePolicy(self.context, synergy_weight=5.0)

    def test_prefers_lands_first(self):
        state = create_game(seed=42)
        state.phase = Phase.MAIN_1

        forest = BASIC_LANDS["Forest"]
        creature = Card(
            name="High Synergy Card",
            mana_cost=ManaCost(green=2),
            card_types={CardType.CREATURE},
        )

        # Both land and creature available
        actions = [
            Action(action_type=ActionType.PLAY_LAND, card=forest),
            Action(action_type=ActionType.CAST_SPELL, card=creature),
            Action(action_type=ActionType.PASS),
        ]

        selected = self.policy.select_action(state, actions)
        assert selected.action_type == ActionType.PLAY_LAND

    def test_prefers_high_synergy(self):
        state = create_game(seed=42)
        state.phase = Phase.MAIN_1
        state.players[0].land_played_this_turn = True  # Can't play land

        high_syn = Card(
            name="High Synergy Card",
            mana_cost=ManaCost(green=1),
            card_types={CardType.CREATURE},
        )
        low_syn = Card(
            name="Low Synergy Card",
            mana_cost=ManaCost(green=1),
            card_types={CardType.CREATURE},
        )

        actions = [
            Action(action_type=ActionType.CAST_SPELL, card=high_syn),
            Action(action_type=ActionType.CAST_SPELL, card=low_syn),
            Action(action_type=ActionType.PASS),
        ]

        selected = self.policy.select_action(state, actions)
        assert selected.card.name == "High Synergy Card"

    def test_tracks_synergy_bonus(self):
        state = create_game(seed=42)
        state.phase = Phase.MAIN_1
        state.players[0].land_played_this_turn = True

        high_syn = Card(
            name="High Synergy Card",
            mana_cost=ManaCost(green=1),
            card_types={CardType.CREATURE},
        )

        actions = [
            Action(action_type=ActionType.CAST_SPELL, card=high_syn),
            Action(action_type=ActionType.PASS),
        ]

        self.policy.reset_bonuses()
        self.policy.select_action(state, actions)

        bonuses = self.policy.get_episode_bonuses()
        assert len(bonuses.cards_played) == 1
        assert bonuses.cards_played[0] == ("High Synergy Card", 0.50)

    def test_category_bonus_ramp_early(self):
        """Ramp cards should get bonus in early game with few lands."""
        state = create_game(seed=42)
        state.turn_number = 2
        state.phase = Phase.MAIN_1
        state.players[0].land_played_this_turn = True
        # Only 2 lands on battlefield
        state.players[0].battlefield = [
            Permanent(card=BASIC_LANDS["Forest"], summoning_sick=False),
            Permanent(card=BASIC_LANDS["Forest"], summoning_sick=False),
        ]

        ramp = Card(
            name="Ramp Card",
            mana_cost=ManaCost(green=2),
            card_types={CardType.SORCERY},
        )
        other = Card(
            name="Some Card",
            mana_cost=ManaCost(green=2),
            card_types={CardType.SORCERY},
        )

        actions = [
            Action(action_type=ActionType.CAST_SPELL, card=ramp),
            Action(action_type=ActionType.CAST_SPELL, card=other),
        ]

        # Score the ramp card vs other card
        ramp_score = self.policy._score_action(actions[0], state)
        other_score = self.policy._score_action(actions[1], state)

        # Ramp should score higher due to category bonus
        assert ramp_score > other_score

    def test_category_bonus_draw_low_hand(self):
        """Card draw should get bonus when hand is low."""
        state = create_game(seed=42)
        state.phase = Phase.MAIN_1
        state.players[0].hand = []  # Empty hand

        draw = Card(
            name="Draw Card",
            mana_cost=ManaCost(blue=1),
            card_types={CardType.INSTANT},
        )

        action = Action(action_type=ActionType.CAST_SPELL, card=draw)
        bonus = self.policy._get_category_bonus("Card Draw", state)

        # Should get bonus for low hand
        assert bonus > 0

    def test_reset_bonuses(self):
        state = create_game(seed=42)
        state.phase = Phase.MAIN_1

        high_syn = Card(
            name="High Synergy Card",
            mana_cost=ManaCost(green=1),
            card_types={CardType.CREATURE},
        )

        actions = [
            Action(action_type=ActionType.CAST_SPELL, card=high_syn),
        ]

        # Play a card
        self.policy.select_action(state, actions)
        assert len(self.policy.get_episode_bonuses().cards_played) == 1

        # Reset
        self.policy.reset_bonuses()
        assert len(self.policy.get_episode_bonuses().cards_played) == 0

    def test_passes_when_no_actions(self):
        state = create_game(seed=42)
        selected = self.policy.select_action(state, [])
        assert selected.action_type == ActionType.PASS


class TestCategoryBonuses:
    def setup_method(self):
        self.context = create_empty_context("Test Commander")
        self.policy = SynergyAwarePolicy(self.context)

    def test_ramp_bonus_early_game(self):
        state = create_game(seed=42)
        state.turn_number = 3
        state.players[0].battlefield = [
            Permanent(card=BASIC_LANDS["Forest"], summoning_sick=False)
            for _ in range(3)
        ]

        bonus = self.policy._get_category_bonus("Ramp", state)
        assert bonus == 10  # Turn <= 4, lands < 6

    def test_ramp_no_bonus_late(self):
        state = create_game(seed=42)
        state.turn_number = 10
        state.players[0].battlefield = [
            Permanent(card=BASIC_LANDS["Forest"], summoning_sick=False)
            for _ in range(8)
        ]

        bonus = self.policy._get_category_bonus("Ramp", state)
        assert bonus == 0  # Too late, enough lands

    def test_removal_bonus_threats(self):
        state = create_game(seed=42)

        # Give opponent some creatures
        creature = Card(
            name="Threat",
            mana_cost=ManaCost(red=1),
            card_types={CardType.CREATURE},
        )
        state.players[1].battlefield = [
            Permanent(card=creature, summoning_sick=False)
            for _ in range(4)
        ]

        bonus = self.policy._get_category_bonus("Removal", state)
        assert bonus == 10  # Opponent has >= 3 creatures

    def test_board_wipe_bonus_behind(self):
        state = create_game(seed=42)

        creature = Card(
            name="Creature",
            mana_cost=ManaCost(red=1),
            card_types={CardType.CREATURE},
        )

        # We have 1 creature
        state.players[0].battlefield = [
            Permanent(card=creature, summoning_sick=False)
        ]

        # Opponent has 5 creatures
        state.players[1].battlefield = [
            Permanent(card=creature, summoning_sick=False)
            for _ in range(5)
        ]

        bonus = self.policy._get_category_bonus("Board Wipe", state)
        assert bonus == 15  # Significantly behind


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
