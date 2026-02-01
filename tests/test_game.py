"""
Basic tests for the game simulator.
"""

import pytest
from src.game import (
    Card, ManaCost, ManaPool, CardType, Color,
    GameState, Player, Permanent, Phase,
    Action, ActionType, get_legal_actions, execute_action,
    Simulator, GreedyPolicy, RandomPolicy, create_test_deck,
    create_game, BASIC_LANDS,
)


class TestManaCost:
    def test_parse_simple(self):
        cost = ManaCost.from_string("{2}{U}{U}")
        assert cost.colorless == 2
        assert cost.blue == 2
        assert cost.cmc == 4
    
    def test_parse_no_generic(self):
        cost = ManaCost.from_string("{G}{G}")
        assert cost.green == 2
        assert cost.colorless == 0
        assert cost.cmc == 2
    
    def test_can_pay(self):
        cost = ManaCost(colorless=1, blue=1)
        pool = ManaPool(blue=2)
        assert cost.can_pay_with(pool)
        
        pool2 = ManaPool(blue=1)
        assert not cost.can_pay_with(pool2)


class TestManaPool:
    def test_pay_colored(self):
        pool = ManaPool(blue=3, green=2)
        cost = ManaCost(blue=2, colorless=1)
        
        assert pool.pay(cost)
        assert pool.blue == 0  # Used 2 for blue, 1 for generic
        assert pool.green == 2
    
    def test_pay_insufficient(self):
        pool = ManaPool(blue=1)
        cost = ManaCost(blue=2)
        
        assert not pool.pay(cost)
        assert pool.blue == 1  # Unchanged


class TestCard:
    def test_basic_land(self):
        forest = BASIC_LANDS["Forest"]
        assert forest.is_land
        assert not forest.is_creature
        assert forest.cmc == 0
        assert Color.GREEN in forest.color_identity
    
    def test_creature(self):
        creature = Card(
            name="Test Bear",
            mana_cost=ManaCost(colorless=1, green=1),
            card_types={CardType.CREATURE},
            power=2,
            toughness=2,
            color_identity={Color.GREEN},
        )
        assert creature.is_creature
        assert creature.cmc == 2
        assert creature.power == 2


class TestGameState:
    def test_create_game(self):
        state = create_game("Alice", "Bob", seed=42)
        assert len(state.players) == 2
        assert state.players[0].name == "Alice"
        assert state.players[0].life == 40
    
    def test_draw(self):
        state = create_game(seed=42)
        deck = create_test_deck({Color.GREEN})
        state.players[0].library = deck.copy()
        
        drawn = state.players[0].draw(3)
        assert len(drawn) == 3
        assert len(state.players[0].hand) == 3
        assert len(state.players[0].library) == len(deck) - 3


class TestActions:
    def test_play_land(self):
        state = create_game(seed=42)
        player = state.players[0]
        player.hand = [BASIC_LANDS["Forest"]]
        state.phase = Phase.MAIN_1
        
        actions = get_legal_actions(state)
        land_actions = [a for a in actions if a.action_type == ActionType.PLAY_LAND]
        
        assert len(land_actions) == 1
        
        execute_action(state, land_actions[0])
        
        assert len(player.hand) == 0
        assert len(player.battlefield) == 1
        assert player.land_played_this_turn
    
    def test_cast_creature(self):
        state = create_game(seed=42)
        player = state.players[0]
        state.phase = Phase.MAIN_1
        
        # Give player a land and a creature
        forest = BASIC_LANDS["Forest"]
        creature = Card(
            name="Llanowar Elves",
            mana_cost=ManaCost(green=1),
            card_types={CardType.CREATURE},
            power=1,
            toughness=1,
            color_identity={Color.GREEN},
        )
        
        # Put a forest on battlefield (untapped)
        player.battlefield.append(Permanent(card=forest, summoning_sick=False))
        player.hand = [creature]
        
        actions = get_legal_actions(state)
        cast_actions = [a for a in actions if a.action_type == ActionType.CAST_SPELL]
        
        assert len(cast_actions) == 1
        
        execute_action(state, cast_actions[0])
        
        assert len(player.hand) == 0
        assert len([p for p in player.battlefield if p.is_creature]) == 1


class TestSimulator:
    def test_goldfish_runs(self):
        deck = create_test_deck({Color.GREEN})
        policy = GreedyPolicy()
        sim = Simulator(max_turns=5)
        
        result = sim.run_goldfish(deck, policy, seed=42)
        
        assert result.turns_to_kill <= 5 or result.total_damage > 0
        assert result.cards_played > 0
    
    def test_episode_runs(self):
        deck1 = create_test_deck({Color.GREEN})
        deck2 = create_test_deck({Color.RED})
        
        policy1 = GreedyPolicy()
        policy2 = RandomPolicy(seed=123)
        
        sim = Simulator(max_turns=5)
        result = sim.run_episode(deck1, deck2, policy1, policy2, seed=42)
        
        assert result.turns <= 6  # max_turns + 1 possible
        assert len(result.history) > 0


class TestPermanent:
    def test_summoning_sickness(self):
        creature = Card(
            name="Bear",
            mana_cost=ManaCost(colorless=1, green=1),
            card_types={CardType.CREATURE},
            power=2,
            toughness=2,
        )
        perm = Permanent(card=creature, summoning_sick=True)
        
        assert not perm.can_attack
        
        perm.summoning_sick = False
        assert perm.can_attack
    
    def test_haste(self):
        creature = Card(
            name="Hasty Bear",
            mana_cost=ManaCost(colorless=1, red=1),
            card_types={CardType.CREATURE},
            power=2,
            toughness=2,
            keywords={"haste"},
        )
        perm = Permanent(card=creature, summoning_sick=True)
        
        assert perm.can_attack  # Haste ignores summoning sickness


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
