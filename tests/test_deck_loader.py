"""
Tests for deck loader functionality.

Note: These tests require a populated database to fully work.
Tests that don't require database are marked accordingly.
"""

import pytest
from src.data.deck_loader import (
    DeckLoadResult,
    SynergyData,
    get_deck_stats,
)
from src.game import Card, ManaCost, CardType, Color


class TestDeckLoadResult:
    def test_is_complete_full_deck(self):
        commander = Card(
            name="Test Commander",
            mana_cost=ManaCost(),
            card_types={CardType.CREATURE},
            color_identity={Color.GREEN},
            is_commander=True,
        )

        # Create 99 cards
        deck = [
            Card(
                name=f"Card {i}",
                mana_cost=ManaCost(green=1),
                card_types={CardType.CREATURE},
            )
            for i in range(99)
        ]

        result = DeckLoadResult(
            commander=commander,
            deck=deck,
            missing_cards=[],
            color_identity={Color.GREEN},
        )

        assert result.is_complete is True
        assert result.deck_size == 99

    def test_is_complete_missing_cards(self):
        commander = Card(
            name="Test Commander",
            mana_cost=ManaCost(),
            card_types={CardType.CREATURE},
            color_identity={Color.GREEN},
            is_commander=True,
        )

        # Create 90 cards (9 missing)
        deck = [
            Card(
                name=f"Card {i}",
                mana_cost=ManaCost(green=1),
                card_types={CardType.CREATURE},
            )
            for i in range(90)
        ]

        result = DeckLoadResult(
            commander=commander,
            deck=deck,
            missing_cards=[f"Missing Card {i}" for i in range(9)],
            color_identity={Color.GREEN},
        )

        assert result.is_complete is False
        assert result.deck_size == 90


class TestSynergyData:
    def test_get_synergy(self):
        data = SynergyData(
            commander_name="Test Commander",
            card_synergies={"Sol Ring": 0.15, "Command Tower": 0.05},
            card_inclusions={"Sol Ring": 90.0, "Command Tower": 85.0},
            card_categories={"Sol Ring": "Ramp", "Command Tower": "Lands"},
        )

        assert data.get_synergy("Sol Ring") == 0.15
        assert data.get_synergy("Unknown Card") == 0.0

    def test_get_inclusion(self):
        data = SynergyData(
            commander_name="Test Commander",
            card_synergies={},
            card_inclusions={"Sol Ring": 90.0},
            card_categories={},
        )

        assert data.get_inclusion("Sol Ring") == 90.0
        assert data.get_inclusion("Unknown Card") == 0.0

    def test_get_category(self):
        data = SynergyData(
            commander_name="Test Commander",
            card_synergies={},
            card_inclusions={},
            card_categories={"Sol Ring": "Ramp", "Swords to Plowshares": "Removal"},
        )

        assert data.get_category("Sol Ring") == "Ramp"
        assert data.get_category("Unknown Card") == ""


class TestDeckStats:
    def test_get_deck_stats(self):
        commander = Card(
            name="Test Commander",
            mana_cost=ManaCost(green=3),
            card_types={CardType.CREATURE},
            color_identity={Color.GREEN},
            is_commander=True,
            power=4,
            toughness=4,
        )

        # Create a mix of card types
        deck = []

        # Add 35 lands
        for i in range(35):
            deck.append(
                Card(
                    name=f"Forest {i}",
                    mana_cost=ManaCost(),
                    card_types={CardType.LAND},
                    color_identity={Color.GREEN},
                )
            )

        # Add 30 creatures
        for i in range(30):
            deck.append(
                Card(
                    name=f"Creature {i}",
                    mana_cost=ManaCost(green=1, colorless=i % 6),
                    card_types={CardType.CREATURE},
                    power=i % 5 + 1,
                    toughness=i % 5 + 1,
                )
            )

        # Add 10 instants
        for i in range(10):
            deck.append(
                Card(
                    name=f"Instant {i}",
                    mana_cost=ManaCost(green=1),
                    card_types={CardType.INSTANT},
                )
            )

        # Add 10 sorceries
        for i in range(10):
            deck.append(
                Card(
                    name=f"Sorcery {i}",
                    mana_cost=ManaCost(green=2),
                    card_types={CardType.SORCERY},
                )
            )

        # Add 14 artifacts/enchantments to reach 99
        for i in range(7):
            deck.append(
                Card(
                    name=f"Artifact {i}",
                    mana_cost=ManaCost(colorless=3),
                    card_types={CardType.ARTIFACT},
                )
            )
            deck.append(
                Card(
                    name=f"Enchantment {i}",
                    mana_cost=ManaCost(green=1, colorless=2),
                    card_types={CardType.ENCHANTMENT},
                )
            )

        result = DeckLoadResult(
            commander=commander,
            deck=deck,
            missing_cards=[],
            color_identity={Color.GREEN},
        )

        stats = get_deck_stats(result)

        assert stats["commander"] == "Test Commander"
        assert stats["deck_size"] == 99
        assert stats["lands"] == 35
        assert stats["creatures"] == 30
        assert stats["instants"] == 10
        assert stats["sorceries"] == 10
        assert stats["artifacts"] == 7
        assert stats["enchantments"] == 7
        assert stats["is_complete"] is True
        assert "avg_cmc" in stats
        assert "cmc_distribution" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
