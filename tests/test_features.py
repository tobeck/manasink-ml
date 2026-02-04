"""
Tests for card feature extraction.
"""

import pytest
from src.data.features import (
    CardFeatures,
    extract_features_from_scryfall,
    _parse_mana_cost,
    _encode_keywords,
    _encode_color_identity,
    _compute_role_scores,
)


class TestManaCostParsing:
    def test_parse_simple_cost(self):
        w, u, b, r, g, c = _parse_mana_cost("{2}{G}{G}")
        assert w == 0
        assert u == 0
        assert b == 0
        assert r == 0
        assert g == 2
        assert c == 2

    def test_parse_multicolor(self):
        w, u, b, r, g, c = _parse_mana_cost("{1}{W}{U}{B}{R}{G}")
        assert w == 1
        assert u == 1
        assert b == 1
        assert r == 1
        assert g == 1
        assert c == 1

    def test_parse_empty(self):
        w, u, b, r, g, c = _parse_mana_cost("")
        assert w == u == b == r == g == c == 0

    def test_parse_no_generic(self):
        w, u, b, r, g, c = _parse_mana_cost("{U}{U}{U}")
        assert u == 3
        assert c == 0


class TestKeywordEncoding:
    def test_encode_flying(self):
        bitmap = _encode_keywords("Flying", ["Flying"])
        assert bitmap & 1  # flying is first in list

    def test_encode_multiple(self):
        text = "Flying, First Strike, Lifelink"
        bitmap = _encode_keywords(text, ["Flying", "First strike", "Lifelink"])
        assert bitmap != 0

    def test_encode_from_text(self):
        text = "This creature has flying and trample."
        bitmap = _encode_keywords(text, [])
        assert bitmap != 0  # Should detect flying and trample

    def test_encode_empty(self):
        bitmap = _encode_keywords("", [])
        assert bitmap == 0


class TestColorIdentityEncoding:
    def test_encode_simic(self):
        bitmap = _encode_color_identity(["U", "G"])
        assert bitmap == 0b10010  # U=bit1, G=bit4

    def test_encode_mono(self):
        bitmap = _encode_color_identity(["W"])
        assert bitmap == 0b00001

    def test_encode_five_color(self):
        bitmap = _encode_color_identity(["W", "U", "B", "R", "G"])
        assert bitmap == 0b11111

    def test_encode_empty(self):
        bitmap = _encode_color_identity([])
        assert bitmap == 0


class TestRoleScores:
    def test_ramp_detection(self):
        text = "Add {G}. Search your library for a basic land card and put it onto the battlefield."
        scores = _compute_role_scores(text, "Creature")
        assert scores["ramp"] > 0

    def test_card_draw_detection(self):
        text = "Draw two cards."
        scores = _compute_role_scores(text, "Instant")
        assert scores["card_draw"] > 0

    def test_removal_detection(self):
        text = "Destroy target creature."
        scores = _compute_role_scores(text, "Instant")
        assert scores["removal"] > 0

    def test_board_wipe_detection(self):
        text = "Destroy all creatures."
        scores = _compute_role_scores(text, "Sorcery")
        assert scores["board_wipe"] > 0

    def test_empty_text(self):
        scores = _compute_role_scores("", "Creature")
        assert all(score == 0 for score in scores.values())


class TestFeatureExtraction:
    def test_extract_creature(self):
        scryfall_data = {
            "id": "test-id-123",
            "name": "Llanowar Elves",
            "mana_cost": "{G}",
            "cmc": 1,
            "type_line": "Creature — Elf Druid",
            "oracle_text": "{T}: Add {G}.",
            "power": "1",
            "toughness": "1",
            "color_identity": ["G"],
            "keywords": [],
        }

        features = extract_features_from_scryfall(scryfall_data)

        assert features.card_name == "Llanowar Elves"
        assert features.cmc == 1
        assert features.mana_g == 1
        assert features.is_creature is True
        assert features.is_land is False
        assert features.power == 1
        assert features.toughness == 1
        assert features.role_ramp > 0  # "{T}: Add {G}"

    def test_extract_instant(self):
        scryfall_data = {
            "id": "test-id-456",
            "name": "Counterspell",
            "mana_cost": "{U}{U}",
            "cmc": 2,
            "type_line": "Instant",
            "oracle_text": "Counter target spell.",
            "color_identity": ["U"],
            "keywords": [],
        }

        features = extract_features_from_scryfall(scryfall_data)

        assert features.card_name == "Counterspell"
        assert features.mana_u == 2
        assert features.is_instant is True
        assert features.is_creature is False
        assert features.role_removal > 0  # Counter is removal

    def test_extract_land(self):
        scryfall_data = {
            "id": "test-id-789",
            "name": "Forest",
            "mana_cost": "",
            "cmc": 0,
            "type_line": "Basic Land — Forest",
            "oracle_text": "({T}: Add {G}.)",
            "color_identity": ["G"],
            "keywords": [],
        }

        features = extract_features_from_scryfall(scryfall_data)

        assert features.card_name == "Forest"
        assert features.cmc == 0
        assert features.is_land is True
        assert features.role_ramp >= 0.5  # Lands get ramp boost

    def test_feature_vector(self):
        features = CardFeatures(
            scryfall_id="test",
            card_name="Test Card",
            mana_g=2,
            cmc=2,
            is_creature=True,
        )

        vector = features.to_vector()
        assert len(vector) == CardFeatures.vector_size()
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)

    def test_legendary_detection(self):
        scryfall_data = {
            "id": "test-legendary",
            "name": "Atraxa, Praetors' Voice",
            "mana_cost": "{G}{W}{U}{B}",
            "cmc": 4,
            "type_line": "Legendary Creature — Phyrexian Angel Horror",
            "oracle_text": "Flying, vigilance, deathtouch, lifelink\nAt the beginning of your end step, proliferate.",
            "power": "4",
            "toughness": "4",
            "color_identity": ["W", "U", "B", "G"],
            "keywords": ["Flying", "Vigilance", "Deathtouch", "Lifelink"],
        }

        features = extract_features_from_scryfall(scryfall_data)

        assert features.is_legendary is True
        assert features.is_creature is True
        assert features.keyword_bitmap != 0  # Has keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
