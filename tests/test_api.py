"""
Tests for the API endpoints.

Uses FastAPI TestClient for synchronous testing.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.models import (
    RecommendCardsRequest,
    AnalyzeDeckRequest,
    SynergyRequest,
    SimulateRequest,
)
from src.api.services import get_health_status


# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "database_connected" in data
        assert "cards_loaded" in data
        assert "commanders_available" in data

    def test_health_status_service(self):
        """Test health status service directly."""
        status = get_health_status()
        assert status.status in ["healthy", "degraded"]
        assert isinstance(status.cards_loaded, int)


class TestCommandersEndpoint:
    """Tests for commanders listing endpoint."""

    def test_list_commanders(self):
        """Test listing available commanders."""
        response = client.get("/commanders")
        assert response.status_code == 200

        data = response.json()
        assert "commanders" in data
        assert "count" in data
        assert isinstance(data["commanders"], list)

    def test_list_commanders_with_filters(self):
        """Test listing commanders with color filter."""
        response = client.get("/commanders?min_decks=50&limit=10")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] <= 10

    def test_list_commanders_by_color(self):
        """Test filtering commanders by color identity."""
        # This may return empty if no commanders match
        response = client.get("/commanders?color_identity=UG&limit=5")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["commanders"], list)


class TestRecommendEndpoint:
    """Tests for card recommendation endpoint."""

    def test_recommend_cards_basic(self):
        """Test basic card recommendation."""
        # Get a commander from the database first
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        response = client.post(
            "/recommend/cards",
            json={"commander": commander, "count": 5},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["commander"] == commander
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 5

    def test_recommend_cards_not_found(self):
        """Test recommendation with invalid commander."""
        response = client.post(
            "/recommend/cards",
            json={"commander": "Nonexistent Commander XYZ123", "count": 5},
        )
        assert response.status_code == 404

    def test_recommend_cards_with_exclusions(self):
        """Test recommendations with excluded cards."""
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        response = client.post(
            "/recommend/cards",
            json={
                "commander": commander,
                "count": 10,
                "exclude": ["Sol Ring", "Command Tower"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        card_names = [r["name"] for r in data["recommendations"]]
        assert "Sol Ring" not in card_names
        assert "Command Tower" not in card_names

    def test_recommend_cards_validation(self):
        """Test request validation."""
        # Count too high
        response = client.post(
            "/recommend/cards",
            json={"commander": "Test", "count": 500},
        )
        assert response.status_code == 422  # Validation error

        # Missing commander
        response = client.post(
            "/recommend/cards",
            json={"count": 10},
        )
        assert response.status_code == 422


class TestAnalyzeDeckEndpoint:
    """Tests for deck analysis endpoint."""

    def test_analyze_deck_basic(self):
        """Test basic deck analysis."""
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        # Create a minimal deck
        decklist = ["Sol Ring", "Arcane Signet", "Command Tower"] + [
            f"Forest" for _ in range(40)
        ]

        response = client.post(
            "/analyze/deck",
            json={
                "commander": commander,
                "decklist": decklist,
                "num_simulations": 3,
                "max_turns": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "analysis" in data
        assert "card_performance" in data
        assert "suggestions" in data

    def test_analyze_deck_not_found(self):
        """Test analysis with invalid commander."""
        response = client.post(
            "/analyze/deck",
            json={
                "commander": "Nonexistent Commander XYZ123",
                "decklist": ["Forest"] * 40,
                "num_simulations": 1,
            },
        )
        assert response.status_code == 404

    def test_analyze_deck_response_structure(self):
        """Test deck analysis response structure."""
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]
        decklist = ["Forest"] * 45

        response = client.post(
            "/analyze/deck",
            json={
                "commander": commander,
                "decklist": decklist,
                "num_simulations": 2,
                "max_turns": 3,
            },
        )
        assert response.status_code == 200

        data = response.json()
        analysis = data["analysis"]

        # Check required fields
        assert "total_cards" in analysis
        assert "avg_cmc" in analysis
        assert "land_count" in analysis
        assert "cmc_distribution" in analysis
        assert "avg_damage" in analysis


class TestSynergyEndpoint:
    """Tests for synergy analysis endpoint."""

    def test_synergy_basic(self):
        """Test basic synergy analysis."""
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        response = client.post(
            "/analyze/synergy",
            json={
                "commander": commander,
                "cards": ["Sol Ring", "Arcane Signet"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["commander"] == commander
        assert "card_synergies" in data
        assert "pair_synergies" in data
        assert "average_synergy" in data

    def test_synergy_not_found(self):
        """Test synergy with invalid commander."""
        response = client.post(
            "/analyze/synergy",
            json={
                "commander": "Nonexistent Commander XYZ123",
                "cards": ["Sol Ring", "Arcane Signet"],
            },
        )
        assert response.status_code == 404

    def test_synergy_validation(self):
        """Test synergy request validation."""
        # Too few cards
        response = client.post(
            "/analyze/synergy",
            json={
                "commander": "Test",
                "cards": ["Sol Ring"],  # Need at least 2
            },
        )
        assert response.status_code == 422


class TestSimulateEndpoint:
    """Tests for goldfish simulation endpoint."""

    def test_simulate_basic(self):
        """Test basic simulation."""
        # Create a valid decklist
        decklist = (
            ["Forest"] * 40
            + ["Llanowar Elves", "Elvish Mystic"] * 5
        )

        response = client.post(
            "/simulate/goldfish",
            json={
                "decklist": decklist,
                "num_games": 3,
                "max_turns": 5,
            },
        )
        # May return 400 if cards not resolved, or 200 if successful
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert "num_games" in data
            assert "results" in data
            assert "avg_damage" in data
            assert "win_rate" in data

    def test_simulate_with_seed(self):
        """Test simulation with seed for reproducibility."""
        decklist = ["Forest"] * 50

        response1 = client.post(
            "/simulate/goldfish",
            json={
                "decklist": decklist,
                "num_games": 3,
                "max_turns": 5,
                "seed": 42,
            },
        )

        response2 = client.post(
            "/simulate/goldfish",
            json={
                "decklist": decklist,
                "num_games": 3,
                "max_turns": 5,
                "seed": 42,
            },
        )

        # Both should have same status
        assert response1.status_code == response2.status_code

        if response1.status_code == 200:
            # Results should be identical with same seed
            data1 = response1.json()
            data2 = response2.json()
            assert data1["avg_damage"] == data2["avg_damage"]

    def test_simulate_validation(self):
        """Test simulation request validation."""
        # Too few cards
        response = client.post(
            "/simulate/goldfish",
            json={
                "decklist": ["Forest"] * 10,  # Need at least 40
                "num_games": 1,
            },
        )
        assert response.status_code == 422


class TestAPIIntegration:
    """Integration tests for API workflows."""

    def test_recommend_then_analyze_workflow(self):
        """Test workflow: get recommendations, build deck, analyze it."""
        # Get a commander
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        # Get recommendations
        rec_response = client.post(
            "/recommend/cards",
            json={"commander": commander, "count": 20},
        )
        if rec_response.status_code != 200:
            pytest.skip("No recommendations available")

        recommendations = rec_response.json()["recommendations"]
        if len(recommendations) < 10:
            pytest.skip("Not enough recommendations")

        # Build a deck from recommendations + lands
        decklist = [r["name"] for r in recommendations[:20]] + ["Forest"] * 35

        # Analyze the deck
        analyze_response = client.post(
            "/analyze/deck",
            json={
                "commander": commander,
                "decklist": decklist,
                "num_simulations": 2,
                "max_turns": 5,
            },
        )
        assert analyze_response.status_code == 200

        analysis = analyze_response.json()
        assert analysis["analysis"]["total_cards"] > 0

    def test_synergy_for_recommended_cards(self):
        """Test checking synergy for recommended cards."""
        commanders_response = client.get("/commanders?limit=1")
        if commanders_response.json()["count"] == 0:
            pytest.skip("No commanders in database")

        commander = commanders_response.json()["commanders"][0]["name"]

        # Get recommendations
        rec_response = client.post(
            "/recommend/cards",
            json={"commander": commander, "count": 5},
        )
        if rec_response.status_code != 200:
            pytest.skip("No recommendations available")

        recommendations = rec_response.json()["recommendations"]
        if len(recommendations) < 2:
            pytest.skip("Not enough recommendations")

        # Check synergy between recommended cards
        cards = [r["name"] for r in recommendations[:3]]
        synergy_response = client.post(
            "/analyze/synergy",
            json={"commander": commander, "cards": cards},
        )
        assert synergy_response.status_code == 200

        synergy = synergy_response.json()
        assert synergy["average_synergy"] >= 0  # Should have some synergy


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/recommend/cards",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        response = client.post(
            "/recommend/cards",
            json={},  # Missing commander
        )
        assert response.status_code == 422

    def test_invalid_field_types(self):
        """Test handling of invalid field types."""
        response = client.post(
            "/recommend/cards",
            json={"commander": 123, "count": "not a number"},
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
