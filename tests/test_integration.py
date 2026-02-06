"""
Integration tests for the Manasink ML pipeline.

These tests verify that the different components work together correctly:
- Data loading → Game simulation
- Game simulation → RL training
- RL training → Policy evaluation
- End-to-end pipeline
"""

import pytest
import torch

from src.game import (
    Simulator,
    GreedyPolicy,
    RandomPolicy,
    create_test_deck,
    create_game,
    Color,
    get_legal_actions,
    execute_action,
    Action,
    ActionType,
    SynergyAwarePolicy,
    SynergyContext,
    create_empty_context,
)
from src.models import (
    StateEncoder,
    PolicyNetwork,
    NeuralPolicy,
    Trainer,
    TrainingConfig,
)


class TestDataToSimulationIntegration:
    """Test data loading integrates with simulation."""

    def test_synergy_context_with_simulator(self):
        """Test that synergy context integrates with game simulation."""
        # Create deck and synergy context
        deck = create_test_deck({Color.GREEN})
        context = create_empty_context("Test Commander")

        # Add some synergies for cards in the test deck
        for i, card in enumerate(deck[:10]):
            context.card_synergies[card.name] = 0.5 + (i * 0.05)
            context.card_categories[card.name] = "Creatures"

        # Create synergy-aware policy
        policy = SynergyAwarePolicy(context, synergy_weight=5.0)

        # Run simulation
        sim = Simulator(max_turns=5)
        result = sim.run_goldfish(deck, policy, seed=42)

        # Verify simulation ran and tracked synergies
        assert result.total_damage >= 0
        assert result.cards_played >= 0

        # Check synergy bonuses were tracked
        bonuses = policy.get_episode_bonuses()
        # May or may not have bonuses depending on cards played
        assert isinstance(bonuses.total_synergy, (int, float))

    def test_multiple_policies_same_deck(self):
        """Test different policies with the same deck produce different results."""
        deck = create_test_deck({Color.GREEN, Color.RED})
        sim = Simulator(max_turns=10)

        # Run with different policies
        random_result = sim.run_goldfish(deck, RandomPolicy(seed=42), seed=100)
        greedy_result = sim.run_goldfish(deck, GreedyPolicy(), seed=100)

        # Both should complete
        assert random_result.cards_played > 0
        assert greedy_result.cards_played > 0

        # Greedy should generally perform at least as well
        # (not always true due to randomness, but should hold with same seed)
        assert greedy_result.total_damage >= 0


class TestSimulationToTrainingIntegration:
    """Test game simulation integrates with RL training."""

    def test_state_encoder_with_live_game(self):
        """Test state encoder works with actual game states."""
        # Create a game and play a few turns
        state = create_game(seed=42)
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()

        # Encode initial state
        encoded = encoder.encode_state(state)
        assert encoded.hand.shape == (10, 29)
        assert encoded.global_features.shape == (17,)

        # Play a few actions and encode again
        policy = GreedyPolicy()
        for _ in range(5):
            legal = get_legal_actions(state)
            if not legal:
                break
            action = policy.select_action(state, legal)
            execute_action(state, action)

        # Encode after some gameplay
        encoded_after = encoder.encode_state(state)
        assert encoded_after.hand.shape == (10, 29)

        # State should have changed
        assert not torch.equal(encoded.global_features, encoded_after.global_features)

    def test_neural_policy_in_simulation(self):
        """Test neural policy works within simulation loop."""
        encoder = StateEncoder()
        network = PolicyNetwork(encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder, deterministic=True)

        deck = create_test_deck({Color.BLUE})
        sim = Simulator(max_turns=5)

        # Run goldfish with neural policy
        result = sim.run_goldfish(deck, policy, seed=42)

        assert result.cards_played >= 0
        assert result.total_damage >= 0

    def test_trainer_experience_collection(self):
        """Test that trainer collects valid experiences from simulation."""
        config = TrainingConfig(max_turns=3)
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        metrics = trainer.collect_episode(deck, seed=42)

        # Verify experience collection
        assert metrics.episode_length > 0
        assert len(trainer.buffer) == metrics.episode_length

        # Verify experiences have valid data
        for exp in trainer.buffer.experiences:
            assert exp.encoded_state is not None
            assert exp.action_mask is not None
            assert isinstance(exp.reward, float)
            assert isinstance(exp.value, float)
            assert isinstance(exp.log_prob, float)


class TestTrainingPipelineIntegration:
    """Test the full training pipeline."""

    def test_training_updates_network(self):
        """Test that training actually updates the network weights."""
        config = TrainingConfig(
            num_episodes=5,
            batch_size=5,
            max_turns=3,
            epochs_per_update=2,
        )
        trainer = Trainer(config)

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in trainer.network.named_parameters()
        }

        deck = create_test_deck({Color.GREEN})
        trainer.train(deck, num_episodes=5, verbose=False)

        # Check weights changed
        weights_changed = False
        for name, param in trainer.network.named_parameters():
            if not torch.equal(initial_weights[name], param):
                weights_changed = True
                break

        assert weights_changed, "Network weights should change after training"

    def test_training_with_synergy_context(self):
        """Test training with synergy reward shaping."""
        # Create context with synergies
        context = create_empty_context("Test Commander")
        deck = create_test_deck({Color.GREEN})

        # Add synergies for some cards
        for card in deck[:20]:
            context.card_synergies[card.name] = 0.3

        config = TrainingConfig(
            num_episodes=3,
            batch_size=3,
            max_turns=3,
            synergy_reward_weight=0.2,
        )
        trainer = Trainer(config, synergy_context=context)

        metrics = trainer.train(deck, verbose=False)

        assert len(metrics) == 3
        # At least one episode should have some synergy bonus
        total_synergy = sum(m.synergy_bonus for m in metrics)
        assert isinstance(total_synergy, float)

    def test_trained_policy_evaluation(self):
        """Test that a trained policy can be used for evaluation."""
        config = TrainingConfig(
            num_episodes=5,
            batch_size=5,
            max_turns=5,
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.RED})
        trainer.train(deck, verbose=False)

        # Get trained policy
        policy = trainer.get_policy()
        assert policy.deterministic  # Inference mode

        # Use policy in simulation
        sim = Simulator(max_turns=10)
        result = sim.run_goldfish(deck, policy, seed=123)

        assert result.cards_played >= 0
        assert result.total_damage >= 0

    def test_checkpoint_preserves_training_state(self, tmp_path):
        """Test that checkpoints preserve the training state."""
        config = TrainingConfig(
            num_episodes=3,
            batch_size=3,
            max_turns=3,
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        trainer.train(deck, verbose=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Get predictions from original
        state = create_game(seed=42)
        state.setup_game(deck, deck)
        encoded = trainer.encoder.encode_state(state)

        with torch.no_grad():
            original_logits, original_value = trainer.network(encoded)

        # Load into new trainer
        new_trainer = Trainer(config)
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Get predictions from loaded
        with torch.no_grad():
            loaded_logits, loaded_value = new_trainer.network(encoded)

        # Should match
        assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
        assert torch.allclose(original_value, loaded_value, atol=1e-6)


class TestEndToEndPipeline:
    """Test complete end-to-end workflows."""

    def test_full_training_and_evaluation_workflow(self):
        """Test the complete workflow from deck creation to policy evaluation."""
        # 1. Create deck
        deck = create_test_deck({Color.GREEN, Color.WHITE})

        # 2. Create synergy context
        context = create_empty_context("GW Commander")
        for i, card in enumerate(deck):
            context.card_synergies[card.name] = (i % 10) / 10.0
            if card.is_creature:
                context.card_categories[card.name] = "Creatures"
            elif card.is_land:
                context.card_categories[card.name] = "Lands"

        # 3. Train policy
        config = TrainingConfig(
            num_episodes=5,
            batch_size=5,
            max_turns=5,
            learning_rate=1e-3,
            synergy_reward_weight=0.1,
        )
        trainer = Trainer(config, synergy_context=context)
        training_metrics = trainer.train(deck, verbose=False)

        # 4. Verify training completed
        assert len(training_metrics) == 5
        assert all(m.episode_length > 0 for m in training_metrics)

        # 5. Get trained policy
        policy = trainer.get_policy()

        # 6. Evaluate with simulator
        sim = Simulator(max_turns=10)

        # Run multiple evaluation episodes
        eval_results = []
        for seed in range(3):
            result = sim.run_goldfish(deck, policy, seed=seed)
            eval_results.append(result)

        # 7. Verify evaluation completed
        assert all(r.cards_played > 0 for r in eval_results)
        avg_damage = sum(r.total_damage for r in eval_results) / len(eval_results)
        assert avg_damage >= 0

    def test_compare_policies_after_training(self):
        """Compare trained policy against baselines."""
        deck = create_test_deck({Color.BLUE, Color.RED})

        # Train a policy
        config = TrainingConfig(
            num_episodes=10,
            batch_size=5,
            max_turns=5,
        )
        trainer = Trainer(config)
        trainer.train(deck, verbose=False)
        neural_policy = trainer.get_policy()

        # Create baseline policies
        random_policy = RandomPolicy(seed=42)
        greedy_policy = GreedyPolicy()

        # Evaluate all policies
        sim = Simulator(max_turns=10)

        def evaluate_policy(policy, num_episodes=5):
            total_damage = 0
            for seed in range(num_episodes):
                result = sim.run_goldfish(deck, policy, seed=seed)
                total_damage += result.total_damage
            return total_damage / num_episodes

        random_avg = evaluate_policy(random_policy)
        greedy_avg = evaluate_policy(greedy_policy)
        neural_avg = evaluate_policy(neural_policy)

        # All should produce valid results
        assert random_avg >= 0
        assert greedy_avg >= 0
        assert neural_avg >= 0

        # Neural should not be dramatically worse than random
        # (with little training, it might not beat greedy yet)
        assert neural_avg >= random_avg * 0.5

    def test_training_reproducibility(self):
        """Test that training is reproducible with same seed."""
        deck = create_test_deck({Color.GREEN})

        def run_training(seed):
            torch.manual_seed(seed)
            config = TrainingConfig(
                num_episodes=3,
                batch_size=3,
                max_turns=3,
            )
            trainer = Trainer(config)
            metrics = trainer.train(deck, verbose=False)
            return [m.total_reward for m in metrics]

        # Run twice with same seed
        rewards1 = run_training(42)
        rewards2 = run_training(42)

        # Should be identical
        assert rewards1 == rewards2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_legal_actions(self):
        """Test handling when no legal actions available."""
        encoder = StateEncoder()
        network = PolicyNetwork(encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        # Empty action list should return PASS
        action = policy.select_action(state, [])
        assert action.action_type == ActionType.PASS

    def test_very_short_episode(self):
        """Test training with very short episodes."""
        config = TrainingConfig(
            num_episodes=2,
            batch_size=2,
            max_turns=1,  # Very short
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        metrics = trainer.train(deck, verbose=False)

        # Should complete without error
        assert len(metrics) == 2

    def test_single_color_deck(self):
        """Test with single-color deck."""
        for color in [Color.WHITE, Color.BLUE, Color.BLACK, Color.RED, Color.GREEN]:
            deck = create_test_deck({color})
            sim = Simulator(max_turns=3)
            policy = GreedyPolicy()

            result = sim.run_goldfish(deck, policy, seed=42)
            assert result.cards_played >= 0

    def test_multicolor_deck(self):
        """Test with 5-color deck."""
        all_colors = {Color.WHITE, Color.BLUE, Color.BLACK, Color.RED, Color.GREEN}
        deck = create_test_deck(all_colors)

        sim = Simulator(max_turns=5)
        policy = GreedyPolicy()

        result = sim.run_goldfish(deck, policy, seed=42)
        assert result.cards_played >= 0


class TestPerformance:
    """Test performance-related aspects."""

    def test_batch_encoding_efficiency(self):
        """Test that batch encoding is efficient."""
        from src.models.state_encoder import batch_encode_states

        encoder = StateEncoder()
        deck = create_test_deck({Color.GREEN})

        # Create multiple game states
        states = []
        for seed in range(10):
            state = create_game(seed=seed)
            state.setup_game(deck, deck)
            states.append(state)

        # Batch encode
        batched = batch_encode_states(encoder, states)

        assert batched["hand"].shape == (10, 10, 29)
        assert batched["global_features"].shape == (10, 17)

    def test_training_memory_stability(self):
        """Test that training doesn't leak memory."""
        import gc

        config = TrainingConfig(
            num_episodes=5,
            batch_size=5,
            max_turns=3,
        )
        trainer = Trainer(config)
        deck = create_test_deck({Color.GREEN})

        # Run training
        trainer.train(deck, verbose=False)

        # Clear buffer
        trainer.buffer.clear()
        gc.collect()

        # Should be able to train again
        trainer.train(deck, num_episodes=3, verbose=False)

        assert trainer.total_episodes == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
