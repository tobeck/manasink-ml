"""
Tests for RL training infrastructure.

Tests cover:
- StateEncoder: Encoding game states to tensors
- PolicyNetwork: Forward pass and action selection
- Trainer: Short training run to verify infrastructure
"""

import pytest
import torch

from src.game import (
    create_game,
    create_test_deck,
    Color,
    get_legal_actions,
    execute_action,
    Action,
    ActionType,
)
from src.models import (
    StateEncoder,
    EncodedState,
    PolicyNetwork,
    NeuralPolicy,
    Trainer,
    TrainingConfig,
    CARD_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
)


class TestStateEncoder:
    """Tests for StateEncoder."""

    def test_encoder_creation(self):
        """Test encoder can be created with default settings."""
        encoder = StateEncoder()
        assert encoder.card_dim == CARD_FEATURE_DIM
        assert encoder.max_hand == 10
        assert encoder.max_battlefield == 20

    def test_encoder_custom_settings(self):
        """Test encoder with custom settings."""
        encoder = StateEncoder(card_dim=32, max_hand=7, max_battlefield=15)
        assert encoder.card_dim == 32
        assert encoder.max_hand == 7
        assert encoder.max_battlefield == 15

    def test_get_state_dim(self):
        """Test state dimension calculation."""
        encoder = StateEncoder()
        state_dim = encoder.get_state_dim()

        # hand + battlefield + opponent + global
        expected = (
            10 * CARD_FEATURE_DIM  # hand
            + 20 * CARD_FEATURE_DIM  # battlefield
            + 20 * CARD_FEATURE_DIM  # opponent
            + GLOBAL_FEATURE_DIM  # global
        )
        assert state_dim == expected

    def test_encode_state_basic(self):
        """Test encoding a basic game state."""
        # Create a simple game
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        encoded = encoder.encode_state(state)

        # Check output types
        assert isinstance(encoded, EncodedState)
        assert isinstance(encoded.hand, torch.Tensor)
        assert isinstance(encoded.battlefield, torch.Tensor)
        assert isinstance(encoded.global_features, torch.Tensor)

    def test_encode_state_shapes(self):
        """Test encoded state tensor shapes."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        encoded = encoder.encode_state(state)

        assert encoded.hand.shape == (10, CARD_FEATURE_DIM)
        assert encoded.battlefield.shape == (20, CARD_FEATURE_DIM)
        assert encoded.opponent_battlefield.shape == (20, CARD_FEATURE_DIM)
        assert encoded.global_features.shape == (GLOBAL_FEATURE_DIM,)

        # Mask shapes
        assert encoded.hand_mask.shape == (10,)
        assert encoded.battlefield_mask.shape == (20,)
        assert encoded.opponent_battlefield_mask.shape == (20,)

    def test_encode_state_hand_mask(self):
        """Test that hand mask reflects actual hand size."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        encoded = encoder.encode_state(state)

        # Hand should have 7 cards after setup
        hand_size = len(state.active_player.hand)
        assert hand_size == 7

        # Mask should have 1s for actual cards, 0s for padding
        assert encoded.hand_mask[:hand_size].sum() == hand_size
        assert encoded.hand_mask[hand_size:].sum() == 0

    def test_encode_card(self):
        """Test encoding a single card."""
        deck = create_test_deck({Color.GREEN})
        card = deck[0]  # Get a card

        encoder = StateEncoder()
        encoded_card = encoder.encode_card(card)

        assert encoded_card.shape == (CARD_FEATURE_DIM,)
        assert encoded_card.dtype == torch.float32

    def test_encode_card_caching(self):
        """Test that card encoding is cached."""
        deck = create_test_deck({Color.GREEN})
        card = deck[0]

        encoder = StateEncoder()

        # First encoding
        encoded1 = encoder.encode_card(card)

        # Should be cached
        assert card.name in encoder._card_cache

        # Second encoding should return same tensor
        encoded2 = encoder.encode_card(card)
        assert torch.equal(encoded1, encoded2)

    def test_encode_global_features(self):
        """Test global features encoding."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        encoded = encoder.encode_state(state)

        global_features = encoded.global_features

        # Check specific features
        # Turn 1 normalized: 1/20 = 0.05
        assert global_features[0].item() == pytest.approx(0.05, abs=0.01)

        # Life totals (40/40 = 1.0)
        assert global_features[1].item() == pytest.approx(1.0, abs=0.01)  # Our life
        assert global_features[2].item() == pytest.approx(1.0, abs=0.01)  # Opponent life

    def test_encoded_state_to_device(self):
        """Test moving encoded state to device."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        encoded = encoder.encode_state(state)

        # Move to CPU (should work even if already there)
        encoded_cpu = encoded.to(torch.device("cpu"))
        assert encoded_cpu.hand.device.type == "cpu"


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_network_creation(self):
        """Test network can be created."""
        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())

        assert network.hidden_dim == 256
        assert network.num_actions == 256

    def test_network_custom_settings(self):
        """Test network with custom settings."""
        encoder = StateEncoder()
        network = PolicyNetwork(
            state_dim=encoder.get_state_dim(),
            hidden_dim=128,
            num_actions=64,
        )

        assert network.hidden_dim == 128
        assert network.num_actions == 64

    def test_forward_pass(self):
        """Test network forward pass."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())

        encoded = encoder.encode_state(state)
        logits, value = network(encoded)

        assert logits.shape == (256,)  # num_actions
        assert value.shape == (1,) or value.shape == ()

    def test_forward_pass_with_batch(self):
        """Test network forward pass with batch dimension."""
        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())

        # Create batched input
        batch_size = 4
        hand = torch.randn(batch_size, 10, CARD_FEATURE_DIM)
        battlefield = torch.randn(batch_size, 20, CARD_FEATURE_DIM)
        opponent_bf = torch.randn(batch_size, 20, CARD_FEATURE_DIM)
        global_feat = torch.randn(batch_size, GLOBAL_FEATURE_DIM)
        hand_mask = torch.ones(batch_size, 10)
        bf_mask = torch.ones(batch_size, 20)
        opp_mask = torch.ones(batch_size, 20)

        encoded = EncodedState(
            hand=hand,
            battlefield=battlefield,
            opponent_battlefield=opponent_bf,
            global_features=global_feat,
            hand_mask=hand_mask,
            battlefield_mask=bf_mask,
            opponent_battlefield_mask=opp_mask,
        )

        logits, value = network(encoded)

        assert logits.shape == (batch_size, 256)
        assert value.shape == (batch_size, 1)

    def test_forward_with_action_mask(self):
        """Test forward pass with action masking."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())

        encoded = encoder.encode_state(state)

        # Create mask that only allows first 3 actions
        action_mask = torch.zeros(256)
        action_mask[:3] = 1.0

        logits, value = network(encoded, action_mask)

        # Masked actions should have -inf logits
        assert torch.isinf(logits[3:]).all()
        assert not torch.isinf(logits[:3]).any()

    def test_get_action_probs(self):
        """Test getting action probabilities."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())

        encoded = encoder.encode_state(state)
        probs = network.get_action_probs(encoded)

        # Probabilities should sum to 1
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        # All probabilities should be non-negative
        assert (probs >= 0).all()


class TestNeuralPolicy:
    """Tests for NeuralPolicy wrapper."""

    def test_policy_creation(self):
        """Test policy can be created."""
        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        assert policy.temperature == 1.0
        assert not policy.deterministic

    def test_select_action(self):
        """Test action selection."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        legal_actions = get_legal_actions(state)
        action = policy.select_action(state, legal_actions)

        # Should return a valid action
        assert isinstance(action, Action)
        assert action in legal_actions or action.action_type == ActionType.PASS

    def test_select_action_deterministic(self):
        """Test deterministic action selection."""
        state = create_game(seed=42)
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder, deterministic=True)

        legal_actions = get_legal_actions(state)

        # Same state should give same action
        action1 = policy.select_action(state, legal_actions)
        action2 = policy.select_action(state, legal_actions)

        assert action1 == action2

    def test_get_action_and_value(self):
        """Test getting action with value and log_prob."""
        state = create_game()
        deck = create_test_deck({Color.GREEN})
        state.setup_game(deck, deck)

        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        legal_actions = get_legal_actions(state)
        action, log_prob, value, entropy = policy.get_action_and_value(
            state, legal_actions
        )

        assert isinstance(action, Action)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert isinstance(entropy, torch.Tensor)

    def test_policy_modes(self):
        """Test eval and train modes."""
        encoder = StateEncoder()
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        policy.set_eval_mode()
        assert not network.training

        policy.set_train_mode()
        assert network.training


class TestTrainer:
    """Tests for Trainer."""

    def test_trainer_creation(self):
        """Test trainer can be created."""
        config = TrainingConfig(num_episodes=10)
        trainer = Trainer(config)

        assert trainer.total_episodes == 0
        assert trainer.total_steps == 0

    def test_trainer_with_synergy_context(self):
        """Test trainer with synergy context."""
        from src.game import create_empty_context

        config = TrainingConfig(num_episodes=10)
        context = create_empty_context("Test Commander")
        trainer = Trainer(config, synergy_context=context)

        assert trainer.synergy_context is not None

    def test_collect_episode(self):
        """Test collecting a single episode."""
        config = TrainingConfig(max_turns=5)
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        metrics = trainer.collect_episode(deck, seed=42)

        assert metrics.episode_length > 0
        assert metrics.damage_dealt >= 0
        assert len(trainer.buffer) == metrics.episode_length

    def test_short_training_run(self):
        """Test a short training run completes without errors."""
        config = TrainingConfig(
            num_episodes=5,
            batch_size=4,
            max_turns=5,
            log_every=10,  # Don't log during test
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        metrics = trainer.train(deck, num_episodes=5, verbose=False)

        assert len(metrics) == 5
        assert trainer.total_episodes == 5
        assert trainer.total_steps > 0

    def test_training_improves_or_stable(self):
        """Test that training doesn't completely break."""
        config = TrainingConfig(
            num_episodes=10,
            batch_size=8,
            max_turns=5,
            learning_rate=1e-4,
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        metrics = trainer.train(deck, num_episodes=10, verbose=False)

        # Just verify we got valid metrics
        for m in metrics:
            assert m.episode_length >= 0
            assert m.damage_dealt >= 0
            # Total reward could be negative or positive

    def test_get_policy(self):
        """Test getting trained policy."""
        config = TrainingConfig(num_episodes=2, max_turns=3)
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        trainer.train(deck, verbose=False)

        policy = trainer.get_policy()
        assert isinstance(policy, NeuralPolicy)
        assert policy.deterministic  # Inference mode

    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        config = TrainingConfig(num_episodes=2, max_turns=3)
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})
        trainer.train(deck, verbose=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load into new trainer
        new_trainer = Trainer(config)
        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.total_episodes == trainer.total_episodes

    def test_ppo_update(self):
        """Test PPO update step."""
        config = TrainingConfig(
            num_episodes=10,
            batch_size=8,
            max_turns=5,
        )
        trainer = Trainer(config)

        deck = create_test_deck({Color.GREEN})

        # Collect enough experiences
        for _ in range(8):
            trainer.collect_episode(deck)

        # Run PPO update
        update_metrics = trainer.ppo_update()

        assert "policy_loss" in update_metrics
        assert "value_loss" in update_metrics
        assert "entropy" in update_metrics

        # Losses should be finite
        assert not torch.isnan(torch.tensor(update_metrics["policy_loss"]))
        assert not torch.isnan(torch.tensor(update_metrics["value_loss"]))


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.num_episodes == 1000
        assert config.batch_size == 32
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.clip_epsilon == 0.2

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            num_episodes=500,
            learning_rate=1e-3,
            clip_epsilon=0.1,
        )

        assert config.num_episodes == 500
        assert config.learning_rate == 1e-3
        assert config.clip_epsilon == 0.1


class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_full_pipeline_with_test_deck(self):
        """Test full training pipeline with test deck."""
        # Create deck
        deck = create_test_deck({Color.GREEN})

        # Configure training
        config = TrainingConfig(
            num_episodes=3,
            batch_size=3,
            max_turns=5,
            checkpoint_every=100,  # Don't checkpoint during test
        )

        # Train
        trainer = Trainer(config)
        metrics = trainer.train(deck, verbose=False)

        # Verify training completed
        assert len(metrics) == 3

        # Get policy and run inference
        policy = trainer.get_policy()
        policy.set_eval_mode()

        # Test policy on a new game
        state = create_game(seed=123)
        state.setup_game(deck, [], None, None)

        legal_actions = get_legal_actions(state)
        action = policy.select_action(state, legal_actions)

        assert action is not None

    def test_training_with_synergy_rewards(self):
        """Test training with synergy reward shaping."""
        from src.game import create_empty_context

        deck = create_test_deck({Color.GREEN})

        # Create context with some fake synergies
        context = create_empty_context("Test Commander")
        context.card_synergies["Test Creature 1"] = 0.5
        context.card_synergies["Test Creature 2"] = 0.3

        config = TrainingConfig(
            num_episodes=3,
            batch_size=3,
            max_turns=5,
            synergy_reward_weight=0.2,
        )

        trainer = Trainer(config, synergy_context=context)
        metrics = trainer.train(deck, verbose=False)

        # Training should complete
        assert len(metrics) == 3

        # Synergy bonus should be tracked (might be 0 if cards not played)
        # Just verify it's a valid number
        for m in metrics:
            assert isinstance(m.synergy_bonus, float)
