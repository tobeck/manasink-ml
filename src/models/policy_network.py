"""
Neural network policy for MTG gameplay.

This module provides:
- PolicyNetwork: A neural network that outputs action probabilities and state values
- NeuralPolicy: A wrapper that implements the Policy interface for the Simulator

Architecture:
- Separate encoders for hand, battlefield, and opponent battlefield
- Attention-based aggregation for variable-length card sets
- Combined representation fed to policy and value heads

Design decisions:
- Start with MLP encoder (can upgrade to Transformer later)
- Action masking for illegal actions
- Separate value head for PPO advantage estimation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .state_encoder import (
    CARD_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    EncodedState,
    StateEncoder,
)

if TYPE_CHECKING:
    from src.game.actions import Action
    from src.game.state import GameState


# Default architecture hyperparameters
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_ACTIONS = 256  # Max action space size


class CardSetEncoder(nn.Module):
    """
    Encodes a set of cards into a fixed-size representation.

    Uses mean pooling over card embeddings with masking for padding.
    Can be upgraded to attention/transformer for better performance.
    """

    def __init__(
        self,
        card_dim: int = CARD_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()

        self.card_encoder = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        cards: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a set of cards.

        Args:
            cards: (batch, max_cards, card_dim) card features
            mask: (batch, max_cards) binary mask (1 = valid, 0 = padding)

        Returns:
            (batch, hidden_dim) aggregated representation
        """
        # Encode each card
        # cards: (batch, max_cards, card_dim) -> (batch, max_cards, hidden_dim)
        encoded = self.card_encoder(cards)

        # Apply mask for mean pooling
        # Expand mask to match encoded dimensions
        mask_expanded = mask.unsqueeze(-1)  # (batch, max_cards, 1)

        # Masked sum
        masked_encoded = encoded * mask_expanded
        summed = masked_encoded.sum(dim=1)  # (batch, hidden_dim)

        # Divide by count (avoid division by zero)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        pooled = summed / count

        return pooled


class PolicyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities and state values.

    Architecture:
    - CardSetEncoder for hand, battlefield, and opponent battlefield
    - Global feature encoder
    - Combined representation
    - Policy head: outputs logits for each action
    - Value head: outputs scalar state value

    Usage:
        network = PolicyNetwork(state_dim=encoder.get_state_dim())
        logits, value = network(encoded_state)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        card_dim: int = CARD_FEATURE_DIM,
    ):
        """
        Initialize the policy network.

        Args:
            state_dim: Dimension of flattened state (for compatibility check)
            hidden_dim: Hidden layer dimension
            num_actions: Maximum number of actions
            card_dim: Dimension of card feature vectors
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Encoders for card sets
        self.hand_encoder = CardSetEncoder(card_dim, hidden_dim)
        self.battlefield_encoder = CardSetEncoder(card_dim, hidden_dim)
        self.opponent_encoder = CardSetEncoder(card_dim, hidden_dim)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        # Combined dimension: 3 card encoders + global encoder
        combined_dim = hidden_dim * 3 + hidden_dim // 2

        # Trunk network (shared between policy and value)
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: EncodedState,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: EncodedState with all tensor components
            action_mask: Optional (batch, num_actions) mask for legal actions
                        (1 = legal, 0 = illegal)

        Returns:
            Tuple of (action_logits, state_value)
            - action_logits: (batch, num_actions) or (num_actions,)
            - state_value: (batch, 1) or (1,)
        """
        # Ensure we have batch dimension
        hand = state.hand
        if hand.dim() == 2:
            # Add batch dimension
            hand = hand.unsqueeze(0)
            battlefield = state.battlefield.unsqueeze(0)
            opponent_bf = state.opponent_battlefield.unsqueeze(0)
            global_feat = state.global_features.unsqueeze(0)
            hand_mask = state.hand_mask.unsqueeze(0)
            bf_mask = state.battlefield_mask.unsqueeze(0)
            opp_mask = state.opponent_battlefield_mask.unsqueeze(0)
            squeeze_output = True
        else:
            battlefield = state.battlefield
            opponent_bf = state.opponent_battlefield
            global_feat = state.global_features
            hand_mask = state.hand_mask
            bf_mask = state.battlefield_mask
            opp_mask = state.opponent_battlefield_mask
            squeeze_output = False

        # Encode each component
        hand_encoded = self.hand_encoder(hand, hand_mask)
        bf_encoded = self.battlefield_encoder(battlefield, bf_mask)
        opp_encoded = self.opponent_encoder(opponent_bf, opp_mask)
        global_encoded = self.global_encoder(global_feat)

        # Combine all features
        combined = torch.cat(
            [hand_encoded, bf_encoded, opp_encoded, global_encoded],
            dim=-1,
        )

        # Pass through trunk
        trunk_output = self.trunk(combined)

        # Policy head
        logits = self.policy_head(trunk_output)

        # Apply action mask if provided
        if action_mask is not None:
            # Set masked actions to large negative value
            mask = action_mask if action_mask.dim() == 2 else action_mask.unsqueeze(0)
            logits = logits.masked_fill(mask == 0, float("-inf"))

        # Value head
        value = self.value_head(trunk_output)

        if squeeze_output:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value

    def get_action_probs(
        self,
        state: EncodedState,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get action probabilities (softmax of logits).

        Args:
            state: Encoded game state
            action_mask: Optional mask for legal actions

        Returns:
            Action probabilities (batch, num_actions) or (num_actions,)
        """
        logits, _ = self.forward(state, action_mask)
        return F.softmax(logits, dim=-1)


class NeuralPolicy:
    """
    Wraps PolicyNetwork for use with the game Simulator.

    Implements the Policy interface expected by Simulator.run_episode()
    and Simulator.run_goldfish().

    Usage:
        encoder = StateEncoder()
        network = PolicyNetwork(encoder.get_state_dim())
        policy = NeuralPolicy(network, encoder)

        # Use with simulator
        sim = Simulator()
        result = sim.run_goldfish(deck, policy)
    """

    def __init__(
        self,
        network: PolicyNetwork,
        encoder: StateEncoder,
        device: torch.device | None = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ):
        """
        Initialize the neural policy.

        Args:
            network: The policy network
            encoder: State encoder for converting GameState to tensors
            device: Torch device (CPU/GPU)
            temperature: Sampling temperature (1.0 = normal, <1 = more greedy)
            deterministic: If True, always select highest probability action
        """
        self.network = network
        self.encoder = encoder
        self.device = device or torch.device("cpu")
        self.temperature = temperature
        self.deterministic = deterministic

        # Move network to device
        self.network.to(self.device)

    def select_action(
        self,
        state: GameState,
        legal_actions: list[Action],
    ) -> Action:
        """
        Select an action given the game state and legal actions.

        Args:
            state: Current game state
            legal_actions: List of legal actions

        Returns:
            Selected action
        """
        from src.game.actions import Action, ActionType

        if not legal_actions:
            return Action(action_type=ActionType.PASS)

        # Encode state
        encoded = self.encoder.encode_state(state)
        encoded = encoded.to(self.device)

        # Create action mask
        action_mask = self._create_action_mask(legal_actions)
        action_mask = action_mask.to(self.device)

        # Get logits from network
        with torch.no_grad():
            logits, _ = self.network(encoded, action_mask)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        if self.deterministic:
            # Select highest probability action
            action_idx = probs.argmax().item()
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        # Map back to legal action
        if action_idx < len(legal_actions):
            return legal_actions[action_idx]
        else:
            # Fallback if index out of range
            return legal_actions[0]

    def get_action_and_value(
        self,
        state: GameState,
        legal_actions: list[Action],
    ) -> tuple[Action, float, float, torch.Tensor]:
        """
        Get action, value, log probability, and entropy for training.

        Args:
            state: Current game state
            legal_actions: List of legal actions

        Returns:
            Tuple of (action, log_prob, value, entropy)
        """
        from src.game.actions import Action, ActionType

        if not legal_actions:
            action = Action(action_type=ActionType.PASS)
            return action, 0.0, 0.0, torch.tensor(0.0)

        # Encode state
        encoded = self.encoder.encode_state(state)
        encoded = encoded.to(self.device)

        # Create action mask
        action_mask = self._create_action_mask(legal_actions)
        action_mask = action_mask.to(self.device)

        # Get logits and value from network
        logits, value = self.network(encoded, action_mask)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if self.deterministic:
            action_idx = probs.argmax().item()
            action_tensor = torch.tensor(action_idx)
        else:
            action_tensor = dist.sample()
            action_idx = action_tensor.item()

        # Get log probability and entropy
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()

        # Map back to legal action
        if action_idx < len(legal_actions):
            action = legal_actions[action_idx]
        else:
            action = legal_actions[0]
            # Recalculate log_prob for the fallback action
            log_prob = dist.log_prob(torch.tensor(0))

        return action, log_prob.item(), value.squeeze().item(), entropy

    def evaluate_actions(
        self,
        encoded_states: dict[str, torch.Tensor],
        action_indices: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of actions for PPO update.

        Args:
            encoded_states: Batched encoded states
            action_indices: (batch,) indices of actions taken
            action_masks: (batch, num_actions) masks for legal actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        # Create EncodedState from batched tensors
        encoded = EncodedState(
            hand=encoded_states["hand"],
            battlefield=encoded_states["battlefield"],
            opponent_battlefield=encoded_states["opponent_battlefield"],
            global_features=encoded_states["global_features"],
            hand_mask=encoded_states["hand_mask"],
            battlefield_mask=encoded_states["battlefield_mask"],
            opponent_battlefield_mask=encoded_states["opponent_battlefield_mask"],
        )

        # Forward pass
        logits, values = self.network(encoded, action_masks)

        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Get log probs for the taken actions
        log_probs = dist.log_prob(action_indices)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def _create_action_mask(self, legal_actions: list[Action]) -> torch.Tensor:
        """
        Create an action mask tensor.

        Args:
            legal_actions: List of legal actions

        Returns:
            (num_actions,) tensor with 1s for legal action indices
        """
        mask = torch.zeros(self.network.num_actions, dtype=torch.float32)
        # Mark legal action indices as 1
        for i in range(min(len(legal_actions), self.network.num_actions)):
            mask[i] = 1.0
        return mask

    def set_eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()

    def set_train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()
