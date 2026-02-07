"""
State encoding for neural network input.

This module converts GameState objects into fixed-size tensor representations
suitable for neural network processing. The encoding captures:

- Hand: Cards in the active player's hand
- Battlefield: Permanents controlled by the active player
- Opponent battlefield: Permanents controlled by the opponent
- Global features: Turn number, life totals, mana, phase, etc.

Design decisions:
- Fixed-size tensors with padding for variable-length card sets
- Uses CardFeatures (29 dims) for card encoding when available
- Falls back to basic card properties when features unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.data.features import CardFeatures
    from src.game.actions import Action
    from src.game.card import Card
    from src.game.state import GameState


# Card encoding dimension (matches CardFeatures.vector_size())
CARD_FEATURE_DIM = 29

# Number of game phases
NUM_PHASES = 7

# Global feature dimensions breakdown:
# - Turn number (1, normalized)
# - Our life (1, normalized by 40)
# - Opponent life (1, normalized by 40)
# - Available mana per color (6: W, U, B, R, G, C)
# - Phase one-hot (7)
# - Land played this turn (1)
GLOBAL_FEATURE_DIM = 1 + 1 + 1 + 6 + 7 + 1  # = 17


@dataclass
class EncodedState:
    """Encoded game state as tensors."""

    hand: torch.Tensor  # (max_hand, card_dim)
    battlefield: torch.Tensor  # (max_battlefield, card_dim)
    opponent_battlefield: torch.Tensor  # (max_battlefield, card_dim)
    global_features: torch.Tensor  # (global_dim,)

    # Masks for variable-length inputs (1 = valid, 0 = padding)
    hand_mask: torch.Tensor  # (max_hand,)
    battlefield_mask: torch.Tensor  # (max_battlefield,)
    opponent_battlefield_mask: torch.Tensor  # (max_battlefield,)

    def to(self, device: torch.device) -> EncodedState:
        """Move all tensors to the specified device."""
        return EncodedState(
            hand=self.hand.to(device),
            battlefield=self.battlefield.to(device),
            opponent_battlefield=self.opponent_battlefield.to(device),
            global_features=self.global_features.to(device),
            hand_mask=self.hand_mask.to(device),
            battlefield_mask=self.battlefield_mask.to(device),
            opponent_battlefield_mask=self.opponent_battlefield_mask.to(device),
        )


class StateEncoder:
    """
    Encodes GameState into tensor representation for neural networks.

    The encoder uses a fixed-size representation:
    - Cards are encoded as feature vectors
    - Variable-length zones (hand, battlefield) are padded
    - Global game state is encoded as a flat vector
    """

    def __init__(
        self,
        card_dim: int = CARD_FEATURE_DIM,
        max_hand: int = 10,
        max_battlefield: int = 20,
    ):
        """
        Initialize the state encoder.

        Args:
            card_dim: Dimension of card feature vectors
            max_hand: Maximum cards to encode in hand
            max_battlefield: Maximum permanents to encode on battlefield
        """
        self.card_dim = card_dim
        self.max_hand = max_hand
        self.max_battlefield = max_battlefield

        # Cache for card embeddings (card_name -> tensor)
        self._card_cache: dict[str, torch.Tensor] = {}

    def get_state_dim(self) -> int:
        """
        Return the total dimension of the flattened state representation.

        This is useful for sizing the input layer of a neural network.
        """
        hand_dim = self.max_hand * self.card_dim
        battlefield_dim = self.max_battlefield * self.card_dim
        opponent_dim = self.max_battlefield * self.card_dim
        return hand_dim + battlefield_dim + opponent_dim + GLOBAL_FEATURE_DIM

    def encode_state(self, state: GameState) -> EncodedState:
        """
        Encode a GameState into tensors.

        Args:
            state: The game state to encode

        Returns:
            EncodedState with all encoded tensors
        """
        player = state.active_player
        opponent_idx = 1 - state.active_player_index
        opponent = state.players[opponent_idx]

        # Encode hand
        hand, hand_mask = self._encode_card_zone(
            [card for card in player.hand],
            self.max_hand,
        )

        # Encode our battlefield (just the cards from permanents)
        our_cards = [perm.card for perm in player.battlefield]
        battlefield, battlefield_mask = self._encode_card_zone(
            our_cards,
            self.max_battlefield,
        )

        # Encode opponent battlefield
        opp_cards = [perm.card for perm in opponent.battlefield]
        opp_battlefield, opp_mask = self._encode_card_zone(
            opp_cards,
            self.max_battlefield,
        )

        # Encode global features
        global_features = self._encode_global_features(state)

        return EncodedState(
            hand=hand,
            battlefield=battlefield,
            opponent_battlefield=opp_battlefield,
            global_features=global_features,
            hand_mask=hand_mask,
            battlefield_mask=battlefield_mask,
            opponent_battlefield_mask=opp_mask,
        )

    def encode_card(self, card: Card) -> torch.Tensor:
        """
        Encode a single card as a feature vector.

        Uses cached embeddings when available, otherwise computes from
        card properties.

        Args:
            card: The card to encode

        Returns:
            Tensor of shape (card_dim,)
        """
        # Check cache first
        if card.name in self._card_cache:
            return self._card_cache[card.name]

        # Try to get features from database
        features = self._get_card_features(card)

        if features is not None:
            tensor = torch.tensor(features.to_vector(), dtype=torch.float32)
        else:
            # Fall back to basic encoding from card properties
            tensor = self._encode_card_basic(card)

        # Cache and return
        self._card_cache[card.name] = tensor
        return tensor

    def encode_action(
        self,
        action: Action,
        legal_actions: list[Action],
    ) -> int:
        """
        Encode an action as an index into the action space.

        Args:
            action: The action to encode
            legal_actions: List of all legal actions (for indexing)

        Returns:
            Integer index of the action
        """
        try:
            return legal_actions.index(action)
        except ValueError:
            return 0  # Default to first action if not found

    def _encode_card_zone(
        self,
        cards: list[Card],
        max_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a zone of cards with padding.

        Args:
            cards: List of cards in the zone
            max_size: Maximum size (will pad or truncate)

        Returns:
            Tuple of (encoded_cards, mask) tensors
        """
        # Truncate if necessary
        cards = cards[:max_size]

        # Encode each card
        encoded = []
        for card in cards:
            encoded.append(self.encode_card(card))

        # Pad with zeros
        while len(encoded) < max_size:
            encoded.append(torch.zeros(self.card_dim, dtype=torch.float32))

        # Stack into tensor
        zone_tensor = torch.stack(encoded)

        # Create mask (1 for real cards, 0 for padding)
        mask = torch.zeros(max_size, dtype=torch.float32)
        mask[: len(cards)] = 1.0

        return zone_tensor, mask

    def _encode_global_features(self, state: GameState) -> torch.Tensor:
        """
        Encode global game state features.

        Features (17 total):
        - Turn number (normalized by 20)
        - Our life (normalized by 40)
        - Opponent life (normalized by 40)
        - Available mana: W, U, B, R, G, colorless (6)
        - Phase one-hot (7)
        - Land played this turn (1)
        """
        from src.game.state import Phase

        player = state.active_player
        opponent_idx = 1 - state.active_player_index
        opponent = state.players[opponent_idx]

        features = []

        # Turn number (normalized)
        features.append(min(state.turn_number / 20.0, 1.0))

        # Life totals (normalized by commander starting life)
        features.append(player.life / 40.0)
        features.append(opponent.life / 40.0)

        # Available mana from untapped lands
        mana = player.get_available_mana()
        features.append(mana.white / 10.0)
        features.append(mana.blue / 10.0)
        features.append(mana.black / 10.0)
        features.append(mana.red / 10.0)
        features.append(mana.green / 10.0)
        features.append(mana.colorless / 10.0)

        # Phase one-hot
        phase_idx = list(Phase).index(state.phase)
        phase_one_hot = [0.0] * NUM_PHASES
        phase_one_hot[phase_idx] = 1.0
        features.extend(phase_one_hot)

        # Land played this turn
        features.append(1.0 if player.land_played_this_turn else 0.0)

        return torch.tensor(features, dtype=torch.float32)

    def _encode_card_basic(self, card: Card) -> torch.Tensor:
        """
        Basic card encoding from card properties when database features unavailable.

        Encodes to match CardFeatures.vector_size() = 29 dimensions.
        """
        from src.game.card import CardType, Color

        features = []

        # Mana costs (7 dims: W, U, B, R, G, C, CMC)
        cost = card.mana_cost
        features.extend(
            [
                float(cost.white),
                float(cost.blue),
                float(cost.black),
                float(cost.red),
                float(cost.green),
                float(cost.colorless),
                float(card.cmc),
            ]
        )

        # Card types (8 dims)
        features.append(1.0 if card.is_creature else 0.0)
        features.append(1.0 if CardType.INSTANT in card.card_types else 0.0)
        features.append(1.0 if CardType.SORCERY in card.card_types else 0.0)
        features.append(1.0 if CardType.ARTIFACT in card.card_types else 0.0)
        features.append(1.0 if CardType.ENCHANTMENT in card.card_types else 0.0)
        features.append(1.0 if CardType.PLANESWALKER in card.card_types else 0.0)
        features.append(1.0 if card.is_land else 0.0)
        # Legendary flag (check if commander or type line contains "legendary")
        features.append(1.0 if card.is_commander else 0.0)

        # Power/toughness (2 dims)
        features.append(float(card.power or 0))
        features.append(float(card.toughness or 0))

        # Color identity (5 dims)
        color_map = [Color.WHITE, Color.BLUE, Color.BLACK, Color.RED, Color.GREEN]
        for color in color_map:
            features.append(1.0 if color in card.color_identity else 0.0)

        # Role scores (7 dims) - zeros without database
        features.extend([0.0] * 7)

        assert len(features) == CARD_FEATURE_DIM
        return torch.tensor(features, dtype=torch.float32)

    def _get_card_features(self, card: Card) -> CardFeatures | None:
        """
        Try to get CardFeatures from the database.

        Returns None if database unavailable or card not found.
        """
        try:
            from src.data.features import get_feature_vector

            return get_feature_vector(card.name)
        except Exception:
            return None


def batch_encode_states(
    encoder: StateEncoder,
    states: list[GameState],
) -> dict[str, torch.Tensor]:
    """
    Encode multiple states into batched tensors.

    Args:
        encoder: The state encoder to use
        states: List of game states

    Returns:
        Dict with batched tensors for each component
    """
    encoded_states = [encoder.encode_state(state) for state in states]

    return {
        "hand": torch.stack([s.hand for s in encoded_states]),
        "battlefield": torch.stack([s.battlefield for s in encoded_states]),
        "opponent_battlefield": torch.stack([s.opponent_battlefield for s in encoded_states]),
        "global_features": torch.stack([s.global_features for s in encoded_states]),
        "hand_mask": torch.stack([s.hand_mask for s in encoded_states]),
        "battlefield_mask": torch.stack([s.battlefield_mask for s in encoded_states]),
        "opponent_battlefield_mask": torch.stack(
            [s.opponent_battlefield_mask for s in encoded_states]
        ),
    }
