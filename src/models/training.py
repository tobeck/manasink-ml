"""
PPO training loop for MTG policy learning.

This module provides:
- Experience: Data class for single-step experience
- TrainingConfig: Hyperparameter configuration
- Trainer: Main training class with PPO algorithm

The training uses goldfish simulation (single-player) as the primary mode,
with reward shaping based on damage dealt and synergy bonuses.

PPO (Proximal Policy Optimization) is used for stable policy updates:
- Clipped surrogate objective prevents large policy changes
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function for baseline subtraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .state_encoder import StateEncoder, EncodedState, batch_encode_states
from .policy_network import PolicyNetwork, NeuralPolicy

if TYPE_CHECKING:
    from src.game.card import Card
    from src.game.state import GameState
    from src.game.synergy_policy import SynergyContext


@dataclass
class Experience:
    """Single step experience for training."""

    # State information (stored as encoded tensors)
    encoded_state: EncodedState
    action_mask: torch.Tensor  # Legal action mask

    # Action taken
    action_idx: int

    # Reward and values
    reward: float
    value: float
    log_prob: float

    # Episode status
    done: bool

    # For debugging/analysis
    turn_number: int = 0
    card_played: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Training loop
    num_episodes: int = 1000
    batch_size: int = 32
    epochs_per_update: int = 4  # PPO epochs per batch

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip range
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient

    # Reward shaping
    synergy_reward_weight: float = 0.1  # Weight for synergy bonuses
    damage_reward_weight: float = 0.1  # Weight for damage dealt
    win_reward: float = 1.0  # Bonus for winning

    # Network architecture
    hidden_dim: int = 256
    num_actions: int = 256

    # Simulation
    max_turns: int = 15
    target_damage: int = 40  # Goldfish target

    # Checkpointing
    checkpoint_every: int = 100  # Episodes between checkpoints
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every: int = 10  # Episodes between logging


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    episode: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    damage_dealt: int = 0
    synergy_bonus: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    win: bool = False


class ExperienceBuffer:
    """Buffer for storing experiences during episode collection."""

    def __init__(self):
        self.experiences: list[Experience] = []
        self._states: list[EncodedState] = []
        self._action_masks: list[torch.Tensor] = []
        self._action_indices: list[int] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._log_probs: list[float] = []
        self._dones: list[bool] = []

    def add(self, exp: Experience) -> None:
        """Add an experience to the buffer."""
        self.experiences.append(exp)
        self._states.append(exp.encoded_state)
        self._action_masks.append(exp.action_mask)
        self._action_indices.append(exp.action_idx)
        self._rewards.append(exp.reward)
        self._values.append(exp.value)
        self._log_probs.append(exp.log_prob)
        self._dones.append(exp.done)

    def clear(self) -> None:
        """Clear the buffer."""
        self.experiences.clear()
        self._states.clear()
        self._action_masks.clear()
        self._action_indices.clear()
        self._rewards.clear()
        self._values.clear()
        self._log_probs.clear()
        self._dones.clear()

    def __len__(self) -> int:
        return len(self.experiences)

    def get_batch_tensors(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Convert buffer to batched tensors for training.

        Returns dict with:
        - encoded states (batched)
        - action_masks
        - action_indices
        - rewards
        - values
        - log_probs
        - dones
        """
        # Stack encoded states
        states = {
            "hand": torch.stack([s.hand for s in self._states]).to(device),
            "battlefield": torch.stack([s.battlefield for s in self._states]).to(device),
            "opponent_battlefield": torch.stack([s.opponent_battlefield for s in self._states]).to(
                device
            ),
            "global_features": torch.stack([s.global_features for s in self._states]).to(device),
            "hand_mask": torch.stack([s.hand_mask for s in self._states]).to(device),
            "battlefield_mask": torch.stack([s.battlefield_mask for s in self._states]).to(device),
            "opponent_battlefield_mask": torch.stack(
                [s.opponent_battlefield_mask for s in self._states]
            ).to(device),
        }

        return {
            "states": states,
            "action_masks": torch.stack(self._action_masks).to(device),
            "action_indices": torch.tensor(self._action_indices, device=device),
            "rewards": torch.tensor(self._rewards, dtype=torch.float32, device=device),
            "values": torch.tensor(self._values, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(self._log_probs, dtype=torch.float32, device=device),
            "dones": torch.tensor(self._dones, dtype=torch.float32, device=device),
        }


class Trainer:
    """
    PPO trainer for MTG policy.

    Handles:
    - Episode collection using goldfish simulation
    - GAE advantage computation
    - PPO policy updates
    - Checkpointing and logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        synergy_context: Optional["SynergyContext"] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            synergy_context: Optional synergy data for reward shaping
            device: Torch device (defaults to CUDA if available)
        """
        self.config = config
        self.synergy_context = synergy_context
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create encoder and network
        self.encoder = StateEncoder()
        self.network = PolicyNetwork(
            state_dim=self.encoder.get_state_dim(),
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
        ).to(self.device)

        # Create policy wrapper
        self.policy = NeuralPolicy(
            network=self.network,
            encoder=self.encoder,
            device=self.device,
            deterministic=False,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # Experience buffer
        self.buffer = ExperienceBuffer()

        # Training state
        self.total_episodes = 0
        self.total_steps = 0
        self._metrics_history: list[TrainingMetrics] = []

    def train(
        self,
        deck: list["Card"],
        commander: Optional["Card"] = None,
        num_episodes: Optional[int] = None,
        verbose: bool = True,
    ) -> list[TrainingMetrics]:
        """
        Main training loop.

        Args:
            deck: The deck to train with
            commander: Optional commander card
            num_episodes: Override config.num_episodes
            verbose: Print progress

        Returns:
            List of training metrics for each episode
        """
        num_episodes = num_episodes or self.config.num_episodes
        metrics_list = []

        for episode in range(num_episodes):
            # Collect episode experiences
            metrics = self.collect_episode(
                deck,
                commander,
                seed=self.total_episodes,
            )
            metrics.episode = self.total_episodes
            metrics_list.append(metrics)
            self._metrics_history.append(metrics)

            # Update policy every batch_size episodes
            if len(self.buffer) >= self.config.batch_size:
                update_metrics = self.ppo_update()
                # Update last metrics with loss values
                metrics.policy_loss = update_metrics["policy_loss"]
                metrics.value_loss = update_metrics["value_loss"]
                metrics.entropy = update_metrics["entropy"]
                self.buffer.clear()

            self.total_episodes += 1

            # Logging
            if verbose and (episode + 1) % self.config.log_every == 0:
                avg_reward = sum(m.total_reward for m in metrics_list[-10:]) / min(
                    10, len(metrics_list)
                )
                avg_damage = sum(m.damage_dealt for m in metrics_list[-10:]) / min(
                    10, len(metrics_list)
                )
                print(
                    f"Episode {self.total_episodes}: "
                    f"Avg Reward={avg_reward:.2f}, "
                    f"Avg Damage={avg_damage:.1f}, "
                    f"Policy Loss={metrics.policy_loss:.4f}"
                )

            # Checkpointing
            if (episode + 1) % self.config.checkpoint_every == 0:
                checkpoint_path = (
                    Path(self.config.checkpoint_dir) / f"checkpoint_{self.total_episodes}.pt"
                )
                self.save_checkpoint(str(checkpoint_path))

        return metrics_list

    def collect_episode(
        self,
        deck: list["Card"],
        commander: Optional["Card"] = None,
        seed: Optional[int] = None,
    ) -> TrainingMetrics:
        """
        Run one goldfish episode and collect experiences.

        Args:
            deck: The deck to use
            commander: Optional commander card
            seed: Random seed for reproducibility

        Returns:
            Episode metrics
        """
        from src.game.simulator import Simulator
        from src.game.state import create_game
        from src.game.actions import Action, ActionType, get_legal_actions, execute_action

        # Create game state
        state = create_game("Trainer", "Dummy", seed=seed)
        state.setup_game(deck, [], commander, None)

        # Give dummy player high life to measure damage
        state.players[1].life = 1000
        initial_dummy_life = state.players[1].life

        metrics = TrainingMetrics()
        total_synergy = 0.0

        self.network.eval()

        # Run episode
        while not state.is_game_over and state.turn_number <= self.config.max_turns:
            # Only active player (0) takes meaningful actions
            if state.active_player_index != 0:
                execute_action(state, Action(action_type=ActionType.PASS))
                continue

            legal_actions = get_legal_actions(state)
            if not legal_actions:
                execute_action(state, Action(action_type=ActionType.PASS))
                continue

            # Get previous state for reward computation
            prev_dummy_life = state.players[1].life

            # Encode state
            encoded_state = self.encoder.encode_state(state)
            action_mask = self.policy._create_action_mask(legal_actions)

            # Get action from policy with value and log_prob
            action, log_prob, value, _ = self.policy.get_action_and_value(state, legal_actions)

            # Find action index
            try:
                action_idx = legal_actions.index(action)
            except ValueError:
                action_idx = 0

            # Execute action
            execute_action(state, action)

            # Compute reward
            reward = self._compute_reward(
                state,
                action,
                prev_dummy_life,
            )

            # Track synergy bonus if playing a card
            if action.card and self.synergy_context:
                synergy = self.synergy_context.get_synergy(action.card.name)
                synergy_reward = synergy * self.config.synergy_reward_weight
                reward += synergy_reward
                total_synergy += synergy

            # Check if episode is done
            done = state.is_game_over or state.turn_number > self.config.max_turns

            # Check for goldfish "win" (dealt enough damage)
            damage_dealt = initial_dummy_life - state.players[1].life
            if damage_dealt >= self.config.target_damage:
                reward += self.config.win_reward
                done = True
                metrics.win = True

            # Create experience
            exp = Experience(
                encoded_state=encoded_state,
                action_mask=action_mask,
                action_idx=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                turn_number=state.turn_number,
                card_played=action.card.name if action.card else None,
            )
            self.buffer.add(exp)

            metrics.total_reward += reward
            metrics.episode_length += 1

            if done:
                break

        # Final metrics
        metrics.damage_dealt = initial_dummy_life - state.players[1].life
        metrics.synergy_bonus = total_synergy

        self.total_steps += metrics.episode_length

        return metrics

    def _compute_reward(
        self,
        state: "GameState",
        action: "Action",
        prev_opponent_life: int,
    ) -> float:
        """
        Compute immediate reward for an action.

        Reward signals:
        - Damage dealt to opponent
        - Playing cards (small bonus)
        """
        from src.game.actions import ActionType

        reward = 0.0

        # Damage dealt
        damage_delta = prev_opponent_life - state.players[1].life
        reward += damage_delta * self.config.damage_reward_weight

        # Small bonus for playing cards (encourages action)
        if action.action_type == ActionType.CAST_SPELL:
            reward += 0.01
        elif action.action_type == ActionType.PLAY_LAND:
            reward += 0.005

        return reward

    def ppo_update(self) -> dict[str, float]:
        """
        Perform PPO update on collected experiences.

        Returns:
            Dict with loss values
        """
        self.network.train()

        # Get batch tensors
        batch = self.buffer.get_batch_tensors(self.device)

        # Compute returns and advantages
        returns, advantages = self._compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        old_log_probs = batch["log_probs"]

        for _ in range(self.config.epochs_per_update):
            # Get new log probs and values
            new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                batch["states"],
                batch["action_indices"],
                batch["action_masks"],
            )

            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: (batch,) rewards
            values: (batch,) value estimates
            dones: (batch,) done flags

        Returns:
            Tuple of (returns, advantages)
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros(batch_size, device=self.device)
        returns = torch.zeros(batch_size, device=self.device)

        last_gae = 0.0
        last_value = 0.0  # Bootstrap value (0 for terminal states)

        for t in reversed(range(batch_size)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = last_value

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = (
                delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            )

            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

            last_value = values[t].item()

        return returns, advantages

    def get_policy(self) -> NeuralPolicy:
        """
        Get the trained policy for inference.

        Returns:
            NeuralPolicy configured for inference
        """
        return NeuralPolicy(
            network=self.network,
            encoder=self.encoder,
            device=self.device,
            deterministic=True,  # Greedy for inference
        )

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: File path for checkpoint
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: File path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        # Update policy reference
        self.policy = NeuralPolicy(
            network=self.network,
            encoder=self.encoder,
            device=self.device,
        )

    def get_metrics_history(self) -> list[TrainingMetrics]:
        """Get all training metrics."""
        return self._metrics_history.copy()


def create_trainer(
    config: Optional[TrainingConfig] = None,
    synergy_context: Optional["SynergyContext"] = None,
) -> Trainer:
    """
    Factory function to create a Trainer with default settings.

    Args:
        config: Optional training config (uses defaults if not provided)
        synergy_context: Optional synergy data for reward shaping

    Returns:
        Configured Trainer instance
    """
    config = config or TrainingConfig()
    return Trainer(config=config, synergy_context=synergy_context)
