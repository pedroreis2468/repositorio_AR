from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ReplayBuffer:
    """A simple replay buffer for experience replay."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Handle None next_states (terminal states) by filling with zeros
        non_terminal_mask = np.array([ns is not None for ns in next_states])
        next_states_filled = np.array([ns if ns is not None else np.zeros_like(states[0]) for ns in next_states])

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            next_states_filled,
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Multi-Layer Perceptron representing action Q-values."""

    def __init__(self, input_dim: int = 27, output_dim: int = 9) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """Deep Q-Network (DQN) agent for TicTacToe.

    Learns from board state features (27-dim encoding) to select cell moves.
    Includes target network and experience replay.
    """

    def __init__(
        self,
        n_features: int = 27,
        n_actions: int = 9,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_every: int = 100,
        device: str | torch.device | None = "auto",
        seed: int | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.device = _resolve_device(device)
        
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.reset()

    def reset(self) -> None:
        """Initialise/reinitialise policy network and target network."""
        self.policy_net = QNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.update_count = 0
        self._episode_buffer: list[tuple[np.ndarray, int, float]] = []

    def select_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Select action using epsilon-greedy strategy over available actions only."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(available))
        
        return self.greedy_action(phi, available)

    def greedy_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Select the greedy action (argmax Q) among legal moves only."""
        phi_t = torch.tensor(phi, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(phi_t).squeeze(0)
        
        # Filter Q-values for available moves
        q_available = q_values[available]
        best_idx = torch.argmax(q_available).item()
        return available[int(best_idx)]

    def store_transition(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        next_phi: np.ndarray | None,
        done: bool,
    ) -> None:
        """Add transition to replay buffer and update network if buffer is ready."""
        self.replay_buffer.push(phi, action, reward, next_phi, done)

    def update_model(self) -> float | None:
        """Perform one step of gradient descent. Returns loss value or None."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Predict current Q-values Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Predict target values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # Calculate loss (Mean Squared Error)
        loss = F.mse_loss(current_q, target_q)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network weights periodically
        self.update_count += 1
        if self.update_count % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
