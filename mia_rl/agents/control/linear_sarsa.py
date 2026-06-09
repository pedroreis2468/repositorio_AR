from __future__ import annotations

import random
from typing import Callable

import numpy as np

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class LinearSarsaControl(ControlAgent[StateT, ActionT]):
    """Semi-gradient SARSA with linear function approximation (NumPy).

    Model:      q_hat(s, a) = w · phi(s, a)
    TD error:   delta = r + gamma * q_hat(s', a') - q_hat(s, a)
    Update:     w += alpha * delta * phi(s, a)

    phi(s, a) uses an action-specific block encoding so that each action
    gets its own independent weight slice (the only non-zero block is the
    one for the current action).
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        phi: Callable[[StateT, ActionT], np.ndarray],
        n_features: int,
        alpha: float = 0.01,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.phi = phi
        self.n_features = n_features
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.w = np.zeros(self.n_features)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._td_errors: list[float] = []

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            q_values = [self.action_value_of(state, a) for a in self.actions]
            best_value = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
            action = self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Apply one semi-gradient SARSA update for the given transition.

        TODO:
        1. Compute the feature vector: phi = self.phi(transition.state, transition.action).
        2. If the transition is not terminal and next_state is not None, look up the cached
           next action from self._selected_actions[transition.next_state] and compute the
           bootstrap value self.action_value_of(next_state, next_action). Otherwise bootstrap = 0.0.
           Hint: terminal transitions have transition.done == True.
        3. Compute the TD error: delta = transition.reward + self.gamma * bootstrap - float(self.w @ phi).
        4. Apply the semi-gradient weight update: self.w += self.alpha * delta * phi.
        5. Record abs(delta) in self._td_errors.
        """
        # 1. Feature vector for (state, action)
        phi = self.phi(transition.state, transition.action)

        # 2. Bootstrap from next (state, action) if non-terminal
        bootstrap = 0.0
        if not transition.done and transition.next_state is not None:
            next_action = self._selected_actions[transition.next_state]
            bootstrap = self.action_value_of(transition.next_state, next_action)

        # 3. TD error: δ = r + γ·Q(s',a') - Q(s,a)
        delta = transition.reward + self.gamma * bootstrap - float(self.w @ phi)

        # 4. Semi-gradient weight update: w += α·δ·φ(s,a)
        self.w += self.alpha * delta * phi

        # 5. Record |δ| for diagnostics
        self._td_errors.append(abs(delta))

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.w @ self.phi(state, action))

    def greedy_action(self, state: StateT) -> ActionT:
        q_values = [self.action_value_of(state, a) for a in self.actions]
        best_value = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
        return self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)

    def flush_td_errors(self) -> list[float]:
        """Return and clear the accumulated per-step TD errors since last flush."""
        errors = list(self._td_errors)
        self._td_errors.clear()
        return errors
