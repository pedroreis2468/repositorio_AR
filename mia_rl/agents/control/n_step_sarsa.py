from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the n-step bootstrap."""
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            action = self.greedy_action(state)
            
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Store the transition and update the oldest state-action when possible."""
        self._pending_transitions.append(transition)

        if transition.done:
            # Episode ended: flush the remaining buffer
            while self._pending_transitions:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)
        elif len(self._pending_transitions) == self.n_steps:
            # Buffer is full: update the oldest transition and slide the window
            self._update_oldest_transition()
            self._pending_transitions.pop(0)

    def _update_oldest_transition(self) -> None:
        """Compute the n-step Sarsa target for the oldest transition in the buffer."""
        # 1. & 2. Sum the discounted rewards inside the current window
        g_return = 0.0
        for i, t in enumerate(self._pending_transitions):
            g_return += (self.gamma ** i) * t.reward

        # 3. Bootstrap if the window is exactly n_steps long and the last step isn't terminal
        last_transition = self._pending_transitions[-1]
        if not last_transition.done and len(self._pending_transitions) == self.n_steps:
            next_state = last_transition.next_state
            next_action = self._selected_actions[next_state]
            g_return += (self.gamma ** self.n_steps) * self.action_value_of(next_state, next_action)

        # 4. Apply the incremental update to the oldest state-action pair
        oldest_transition = self._pending_transitions[0]
        state = oldest_transition.state
        action = oldest_transition.action
        
        current_q = self.action_value_of(state, action)
        self.Q[(state, action)] = current_q + self.alpha * (g_return - current_q)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
