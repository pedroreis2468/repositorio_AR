from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class MonteCarloControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self._episode_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        best_value = max(self.action_value_of(state, action) for action in self.actions)
        best_actions = [action for action in self.actions if self.action_value_of(state, action) == best_value]
        return self.rng.choice(best_actions)

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        # We store all transitions in the current episode and only update the action-value function at the end of the episode, when we have the full returns available.
        self._episode_transitions.append(transition)

        if not transition.done:
            return  #we only update at the end of the episode, so if the episode is not done, we do nothing

        self._update_from_episode() #when the episode is done, we update from the episode and clear the collected transitions for the next episode

    def end_episode(self) -> None: 
        # When episodes are truncated by max_steps, we still close and learn from collected returns.
        if self._episode_transitions: #if there are any transitions collected, update from the episode
            self._update_from_episode() #update from the episode and clear the collected transitions for the next episode

    def _update_from_episode(self) -> None:
        # updated_from_episode is used to update the action-value function at the end of an episode, using the collected transitions in self._episode_transitions.
        if not self._episode_transitions:  #if there are no transitions, do nothing
            return

        returns = [0.0] * len(self._episode_transitions)
        G = 0.0
        for idx in range(len(self._episode_transitions) - 1, -1, -1):
            step = self._episode_transitions[idx]
            G = step.reward + self.gamma * G
            returns[idx] = G

        first_visit_index: dict[tuple[StateT, ActionT], int] = {}
        for idx, step in enumerate(self._episode_transitions):
            first_visit_index.setdefault((step.state, step.action), idx)

        for idx, step in enumerate(self._episode_transitions):
            state_action = (step.state, step.action)
            if first_visit_index[state_action] != idx:
                continue

            self.N[state_action] += 1
            self.Q[state_action] += (returns[idx] - self.Q[state_action]) / self.N[state_action]

        self._episode_transitions.clear()

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        # Used for plotting the greedy policy at the end of training. We break ties by choosing randomly among the best actions.
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
