from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class DoubleQLearningControl(ControlAgent[StateT, ActionT]):
    """Tabular Double Q-learning control agent.

    Maintains two independent Q-tables (QA and QB) to avoid maximization bias.
    Updates alternate randomly with a 50% probability.
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.QA = defaultdict(float)
        self.QB = defaultdict(float)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        """Returns the average value of QA and QB for action selection."""
        return (self.QA[(state, action)] + self.QB[(state, action)]) / 2.0

    def select_action(self, state: StateT) -> ActionT:
        """Epsilon-greedy action selection using the average of QA and QB."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        
        # Greedy selection over QA + QB average
        best_value = max(self.action_value_of(state, a) for a in self.actions)
        best_actions = [a for a in self.actions if self.action_value_of(state, a) == best_value]
        return self.rng.choice(best_actions)

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Applies the Double Q-learning update step."""
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.done

        # Randomly choose which Q-table to update (50% probability)
        if self.rng.random() < 0.5:
            # Update QA using QB for target Q-value
            bootstrap = 0.0
            if not done and next_state is not None:
                # Find greedy action in QA
                best_next_a = max(self.actions, key=lambda a: self.QA[(next_state, a)])
                bootstrap = self.QB[(next_state, best_next_a)]
            
            target = reward + self.gamma * bootstrap
            self.QA[(state, action)] += self.alpha * (target - self.QA[(state, action)])
        else:
            # Update QB using QA for target Q-value
            bootstrap = 0.0
            if not done and next_state is not None:
                # Find greedy action in QB
                best_next_b = max(self.actions, key=lambda a: self.QB[(next_state, a)])
                bootstrap = self.QA[(next_state, best_next_b)]
            
            target = reward + self.gamma * bootstrap
            self.QB[(state, action)] += self.alpha * (target - self.QB[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        """Selects the action with the highest average value."""
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
