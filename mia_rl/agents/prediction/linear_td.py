from __future__ import annotations

from typing import Callable

import numpy as np

from mia_rl.core.base import Transition
from mia_rl.envs.windy_gridworld import WindyGridworldState


class LinearTD0:
    """Online TD(0) with linear function approximation.

    Model:      v_hat(s) = w · phi(s)
    TD error:   delta = r + gamma * v_hat(s') - v_hat(s)
    Update:     w += alpha * delta * phi(s)

    This is semi-gradient descent: we treat v_hat(s') as a fixed target
    (do not differentiate through it), so only phi(s) appears in the update.
    """

    def __init__(
        self,
        phi: Callable[[WindyGridworldState], np.ndarray],
        n_features: int,
        alpha: float = 0.01,
        gamma: float = 1.0,
    ):
        self.phi = phi
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma
        self.w = np.zeros(n_features)

    def value_of(self, state: WindyGridworldState) -> float:
        return float(self.w @ self.phi(state))

    def update(self, transition: Transition) -> float:
        """One online TD(0) update. Returns the TD error delta."""
        phi_s = self.phi(transition.state)
        v_s = float(self.w @ phi_s)

        if transition.done or transition.next_state is None:
            target = transition.reward
        else:
            # Bootstrap: treat v_hat(s') as fixed — semi-gradient
            target = transition.reward + self.gamma * float(self.w @ self.phi(transition.next_state))

        delta = target - v_s
        self.w += self.alpha * delta * phi_s
        return delta
