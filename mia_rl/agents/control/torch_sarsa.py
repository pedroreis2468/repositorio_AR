from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use a single thread to avoid OpenMP conflicts when running multiple agents
# sequentially in the same process, and to keep per-step overhead low.
torch.set_num_threads(1)

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Accepts ``"auto"`` (picks CUDA if available, else CPU), ``"cuda"``,
    ``"cpu"``, or any valid :class:`torch.device` / device string.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TorchSarsaControl(ControlAgent[StateT, ActionT]):
    """Semi-gradient SARSA with linear function approximation (PyTorch).

    The model is nn.Linear(n_features, 1, bias=False), so q_hat(s,a) = w · phi(s,a),
    which is mathematically identical to LinearSarsaControl.  The point of this
    class is to demonstrate two PyTorch update styles side-by-side:

    Manual (use_optimizer=False):
        Calls loss.backward() to get grad_w automatically via autograd,
        then applies the update by hand:  w -= alpha * grad_w
        Since loss = 0.5*(pred - target)^2, grad_w = (pred-target)*phi,
        so the step becomes  w += alpha * delta * phi  — identical to NumPy.

    Optimizer (use_optimizer=True):
        loss.backward(); optimizer.step()  — same math, via torch.optim.SGD.

    The bootstrap target is always detached (semi-gradient: no gradient flows
    through the next-state Q-value estimate).

    Args:
        device: ``"auto"`` (default) selects CUDA if available, else CPU.
                Pass ``"cpu"`` or ``"cuda"`` to force a specific device.
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        phi: Callable[[StateT, ActionT], np.ndarray],
        n_features: int,
        alpha: float = 0.01,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        use_optimizer: bool = False,
        seed: int | None = None,
        device: str | torch.device | None = "auto",
    ):
        self.actions = actions
        self.phi = phi
        self.n_features = n_features
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_optimizer = use_optimizer
        self.rng = random.Random(seed)
        self.device = _resolve_device(device)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        torch.manual_seed(0)
        # Linear model: q_hat(s,a) = w · phi(s,a), no bias, no hidden layers.
        # Mathematically identical to LinearSarsaControl — the value here is
        # demonstrating how PyTorch autograd/optimizers reproduce the same update.
        self.model = nn.Linear(self.n_features, 1, bias=False).to(self.device)
        nn.init.zeros_(self.model.weight)  # start from zero, same as NumPy version
        if self.use_optimizer:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._td_errors: list[float] = []
        self._tensor_cache: dict = {}
        self._batch_cache: dict = {}  # state -> stacked tensor of all (state, action) features

    def _to_tensor(self, state: StateT, action: ActionT) -> torch.Tensor:
        key = (state, action)
        if key not in self._tensor_cache:
            t = torch.tensor(self.phi(state, action), dtype=torch.float32, device=self.device)
            self._tensor_cache[key] = t
        return self._tensor_cache[key]

    def _all_q_values(self, state: StateT) -> list[float]:
        """Compute Q(state, a) for all actions in ONE batched forward pass."""
        if state not in self._batch_cache:
            self._batch_cache[state] = torch.stack(
                [self._to_tensor(state, a) for a in self.actions]
            )  # already on self.device
        with torch.no_grad():
            return self.model(self._batch_cache[state]).squeeze(-1).tolist()

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        with torch.no_grad():
            return self.model(self._to_tensor(state, action)).item()

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            # Single batched forward pass for all actions
            q_values = self._all_q_values(state)
            best_value = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
            action = self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        bootstrap = 0.0
        if not transition.done and transition.next_state is not None:
            next_action = self._selected_actions[transition.next_state]
            bootstrap = self.action_value_of(transition.next_state, next_action)

        target = transition.reward + self.gamma * bootstrap

        if self.use_optimizer:
            phi = self._to_tensor(transition.state, transition.action)
            self.optimizer.zero_grad()
            pred = self.model(phi)
            target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)
            loss = 0.5 * F.mse_loss(pred, target_tensor)
            loss.backward()
            self.optimizer.step()
            delta = abs(target - pred.item())
            self._td_errors.append(delta)
        else:
            phi = self._to_tensor(transition.state, transition.action)
            with torch.no_grad():
                pred_val = self.model(phi).item()
            delta_val = target - pred_val
            # Semi-gradient update: equivalent to  w += alpha * delta * phi
            # Written via autograd to demonstrate what the optimizer does internally:
            #   loss = 0.5 * (pred - target)^2  =>  grad_w = (pred - target) * phi
            #   w -= alpha * grad_w  ==>  w += alpha * delta * phi
            phi_t = self._to_tensor(transition.state, transition.action)
            pred = self.model(phi_t)
            target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)
            loss = 0.5 * F.mse_loss(pred, target_tensor)
            loss.backward()
            with torch.no_grad():
                self.model.weight.data -= self.alpha * self.model.weight.grad
            self.model.zero_grad()
            self._td_errors.append(abs(delta_val))

    def greedy_action(self, state: StateT) -> ActionT:
        q_values = self._all_q_values(state)
        best_value = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
        return self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)

    def flush_td_errors(self) -> list[float]:
        """Return and clear the accumulated per-step TD errors since last flush."""
        errors = list(self._td_errors)
        self._td_errors.clear()
        return errors
