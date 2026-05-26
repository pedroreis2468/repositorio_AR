#!/usr/bin/env python3
"""K-Armed Bandit experiments — Practical 1.

Reproduces the main Sutton & Barto comparison plots:
  1. ε-greedy (ε = 0, 0.01, 0.1) — stationary 10-armed bandit
  2. Optimistic initialisation vs UCB (c=2) — stationary 10-armed bandit
  3. Gradient bandit with/without baseline (α = 0.1, 0.4) — stationary 10-armed bandit

Usage::

    python -m mia_rl.scripts.run_kbandits

Generated figures are saved under ``mia_rl/outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Environment: multi-armed bandit
# ============================================================

class KArmedBandit:
    """Stationary (or non-stationary) k-armed bandit environment."""

    def __init__(self, k: int = 10, stationary: bool = True, walk_std: float = 0.01):
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.reset()

    def reset(self) -> None:
        self.q_true = np.random.randn(self.k)          # true action values
        self.optimal_action = int(np.argmax(self.q_true))

    def step(self, action: int) -> float:
        reward = np.random.randn() + self.q_true[action]
        if not self.stationary:
            self.q_true += np.random.normal(0, self.walk_std, self.k)
            self.optimal_action = int(np.argmax(self.q_true))
        return reward


# ============================================================
# Base agent
# ============================================================

class BanditAgent:
    def __init__(self, k: int = 10):
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError


# ============================================================
# ε-greedy agent
# ============================================================

class EpsilonGreedy(BanditAgent):
    """ε-greedy with optional constant step size and optimistic initialisation."""

    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
        alpha: float | None = None,
        optimistic: float = 0.0,
    ):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        max_q = np.max(self.Q)
        ties = np.flatnonzero(self.Q == max_q)
        return int(np.random.choice(ties))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.N[action] += 1
        step = self.alpha if self.alpha is not None else 1.0 / self.N[action]
        self.Q[action] += step * (reward - self.Q[action])


# ============================================================
# UCB agent
# ============================================================

class UCB(BanditAgent):
    """Upper-Confidence-Bound action selection (UCB1)."""

    def __init__(self, k: int = 10, c: float = 2.0):
        super().__init__(k)
        self.c = c

    def select_action(self) -> int:
        self.t += 1
        for a in range(self.k):
            if self.N[a] == 0:
                return a
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(self.Q + uncertainty))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ============================================================
# Gradient bandit agent
# ============================================================

class GradientBandit(BanditAgent):
    """Gradient bandit with optional reward baseline."""

    def __init__(self, k: int = 10, alpha: float = 0.1, baseline: bool = True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.H = np.zeros(self.k)          # action preferences
        self.avg_reward = 0.0

    def _policy(self) -> np.ndarray:
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self) -> int:
        probs = self._policy()
        return int(np.random.choice(self.k, p=probs))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        probs = self._policy()

        baseline = 0.0
        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward

        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(
    agent: BanditAgent,
    env: KArmedBandit,
    steps: int = 1000,
    runs: int = 2000,
    desc: str = "Runs",
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``runs`` independent runs of agent-env interaction.

    Returns:
        (mean_rewards, optimal_action_frac)  both of shape ``(steps,)``.
    """
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in tqdm(range(runs), desc=desc, unit="run", leave=False):
        env.reset()
        agent.reset()
        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)
            rewards[r, t] = reward
            optimal[r, t] = float(action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)


# ============================================================
# Plot helpers
# ============================================================

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")


def plot_epsilon_greedy(steps: int = 1000, runs: int = 2000) -> None:
    env = KArmedBandit()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for eps in [0, 0.01, 0.1]:
        agent = EpsilonGreedy(epsilon=eps)
        rewards, optimal = run_experiment(agent, env, steps, runs, desc=f"eps={eps}")
        label = f"eps={eps}"
        axes[0].plot(rewards, label=label)
        axes[1].plot(100 * optimal, label=label)

    axes[0].set(xlabel="Steps", ylabel="Average reward", title="eps-greedy: reward")
    axes[1].set(xlabel="Steps", ylabel="% Optimal action", title="eps-greedy: optimal action %")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    _save(fig, "kbandits_epsilon_greedy.png")
    plt.close(fig)


def plot_optimistic_vs_ucb(steps: int = 1000, runs: int = 2000) -> None:
    env = KArmedBandit()
    agents = {
        "Optimistic greedy (Q0=5, eps=0)": EpsilonGreedy(epsilon=0, optimistic=5),
        "UCB  c=2": UCB(c=2),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps, runs, desc=name[:20])
        ax.plot(rewards, label=name)
    ax.set(xlabel="Steps", ylabel="Average reward", title="Optimistic initialisation vs UCB")
    ax.legend()
    fig.tight_layout()
    _save(fig, "kbandits_optimistic_vs_ucb.png")
    plt.close(fig)


def plot_gradient_bandit(steps: int = 1000, runs: int = 2000) -> None:
    env = KArmedBandit()
    agents = {
        "a=0.1  baseline": GradientBandit(alpha=0.1, baseline=True),
        "a=0.4  baseline": GradientBandit(alpha=0.4, baseline=True),
        "a=0.1  no baseline": GradientBandit(alpha=0.1, baseline=False),
        "a=0.4  no baseline": GradientBandit(alpha=0.4, baseline=False),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, agent in agents.items():
        _, optimal = run_experiment(agent, env, steps, runs, desc=name[:20])
        ax.plot(100 * optimal, label=name)
    ax.set(xlabel="Steps", ylabel="% Optimal action", title="Gradient bandit methods")
    ax.legend()
    fig.tight_layout()
    _save(fig, "kbandits_gradient_bandit.png")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Running K-Armed Bandit experiments...")
    print("  [1/3] epsilon-greedy comparison")
    plot_epsilon_greedy()
    print("  [2/3] Optimistic initialisation vs UCB")
    plot_optimistic_vs_ucb()
    print("  [3/3] Gradient bandit")
    plot_gradient_bandit()
    print("Done. All figures saved to mia_rl/outputs/")
