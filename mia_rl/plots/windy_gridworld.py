from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState

ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}


def plot_episode_lengths(lengths: list[int], title: str = "Episode length over training"):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title(title)
    return fig, ax


def plot_episode_rewards(rewards: list[float], title: str = "Episode reward over training"):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    return fig, ax


def plot_policy(
    env: WindyGridworldEnv,
    policy: dict[WindyGridworldState, WindyGridworldAction],
    path: list[WindyGridworldState] | None = None,
    title: str = "Learned greedy policy",
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            wind_strength = env.wind[col]
            ax.text(col + 0.15, row + 0.2, str(wind_strength), fontsize=8, alpha=0.6)
            if state == env.goal:
                ax.text(col + 0.5, row + 0.55, "G", ha="center", va="center", fontsize=14)
                continue
            if state == env.start:
                ax.text(col + 0.25, row + 0.55, "S", ha="center", va="center", fontsize=14)
            action = policy.get(state)
            if action is not None:
                ax.text(col + 0.6, row + 0.55, ARROWS[action], ha="center", va="center", fontsize=16)

    if path is not None and len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color="tab:red", linewidth=2, marker="o", markersize=4)

    return fig, ax


def plot_td_errors(
    errors: list[float],
    window: int = 20,
    title: str = "Mean |TD error| per episode",
):
    """Plot per-episode mean absolute TD error with a rolling-mean overlay."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    arr = np.array(errors)
    ax.plot(arr, alpha=0.3, color="tab:orange", label="per episode")
    if len(arr) >= window:
        rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(arr)), rolling, color="tab:orange", label=f"{window}-ep mean")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean |δ|")
    ax.set_title(title)
    return fig, ax


def plot_value_heatmap(
    env: WindyGridworldEnv,
    value_fn,
    title: str = "Learned state values V(s)",
):
    """Heatmap of V(s) = value_fn(s) for all grid states.

    value_fn can be:
      - LinearTD0:          lambda s: agent.value_of(s)
      - LinearSarsaControl: lambda s: max(agent.action_value_of(s, a) for a in ACTIONS)
    """
    grid = np.zeros((env.rows, env.cols))
    for row in range(env.rows):
        for col in range(env.cols):
            grid[row, col] = value_fn((row, col))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(grid, origin="upper", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="V(s)")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    ax.text(start_col, start_row, "S", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(goal_col, goal_row, "G", ha="center", va="center", color="white", fontsize=14, fontweight="bold")

    return fig, ax


def plot_episode_length_comparison(
    lengths_dict: dict[str, list[int]],
    window: int = 20,
    title: str = "Episode length comparison",
):
    """Overlay episode-length curves for multiple agents with rolling-mean smoothing."""
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for label, lengths in lengths_dict.items():
        arr = np.array(lengths, dtype=float)
        ax.plot(arr, alpha=0.15)
        if len(arr) >= window:
            rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(arr)), rolling, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title(title)
    ax.legend()
    return fig, ax
