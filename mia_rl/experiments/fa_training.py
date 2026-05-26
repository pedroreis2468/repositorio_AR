from __future__ import annotations

from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from mia_rl.core.base import Transition
from mia_rl.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState
from mia_rl.experiments.control import run_control_episode


def train_fa_agent(
    env: WindyGridworldEnv,
    agent,
    num_episodes: int,
    max_steps: int = 1_000,
    desc: str = "Training FA",
    writer: Any | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Train a function-approximation control agent (LinearSarsa or TorchSarsa).

    Returns episode_lengths, episode_rewards, and mean per-episode TD errors.
    The agent must expose a flush_td_errors() method.
    """
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    episode_mean_td_errors: list[float] = []

    for episode_idx in tqdm(range(1, num_episodes + 1), desc=desc, unit="ep"):
        length, reward = run_control_episode(env, agent, max_steps=max_steps)
        episode_lengths.append(length)
        episode_rewards.append(reward)
        errors = agent.flush_td_errors()    # get TD errors for this episode and reset the agent's internal TD error buffer
        mean_error = float(np.mean(errors)) if errors else 0.0
        episode_mean_td_errors.append(mean_error)
        if writer is not None:
            writer.add_scalar("FA/EpisodeLength", length, episode_idx)
            writer.add_scalar("FA/EpisodeReward", reward, episode_idx)
            writer.add_scalar("FA/MeanTDError", mean_error, episode_idx)

    return episode_lengths, episode_rewards, episode_mean_td_errors


def run_linear_td_episode(
    env: WindyGridworldEnv,
    policy,
    agent,
    max_steps: int = 1_000,
) -> tuple[int, float]:
    """One episode of online TD(0) prediction with linear function approximation.

    `policy` must expose select_action(state) -> action.
    `agent` must expose update(transition) -> delta (LinearTD0).
    Returns (episode_length, mean_abs_td_error).
    """
    state = env.reset()
    done = False
    steps = 0
    td_errors: list[float] = []

    while not done and steps < max_steps:
        action = policy.select_action(state)
        next_state, reward, done = env.step(action)
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=None if done else next_state,
            done=done,
        )
        delta = agent.update(transition)    # updates weights and returns the TD error for this transition
        td_errors.append(abs(delta))    # delta is the TD error for this transition
        state = next_state
        steps += 1

    return steps, float(np.mean(td_errors)) if td_errors else 0.0


def train_linear_td_agent(
    env: WindyGridworldEnv,
    policy,
    agent,
    num_episodes: int,
    max_steps: int = 1_000,
    desc: str = "Training Linear TD",
    writer: Any | None = None,
) -> tuple[list[int], list[float]]:
    """Train a LinearTD0 agent over multiple episodes using a fixed policy."""
    episode_lengths: list[int] = []
    episode_mean_td_errors: list[float] = []

    for episode_idx in tqdm(range(1, num_episodes + 1), desc=desc, unit="ep"):
        length, mean_error = run_linear_td_episode(env, policy, agent, max_steps)
        episode_lengths.append(length)
        episode_mean_td_errors.append(mean_error)
        if writer is not None:
            writer.add_scalar("LinearTD/EpisodeLength", length, episode_idx)
            writer.add_scalar("LinearTD/MeanTDError", mean_error, episode_idx)

    return episode_lengths, episode_mean_td_errors
