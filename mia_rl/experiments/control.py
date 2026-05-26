from __future__ import annotations

from typing import Any
from tqdm import tqdm

from mia_rl.agents.control.base import ControlAgent
from mia_rl.core.base import Transition
from mia_rl.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState


def run_control_episode(
    env: WindyGridworldEnv,
    agent: ControlAgent[WindyGridworldState, WindyGridworldAction],
    max_steps: int = 1_000,
) -> tuple[int, float]:
    
    state = env.reset()
    action = agent.select_action(state)
    episode_length = 0
    total_reward = 0.0
    done = False

    while not done and episode_length < max_steps:
        next_state, reward, done = env.step(action)
        if not done:
            next_action = agent.select_action(next_state)
        else:
            next_action = None

        agent.update_transition(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=None if done else next_state,
                done=done,
            )
        )

        state = next_state
        if next_action is not None:
            action = next_action
        episode_length += 1
        total_reward += reward

    agent.end_episode()

    return episode_length, total_reward


def train_control_agent(
    env: WindyGridworldEnv,
    agent: ControlAgent[WindyGridworldState, WindyGridworldAction],
    num_episodes: int,
    max_steps: int = 1_000,
    desc: str = "Training",
    writer: Any | None = None,
) -> tuple[list[int], list[float]]:
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    for episode_idx in tqdm(range(1, num_episodes + 1), desc=desc, unit="ep"):
        episode_length, episode_reward = run_control_episode(env, agent, max_steps=max_steps)
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        if writer is not None:
            writer.add_scalar("Control/EpisodeLength", episode_length, episode_idx)
            writer.add_scalar("Control/EpisodeReward", episode_reward, episode_idx)

    return episode_lengths, episode_rewards


#methods for extracting greedy policy and state values from a control agent, used in the plotting script
def greedy_policy_from_agent(
    env: WindyGridworldEnv,
    agent: ControlAgent[WindyGridworldState, WindyGridworldAction],
) -> dict[WindyGridworldState, WindyGridworldAction]:
    policy: dict[WindyGridworldState, WindyGridworldAction] = {}
    for state in env.states():
        if state == env.goal:
            continue
        actions = env.available_actions(state)
        policy[state] = max(actions, key=lambda action: agent.action_value_of(state, action))
    return policy


def greedy_path(
    env: WindyGridworldEnv,
    policy: dict[WindyGridworldState, WindyGridworldAction],
    max_steps: int = 100,
) -> list[WindyGridworldState]:
    state = env.start
    path = [state]

    for _ in range(max_steps):
        if state == env.goal:
            break
        action = policy.get(state)
        if action is None:
            break
        state, _, _ = env.step_from_state(state, action)
        path.append(state)
        if state == env.goal:
            break

    return path
