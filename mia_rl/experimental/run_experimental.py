#!/usr/bin/env python3
"""Run and verify experimental RL agents with TensorBoard logging.

Trains Double Q-learning on Windy Gridworld and Deep Q-Network (DQN) on TicTacToe.
Saves logs to ``mia_rl/outputs/runs/experimental/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.tensorboard import SummaryWriter

from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
from mia_rl.envs.tictactoe import TicTacToeEnv, _winner
from mia_rl.features.tictactoe import encode_state
from mia_rl.policies.tictactoe import random_action
from mia_rl.experimental.double_q import DoubleQLearningControl
from mia_rl.experimental.dqn import DQNAgent
from mia_rl.experiments.control import train_control_agent


def run_double_q_experiment(episodes: int = 200, log_dir: str | Path = "runs/double_q") -> None:
    print(f"\n--- Training Double Q-learning on Windy Gridworld ({episodes} episodes) ---")
    env = WindyGridworldEnv()
    agent = DoubleQLearningControl(actions=ACTIONS, alpha=0.5, epsilon=0.1, gamma=1.0, seed=42)
    
    writer = SummaryWriter(log_dir=log_dir)
    lengths, rewards = train_control_agent(
        env=env,
        agent=agent,
        num_episodes=episodes,
        desc="Double Q",
        writer=writer
    )
    writer.close()
    
    print(f"Double Q-learning training complete. Last 10 episodes average length: {np.mean(lengths[-10:]):.1f}")


def run_dqn_tictactoe_experiment(episodes: int = 1000, eval_every: int = 100, log_dir: str | Path = "runs/dqn") -> None:
    print(f"\n--- Training DQN on TicTacToe ({episodes} episodes) ---")
    env = TicTacToeEnv()
    agent = DQNAgent(n_features=27, n_actions=9, alpha=0.001, gamma=0.99, epsilon=0.15, seed=42)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    losses = []
    win_rates = []
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        agent_player = 1 if ep % 2 == 0 else -1  # Alternate sides
        
        last_phi = None
        last_action = None
        
        done = False
        while not done:
            player = env.current_player
            available = env.available_actions(state)
            
            if player == agent_player:
                phi = encode_state(state, player)
                action = agent.select_action(phi, available)
                
                # If we made a previous move, store transition for that move now
                if last_phi is not None and last_action is not None:
                    agent.store_transition(last_phi, last_action, 0.0, phi, False)
                
                next_state, reward, done = env.step(action)
                
                last_phi = phi
                last_action = action
                
                # If the move ended the game, store the transition with final reward
                if done:
                    # reward is 1.0 if agent won, 0.0 if draw
                    final_reward = 1.0 if reward == 1.0 else 0.0
                    agent.store_transition(last_phi, last_action, final_reward, None, True)
            else:
                action = random_action(env, state)
                next_state, reward, done = env.step(action)
                
                # If opponent won, penalize agent's last move
                if done and reward == 1.0:
                    if last_phi is not None and last_action is not None:
                        agent.store_transition(last_phi, last_action, -1.0, None, True)
                    
            state = next_state
            
            # Step the optimization model
            loss = agent.update_model()
            if loss is not None:
                losses.append(loss)
                writer.add_scalar("DQN/Loss", loss, agent.update_count)
                
        # Periodic evaluation vs random opponent
        if ep % eval_every == 0:
            wins = 0
            n_eval_games = 50
            for _ in range(n_eval_games):
                eval_state = env.reset()
                eval_agent_player = 1 if _ % 2 == 0 else -1
                
                while not env.is_terminal(eval_state):
                    curr_p = env.current_player
                    avail = env.available_actions(eval_state)
                    if curr_p == eval_agent_player:
                        phi = encode_state(eval_state, curr_p)
                        act = agent.greedy_action(phi, avail)
                        eval_state, _, _ = env.step(act)
                    else:
                        act = random_action(env, eval_state)
                        eval_state, _, _ = env.step(act)
                
                winner = _winner(eval_state)
                if winner == eval_agent_player:
                    wins += 1
            
            win_rate = wins / n_eval_games
            win_rates.append(win_rate)
            writer.add_scalar("DQN/EvalWinRate", win_rate, ep)
            print(f"Episode {ep:4d} | Average Loss: {np.mean(losses[-100:]):.4f} if losses else 0.0 | Win Rate vs Random: {win_rate:.1%}")
            
    writer.close()
    print("DQN training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental agents.")
    parser.add_argument("--double-q-episodes", type=int, default=150, help="Episodes for Double Q.")
    parser.add_argument("--dqn-episodes", type=int, default=500, help="Episodes for DQN.")
    args = parser.parse_args()
    
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "experimental"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.perf_counter()
    run_double_q_experiment(episodes=args.double_q_episodes, log_dir=outputs_dir / "double_q")
    run_dqn_tictactoe_experiment(episodes=args.dqn_episodes, eval_every=100, log_dir=outputs_dir / "dqn")
    print(f"\nAll experiments complete in {time.perf_counter() - t0:.2f}s!")
    print(f"TensorBoard logs written to: {outputs_dir}")


if __name__ == "__main__":
    main()
