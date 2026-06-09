#!/usr/bin/env python3
"""Consolidated Portfolio Benchmark.

Runs a comparative analysis of the various algorithms implemented across
the portfolio and prints a summary table.

  1. Blackjack Value Estimation (MC vs TD0)
  2. Windy Gridworld Control (Tabular vs Linear vs Torch SARSA)
  3. TicTacToe Tournament (Random vs REINFORCE vs MCTS)

Usage::

    python -m mia_rl.scripts.run_benchmark
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def benchmark_blackjack() -> list[list[str]]:
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction
    from mia_rl.policies.blackjack import ThresholdPolicy
    from mia_rl.experiments.training import train_prediction_agent

    print("\n[1/3] Benchmarking Blackjack Prediction Agents...")
    env = BlackjackEnv(seed=42)
    policy = ThresholdPolicy(threshold=20)

    mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
    td_agent = TD0Prediction(alpha=0.05, gamma=1.0)

    # Short run for benchmark
    episodes = 5000

    t0 = time.perf_counter()
    mc_history = train_prediction_agent(env, policy, mc_agent, episodes, desc="MC Prediction")
    t_mc = time.perf_counter() - t0

    t0 = time.perf_counter()
    td_history = train_prediction_agent(env, policy, td_agent, episodes, desc="TD0 Prediction")
    t_td = time.perf_counter() - t0

    # Calculate average absolute difference between TD0 and MC estimates
    mc_vals = np.array(list(mc_history[episodes].values()))
    td_vals = np.array(list(td_history[episodes].values()))
    mean_diff = np.mean(np.abs(mc_vals - td_vals))

    return [
        ["First-Visit MC", f"{episodes} episodes", f"{t_mc:.2f}s", "Reference"],
        ["TD(0)", f"{episodes} episodes", f"{t_td:.2f}s", f"Mean Abs Diff: {mean_diff:.4f}"],
    ]


def benchmark_windy_gridworld() -> list[list[str]]:
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.agents.control import SarsaControl
    from mia_rl.agents.control.linear_sarsa import LinearSarsaControl
    from mia_rl.agents.control.torch_sarsa import TorchSarsaControl
    from mia_rl.experiments.control import train_control_agent, greedy_path, greedy_policy_from_agent
    from mia_rl.experiments.fa_training import train_fa_agent
    from mia_rl.features.windy_gridworld import STATE_ACTION_FEATURE_DIM, state_action_features

    print("\n[2/3] Benchmarking Windy Gridworld Control Agents...")
    env = WindyGridworldEnv()
    episodes = 1000

    def phi(s, a):
        return state_action_features(s, a, env)

    common_fa = dict(actions=ACTIONS, phi=phi, n_features=STATE_ACTION_FEATURE_DIM, alpha=0.5, epsilon=0.1, gamma=1.0, seed=42)

    agents = {
        "Tabular SARSA": SarsaControl(actions=ACTIONS, alpha=0.5, epsilon=0.1, gamma=1.0, seed=42),
        "Linear SARSA (NumPy)": LinearSarsaControl(**common_fa),
        "Torch SARSA (manual)": TorchSarsaControl(**common_fa, use_optimizer=False),
        "Torch SARSA (optimizer)": TorchSarsaControl(**common_fa, use_optimizer=True),
    }

    rows = []
    for name, agent in agents.items():
        t0 = time.perf_counter()
        if name == "Tabular SARSA":
            lengths, _ = train_control_agent(env, agent, episodes, desc=f"{name[:15]}")
        else:
            lengths, _, _ = train_fa_agent(env, agent, episodes, desc=f"{name[:15]}")
        dur = time.perf_counter() - t0

        policy = greedy_policy_from_agent(env, agent)
        path = greedy_path(env, policy)
        final_len = len(path) - 1

        # Average length of last 10 episodes
        avg_len_last_10 = np.mean(lengths[-10:])

        rows.append([name, f"{dur:.2f}s", f"{avg_len_last_10:.1f}", f"{final_len}"])

    return rows


def benchmark_tictactoe() -> list[list[str]]:
    from mia_rl.envs.tictactoe import TicTacToeEnv
    from mia_rl.agents.planning.mcts import MCTSAgent
    from mia_rl.agents.control.reinforce import ReinforceAgent
    from mia_rl.experiments.reinforce_tictactoe import train, make_reinforce_policy
    from mia_rl.experiments.mcts_tictactoe import make_mcts_policy, evaluate_mcts_vs_reinforce, evaluate_vs_random as evaluate_mcts_vs_random
    from mia_rl.experiments.reinforce_tictactoe import evaluate_vs_random as evaluate_reinforce_vs_random
    from mia_rl.policies.tictactoe import random_action

    print("\n[3/3] Benchmarking TicTacToe Agents...")
    env = TicTacToeEnv()

    # 1. Train a quick REINFORCE agent (short training for benchmark)
    print("  Training REINFORCE agent (5000 episodes)...")
    reinforce_agent = ReinforceAgent(n_features=27, n_actions=9, alpha=0.01, entropy_beta=0.05, seed=42)
    _ = train(reinforce_agent, num_episodes=5000, eval_every=1000, eval_episodes=50, seed=42)
    reinforce_policy = make_reinforce_policy(reinforce_agent, greedy=True)

    # 2. Configure MCTS Agent
    mcts_agent = MCTSAgent(n_simulations=100, c=1.414)
    mcts_policy = make_mcts_policy(mcts_agent)

    # 3. Tournament evaluations
    games = 50
    print(f"  Running tournament matches ({games} games each)...")

    # MCTS vs Random
    win_mcts_rnd, draw_mcts_rnd, loss_mcts_rnd = evaluate_mcts_vs_random(env, mcts_agent, n_games=games, as_player=1)

    # REINFORCE vs Random
    win_re_rnd, draw_re_rnd, loss_re_rnd = evaluate_reinforce_vs_random(env, reinforce_agent, n_games=games, as_player=1)

    # MCTS vs REINFORCE
    win_mcts_re, draw_mcts_re, loss_mcts_re = evaluate_mcts_vs_reinforce(env, mcts_agent, reinforce_policy, n_games=games, mcts_as_player=1)

    return [
        ["MCTS (100 sim) vs Random", f"{win_mcts_rnd:.1%}", f"{draw_mcts_rnd:.1%}", f"{loss_mcts_rnd:.1%}"],
        ["REINFORCE (5k ep) vs Random", f"{win_re_rnd:.1%}", f"{draw_re_rnd:.1%}", f"{loss_re_rnd:.1%}"],
        ["MCTS vs REINFORCE", f"{win_mcts_re:.1%}", f"{draw_mcts_re:.1%}", f"{loss_mcts_re:.1%}"],
    ]


def main() -> None:
    print("================================================================================")
    print("                        PORTFOLIO COMPARATIVE BENCHMARK                         ")
    print("================================================================================")

    bj_results = benchmark_blackjack()
    wg_results = benchmark_windy_gridworld()
    ttt_results = benchmark_tictactoe()

    print("\n" + "="*80)
    print("                                BENCHMARK SUMMARY                               ")
    print("" + "="*80)

    # Blackjack Table
    print("\n--- 1. Blackjack Prediction (MC vs TD0) ---")
    headers_bj = ["Algorithm", "Configuration", "Training Time", "Metric"]
    print(tabulate(bj_results, headers=headers_bj, tablefmt="grid"))

    # Windy Gridworld Table
    print("\n--- 2. Windy Gridworld Control (SARSA Variants) ---")
    headers_wg = ["Agent", "Training Time", "Avg Length (Last 10)", "Final Path Length"]
    print(tabulate(wg_results, headers=headers_wg, tablefmt="grid"))

    # TicTacToe Table
    print("\n--- 3. TicTacToe Tournament (Planning vs Learning) ---")
    headers_ttt = ["Matchup (Player X vs Player O)", "X Win Rate", "Draw Rate", "O Win Rate"]
    print(tabulate(ttt_results, headers=headers_ttt, tablefmt="grid"))

    print("\n================================================================================")


if __name__ == "__main__":
    main()
