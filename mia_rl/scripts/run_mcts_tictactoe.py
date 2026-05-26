#!/usr/bin/env python3
"""Run Monte Carlo Tree Search (MCTS) evaluation on TicTacToe.

Evaluates an MCTSAgent playing TicTacToe against a uniform random opponent.
Allows configuring the number of search simulations and the exploration constant.

Usage::

    python -m mia_rl.scripts.run_mcts_tictactoe --simulations 100 --games 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MCTS on TicTacToe.")
    parser.add_argument("--simulations", type=int, default=300, help="Number of MCTS simulations per move.")
    parser.add_argument("--c", type=float, default=1.414, help="UCB exploration constant.")
    parser.add_argument("--games", type=int, default=100, help="Number of evaluation games.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from mia_rl.agents.planning.mcts import MCTSAgent
    from mia_rl.envs.tictactoe import TicTacToeEnv
    from mia_rl.experiments.mcts_tictactoe import evaluate_vs_random
    from tabulate import tabulate

    print(f"Evaluating MCTSAgent (simulations={args.simulations}, c={args.c:.3f}) over {args.games} games...")
    env = TicTacToeEnv()
    agent = MCTSAgent(n_simulations=args.simulations, c=args.c)

    # 1. As player X (First mover)
    win_x, draw_x, loss_x = evaluate_vs_random(env, agent, n_games=args.games, as_player=1)

    # 2. As player O (Second mover)
    win_o, draw_o, loss_o = evaluate_vs_random(env, agent, n_games=args.games, as_player=-1)

    table = [
        ["MCTS Plays", "Win Rate", "Draw Rate", "Loss Rate"],
        ["X (First)", f"{win_x:.1%}", f"{draw_x:.1%}", f"{loss_x:.1%}"],
        ["O (Second)", f"{win_o:.1%}", f"{draw_o:.1%}", f"{loss_o:.1%}"],
    ]

    print("\nResults vs Uniform Random Opponent:")
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


if __name__ == "__main__":
    main()
