#!/usr/bin/env python3
"""Run REINFORCE (Policy Gradient) on TicTacToe.

Trains a REINFORCE policy via self-play against itself and evaluates
against a uniform random player.

Usage::

    python -m mia_rl.scripts.run_reinforce_tictactoe --episodes 10000

Saves training curves under ``mia_rl/outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE policy on TicTacToe.")
    parser.add_argument("--episodes", type=int, default=15000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--entropy-beta", type=float, default=0.05, help="Entropy regularisation coefficient.")
    parser.add_argument("--eval-every", type=int, default=1000, help="Evaluate policy every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation games.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reinforce_tictactoe",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def plot_results(history: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Plot 1: Policy Loss
    losses = history["losses"]
    axes[0].plot(losses, alpha=0.3, color="gray", label="Per-episode Loss")
    if len(losses) >= 100:
        smoothed = np.convolve(losses, np.ones(100)/100, mode="valid")
        axes[0].plot(np.arange(99, len(losses)), smoothed, color="tab:blue", linewidth=2, label="100-ep Average")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("REINFORCE Training Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Plot 2: Win rates vs Random Opponent
    cps = history["eval_checkpoints"]
    axes[1].plot(cps, history["win_rates_as_x"], "o-", label="Win Rate as X (First)", color="tab:green")
    axes[1].plot(cps, history["win_rates_as_o"], "s-", label="Win Rate as O (Second)", color="tab:orange")
    axes[1].plot(cps, history["draw_rates_as_x"], "--", label="Draw Rate as X", color="tab:green", alpha=0.6)
    axes[1].plot(cps, history["draw_rates_as_o"], ":", label="Draw Rate as O", color="tab:orange", alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Fractions")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Greedy Policy vs Random Opponent")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    return fig


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    from mia_rl.agents.control.reinforce import ReinforceAgent
    from mia_rl.experiments.reinforce_tictactoe import train

    print(f"Training REINFORCE agent (alpha={args.alpha}, beta={args.entropy_beta}) for {args.episodes} episodes...")
    agent = ReinforceAgent(
        n_features=27,   # 9 cells x 3 states (my piece, opponent, empty)
        n_actions=9,     # board cell choices
        alpha=args.alpha,
        entropy_beta=args.entropy_beta,
        seed=args.seed,
    )

    import datetime
    from torch.utils.tensorboard import SummaryWriter

    runs_dir = PACKAGE_ROOT / "outputs" / "runs" / "reinforce"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=runs_dir / f"reinforce_{ts}")

    history = train(
        agent=agent,
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        writer=writer,
    )
    writer.close()

    fig = plot_results(history)
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "reinforce_training.png", dpi=150)
    print(f"Saved plots to {output_dir / 'reinforce_training.png'}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
