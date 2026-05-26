"""Run TD(0) with linear function approximation on Windy Gridworld.

Section 1 of the function-approximation class:
  - We first train a SarsaControl agent to get a reasonable policy.
  - We then use that policy as a fixed evaluation target and train LinearTD0
    to estimate its state-value function v_pi(s) = w · phi(s).
  - The result is visualised as a value heatmap and TD-error curve.
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
    parser = argparse.ArgumentParser(description="TD(0) with linear function approximation.")
    parser.add_argument("--sarsa-episodes", type=int, default=5_000, help="Episodes to pre-train the behaviour policy.")
    parser.add_argument("--td-episodes", type=int, default=2_000, help="Episodes to train LinearTD0.")
    parser.add_argument("--alpha", type=float, default=0.02, help="TD(0) step-size.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_linear_td",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--max-steps", type=int, default=1_000, help="Max steps per episode.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    from mia_rl.agents.control import SarsaControl
    from mia_rl.agents.prediction.linear_td import LinearTD0
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.experiments.control import train_control_agent
    from mia_rl.experiments.fa_training import train_linear_td_agent
    from mia_rl.features.windy_gridworld import TILE_STATE_DIM, tile_features
    from mia_rl.plots.windy_gridworld import plot_td_errors, plot_value_heatmap

    env = WindyGridworldEnv()

    # Step 1: train a Sarsa agent to obtain a reasonable behaviour policy
    print(f"Pre-training SarsaControl for {args.sarsa_episodes} episodes...")
    sarsa = SarsaControl(actions=ACTIONS, alpha=0.5, epsilon=0.1, gamma=args.gamma, seed=args.seed)
    train_control_agent(env, sarsa, args.sarsa_episodes, max_steps=args.max_steps)

    # Step 2: evaluate the epsilon-greedy behaviour policy with LinearTD0.
    # We keep epsilon at 0.1 so that all grid states are reachable and episodes
    # always terminate within a reasonable number of steps.  The resulting
    # value function estimates V^pi for this soft policy.
    phi = tile_features  # tile coding: 4 tilings × 24 tiles = 96-dim sparse features
    td_agent = LinearTD0(phi=phi, n_features=TILE_STATE_DIM, alpha=args.alpha, gamma=args.gamma)

    print(f"Training LinearTD0 for {args.td_episodes} episodes...")
    _, td_error_curve = train_linear_td_agent(
        env,
        policy=sarsa,
        agent=td_agent,
        num_episodes=args.td_episodes,
        max_steps=args.max_steps,
    )

    # Plots
    fig_heatmap, _ = plot_value_heatmap(
        env,
        value_fn=td_agent.value_of,
        title="LinearTD0 value estimates V(s) = w · phi(s)",
    )
    fig_errors, _ = plot_td_errors(td_error_curve, title="LinearTD0: mean |TD error| per episode")

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_heatmap.savefig(output_dir / "value_heatmap.png", dpi=150, bbox_inches="tight")
    fig_errors.savefig(output_dir / "td_errors.png", dpi=150, bbox_inches="tight")
    print(f"Saved plots to {output_dir}")
    print(f"Weight norm: {float(np.linalg.norm(td_agent.w)):.3f}, max|w|: {float(np.max(np.abs(td_agent.w))):.3f}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
