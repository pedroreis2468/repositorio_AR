from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Windy Gridworld with Monte Carlo control.")
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy control.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--max-steps", type=int, default=1_000, help="Maximum steps per episode before truncation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_mc_control",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from mia_rl.agents.control import MonteCarloControl
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent
    from mia_rl.plots.windy_gridworld import plot_episode_lengths, plot_episode_rewards, plot_policy

    try:
        env = WindyGridworldEnv()
        agent = MonteCarloControl(actions=ACTIONS, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)

        episode_lengths, episode_rewards = train_control_agent(env, agent, args.episodes, max_steps=args.max_steps)
        policy = greedy_policy_from_agent(env, agent)
        path = greedy_path(env, policy)

        fig_lengths, _ = plot_episode_lengths(episode_lengths, title="MC control: episode length over training")
        fig_rewards, _ = plot_episode_rewards(episode_rewards, title="MC control: episode reward over training")
        fig_policy, _ = plot_policy(env, policy, path=path, title="Windy Gridworld greedy policy after MC control training")

        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_lengths.savefig(output_dir / "windy_lengths.png", dpi=150, bbox_inches="tight")
        fig_rewards.savefig(output_dir / "windy_rewards.png", dpi=150, bbox_inches="tight")
        fig_policy.savefig(output_dir / "windy_policy.png", dpi=150, bbox_inches="tight")
        print(f"Saved plots to {output_dir}")
        print(f"Final greedy path length: {len(path) - 1}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()
    except NotImplementedError as exc:
        print("\nThis practical is not complete yet.")
        print("Please finish the TODOs in:")
        print("- mia_rl/envs/windy_gridworld.py")
        print(f"\nOriginal message: {exc}")
        return


if __name__ == "__main__":
    main()
