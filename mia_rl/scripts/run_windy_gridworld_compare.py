from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Sarsa, n-step Sarsa, and MC control on Windy Gridworld.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--n-steps", type=int, default=4, help="Number of steps for n-step Sarsa.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Step-size for Sarsa algorithms.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode before truncation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_comparison",
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
    from matplotlib.lines import Line2D
    import random

    from mia_rl.agents.control import MonteCarloControl, NStepSarsaControl, SarsaControl
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent
    from mia_rl.plots.windy_gridworld import plot_policy

    try:
        experiment_configs = {
            "SARSA": {
                "agent_cls": SarsaControl,
                "kwargs": {"actions": ACTIONS, "alpha": args.alpha, "epsilon": args.epsilon, "gamma": args.gamma, "seed": args.seed},
                "episodes": 500,
            },
            f"{args.n_steps}-Step SARSA": {
                "agent_cls": NStepSarsaControl,
                "kwargs": {"actions": ACTIONS, "n_steps": args.n_steps, "alpha": args.alpha, "epsilon": args.epsilon, "gamma": args.gamma, "seed": args.seed},
                "episodes": 500,
            },
            "Monte Carlo": {
                "agent_cls": MonteCarloControl,
                "kwargs": {"actions": ACTIONS, "epsilon": args.epsilon, "gamma": args.gamma, "seed": args.seed},
                "episodes": 1000,
            },
        }

        results = {}
        env = WindyGridworldEnv() 

        # Train and evaluate
        for name, config in experiment_configs.items():
            print(f"\nTraining {name} for {config['episodes']} episodes...")
            
            random.seed(args.seed)
            current_env = WindyGridworldEnv()
            agent = config["agent_cls"](**config["kwargs"])

            lengths, rewards = train_control_agent(current_env, agent, config["episodes"], max_steps=args.max_steps)
            policy = greedy_policy_from_agent(current_env, agent)
            path = greedy_path(current_env, policy)
            
            results[name] = {
                "lengths": lengths,
                "rewards": rewards,
                "policy": policy,
                "path": path
            }
            print(f"{name} final greedy path length: {len(path) - 1}")

        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot Episode Lengths
        fig_lengths, ax_lengths = plt.subplots(figsize=(10, 6))
        for name, data in results.items():
            ax_lengths.plot(data["lengths"], label=name, alpha=0.7)
        ax_lengths.set_xlabel("Episode")
        ax_lengths.set_ylabel("Steps per episode")
        ax_lengths.set_title("Agent Comparison: Episode Lengths over Training")
        ax_lengths.legend()
        fig_lengths.savefig(output_dir / "comparison_lengths.png", dpi=150, bbox_inches="tight")

        # Plot Episode Rewards
        fig_rewards, ax_rewards = plt.subplots(figsize=(10, 6))
        for name, data in results.items():
            ax_rewards.plot(data["rewards"], label=name, alpha=0.7)
        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Reward per episode")
        ax_rewards.set_title("Agent Comparison: Episode Rewards over Training")
        ax_rewards.legend()
        fig_rewards.savefig(output_dir / "comparison_rewards.png", dpi=150, bbox_inches="tight")

        # Plot Individual Policies
        for name, data in results.items():
            clean_name = name.lower().replace(" ", "_").replace("-", "_")
            fig_policy, _ = plot_policy(
                env, 
                data["policy"], 
                path=data["path"], 
                title=f"Windy Gridworld policy: {name}"
            )
            fig_policy.savefig(output_dir / f"policy_{clean_name}.png", dpi=150, bbox_inches="tight")

        # Plot Combined Paths on a Single Grid with FULL Legend
        fig_paths, ax_paths = plt.subplots(figsize=(10, 6))
        
        ax_paths.set_xlim(-0.5, env.cols - 0.5)
        ax_paths.set_ylim(-0.5, env.rows - 0.5)
        ax_paths.set_xticks([x - 0.5 for x in range(env.cols + 1)], minor=True)
        ax_paths.set_yticks([y - 0.5 for y in range(env.rows + 1)], minor=True)
        ax_paths.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax_paths.set_xticks(range(env.cols))
        ax_paths.set_yticks(range(env.rows))
        ax_paths.invert_yaxis() 
        
        # Annotate Cells: Wind values, Start (S), and Goal (G)
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) == env.start:
                    ax_paths.text(c, r, 'S', ha='center', va='center', fontsize=20, fontweight='bold', color='black', zorder=5)
                elif (r, c) == env.goal:
                    ax_paths.text(c, r, 'G', ha='center', va='center', fontsize=20, fontweight='bold', color='black', zorder=5)
                
                ax_paths.text(
                    c, r, str(env.wind[c]), 
                    ha='center', va='center', 
                    fontsize=24, color='dodgerblue', alpha=0.15, fontweight='bold', zorder=1
                )

        # Overlay paths
        for idx, (name, data) in enumerate(results.items()):
            path = data["path"]
            if not path:
                continue
            cols = [state[1] for state in path]
            rows = [state[0] for state in path]
            offset = (idx - 1) * 0.15 
            cols_offset = [c + offset for c in cols]
            rows_offset = [r + offset for r in rows]
            
            ax_paths.plot(cols_offset, rows_offset, label=name, marker='o', markersize=5, linewidth=2, alpha=0.9, zorder=3)

        # Build Custom Full Legend
        handles, labels = ax_paths.get_legend_handles_labels() # Gets the 3 agent lines
        
        # Add proxy handles for the text elements
        handles.extend([
            Line2D([0], [0], marker='$\mathbf{S}$', color='w', markerfacecolor='black', markersize=14, label='Start State'),
            Line2D([0], [0], marker='$\mathbf{G}$', color='w', markerfacecolor='black', markersize=14, label='Goal State'),
            Line2D([0], [0], marker='$\mathbf{\#}$', color='w', markeredgecolor='dodgerblue', markerfacecolor='dodgerblue', markersize=14, alpha=0.5, label='Wind Strength')
        ])
        
        ax_paths.set_title("Agent Comparison: Final Greedy Paths & Wind Strengths")
        
        # Move legend outside the grid to the right, and use tight_layout so it isn't cropped
        ax_paths.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Legend")
        fig_paths.tight_layout() 
        
        fig_paths.savefig(output_dir / "comparison_combined_paths.png", dpi=150)

        print(f"\nSaved all comparison plots, individual policies, and combined path map to {output_dir}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()

    except NotImplementedError as exc:
        print("\nEnsure all practicals are complete before running the comparison.")
        print(f"\nOriginal message: {exc}")
        return


if __name__ == "__main__":
    main()
