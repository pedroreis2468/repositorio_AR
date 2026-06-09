"""Run PyTorch SARSA with linear function approximation on Windy Gridworld.

Section 3 of the function-approximation class:
  Implements semi-gradient SARSA in two PyTorch modes:

  Manual update (use_optimizer=False):
      with torch.no_grad():
          w += alpha * delta * phi(s, a)
      Equivalent to LinearSarsaControl — no autograd involved.

  Optimizer update (use_optimizer=True):
      loss = 0.5 * (target.detach() - q_hat(s, a)) ** 2
      loss.backward()
      optimizer.step()
      The target MUST be detached to enforce semi-gradient.
      With 0.5 * MSE and SGD(lr=alpha), the update is identical to the
      manual version: w -= alpha * (q_hat - target) * phi = w += alpha * delta * phi.

The script produces a comparison plot of episode lengths across:
  - Tabular SARSA (from previous class)
  - Linear SARSA (NumPy)
  - Torch SARSA (manual)
  - Torch SARSA (optimizer)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch SARSA on Windy Gridworld.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Step-size for FA agents.")
    parser.add_argument("--tabular-alpha", type=float, default=0.5, help="Step-size for tabular SARSA.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_torch_sarsa",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run PyTorch on (auto, cuda, cpu).")
    return parser.parse_args()


def _rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _plot_length_comparison(lengths_by_agent: dict[str, list[int]], window: int = 20):
    import matplotlib.pyplot as plt

    colors = {
        "Tabular SARSA": "#1b9e77",
        "Linear SARSA (NumPy)": "#d95f02",
        "Torch SARSA (manual)": "#7570b3",
        "Torch SARSA (optimizer)": "#e7298a",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for label, lengths in lengths_by_agent.items():
        arr = np.asarray(lengths, dtype=float)
        color = colors[label]
        ax.plot(arr, alpha=0.12, linewidth=1.0, color=color)
        if len(arr) >= window:
            smoothed = _rolling_mean(arr, window)
            xs = np.arange(window - 1, len(arr))
        else:
            smoothed = arr
            xs = np.arange(len(arr))
        ax.plot(xs, smoothed, linewidth=2.4, color=color, label=label)
        ax.scatter(xs[-1], smoothed[-1], color=color, s=28, zorder=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title("Windy Gridworld: episode length comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return fig, ax


def _plot_td_error_panels(td_errors_by_agent: dict[str, list[float]], window: int = 20):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(td_errors_by_agent), figsize=(15, 4.2), constrained_layout=True, sharey=True)
    if len(td_errors_by_agent) == 1:
        axes = [axes]

    max_y = max(max(errors) for errors in td_errors_by_agent.values() if errors)
    for ax, (name, errors) in zip(axes, td_errors_by_agent.items()):
        arr = np.asarray(errors, dtype=float)
        ax.plot(arr, alpha=0.25, color="tab:orange", linewidth=1.0)
        if len(arr) >= window:
            smoothed = _rolling_mean(arr, window)
            ax.plot(np.arange(window - 1, len(arr)), smoothed, color="tab:orange", linewidth=2.2)
        ax.set_title(name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean |δ|")
        ax.set_ylim(0.0, max_y * 1.05 if max_y > 0 else 1.0)
        ax.grid(alpha=0.25)
    fig.suptitle("TD error comparison")
    return fig, axes


def _draw_value_panel(ax, env, agent, title: str, actions, vmin: float, vmax: float):
    grid = np.zeros((env.rows, env.cols))
    for row in range(env.rows):
        for col in range(env.cols):
            grid[row, col] = max(agent.action_value_of((row, col), action) for action in actions)
    im = ax.imshow(grid, origin="upper", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return im


def _draw_policy_panel(ax, env, policy, path, title: str):
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    ax.set_title(title)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            ax.text(col + 0.14, row + 0.18, str(env.wind[col]), fontsize=7, alpha=0.55)
            if state == env.goal:
                ax.text(col + 0.5, row + 0.55, "G", ha="center", va="center", fontsize=13)
                continue
            if state == env.start:
                ax.text(col + 0.25, row + 0.55, "S", ha="center", va="center", fontsize=13)
            action = policy.get(state)
            if action is not None:
                ax.text(col + 0.62, row + 0.55, arrows[action], ha="center", va="center", fontsize=14)

    if path is not None and len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color="tab:red", linewidth=1.8, marker="o", markersize=3.5)


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from mia_rl.agents.control import SarsaControl
    from mia_rl.agents.control.linear_sarsa import LinearSarsaControl
    from mia_rl.agents.control.torch_sarsa import TorchSarsaControl
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent
    from mia_rl.experiments.fa_training import train_fa_agent
    from mia_rl.features.windy_gridworld import STATE_ACTION_FEATURE_DIM, state_action_features

    env = WindyGridworldEnv()
    # Memoize phi: the grid is tiny (70 states × 4 actions = 280 entries) and the
    # feature function calls env.step_from_state internally, so caching eliminates
    # repeated env calls during action selection and bootstrap computation.
    _phi_cache: dict = {}
    def phi(s, a):
        key = (s, a)
        if key not in _phi_cache:
            _phi_cache[key] = state_action_features(s, a, env)
        return _phi_cache[key]

    common = dict(actions=ACTIONS, alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)
    common_fa = dict(**common, phi=phi, n_features=STATE_ACTION_FEATURE_DIM)

    agents = {
        "Tabular SARSA":           SarsaControl(actions=ACTIONS, alpha=args.tabular_alpha, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed),
        "Linear SARSA (NumPy)":    LinearSarsaControl(**common_fa),
        "Torch SARSA (manual)":    TorchSarsaControl(**common_fa, use_optimizer=False, device=args.device),
        "Torch SARSA (optimizer)": TorchSarsaControl(**common_fa, use_optimizer=True, device=args.device),
    }

    all_lengths: dict[str, list[int]] = {}
    all_td_errors: dict[str, list[float]] = {}
    policies: dict[str, dict] = {}
    paths: dict[str, list] = {}

    import datetime
    from torch.utils.tensorboard import SummaryWriter

    runs_dir = PACKAGE_ROOT / "outputs" / "runs" / "torch_sarsa_compare"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, agent in agents.items():
        print(f"Training {name} for {args.episodes} episodes...")
        run_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        writer = SummaryWriter(log_dir=runs_dir / f"{run_name}_{ts}")

        if name == "Tabular SARSA":
            lengths, _ = train_control_agent(env, agent, args.episodes, max_steps=args.max_steps, writer=writer)
            all_lengths[name] = lengths
        else:
            lengths, _, td_errors = train_fa_agent(env, agent, args.episodes, max_steps=args.max_steps, writer=writer)
            all_lengths[name] = lengths
            all_td_errors[name] = td_errors

        writer.close()

        policies[name] = greedy_policy_from_agent(env, agent)
        paths[name] = greedy_path(env, policies[name])
        print(f"  final greedy path length: {len(paths[name]) - 1}")

    fig_compare, _ = _plot_length_comparison(all_lengths)
    fig_td, _ = _plot_td_error_panels(all_td_errors)

    value_grids = {}
    for name, agent in agents.items():
        grid = np.zeros((env.rows, env.cols))
        for row in range(env.rows):
            for col in range(env.cols):
                grid[row, col] = max(agent.action_value_of((row, col), action) for action in ACTIONS)
        value_grids[name] = grid

    all_values = np.concatenate([grid.ravel() for grid in value_grids.values()])
    vmin = float(all_values.min())
    vmax = float(all_values.max())

    fig_heat, axes_heat = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    last_im = None
    for ax, (name, agent) in zip(axes_heat.ravel(), agents.items()):
        last_im = _draw_value_panel(ax, env, agent, name, ACTIONS, vmin=vmin, vmax=vmax)
    if last_im is not None:
        fig_heat.colorbar(last_im, ax=axes_heat.ravel().tolist(), shrink=0.92, label="V(s) = max_a q(s, a)")
    fig_heat.suptitle("Windy Gridworld: learned value surfaces")

    fig_policy, axes_policy = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (name, policy) in zip(axes_policy.ravel(), policies.items()):
        _draw_policy_panel(ax, env, policy, paths[name], f"{name} | path={len(paths[name]) - 1}")
    fig_policy.suptitle("Windy Gridworld: greedy policy comparison")

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_compare.savefig(output_dir / "comparison_lengths.png", dpi=150, bbox_inches="tight")
    fig_td.savefig(output_dir / "td_errors.png", dpi=150, bbox_inches="tight")
    fig_heat.savefig(output_dir / "value_heatmaps.png", dpi=150, bbox_inches="tight")
    fig_policy.savefig(output_dir / "policy_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
