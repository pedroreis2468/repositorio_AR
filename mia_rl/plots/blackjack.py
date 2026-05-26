from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.blackjack import BlackjackState

PLAYER_SUMS = tuple(range(12, 22))
DEALER_SHOWING = tuple(range(1, 11))


def values_to_array(values: dict[BlackjackState, float], usable_ace: bool) -> np.ndarray:
    arr = np.zeros((len(PLAYER_SUMS), len(DEALER_SHOWING)), dtype=float)
    for i, player_sum in enumerate(PLAYER_SUMS):
        for j, dealer_showing in enumerate(DEALER_SHOWING):
            arr[i, j] = values.get((player_sum, dealer_showing, usable_ace), 0.0)
    return arr


def plot_value_function(
    values: dict[BlackjackState, float],
    title: str = "",
    axes=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    created_figure = axes is None
    if created_figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    else:
        axes = np.asarray(axes).reshape(-1)
        fig = axes[0].figure

    axes = np.asarray(axes).reshape(-1)
    subtitles = [(False, "No usable ace"), (True, "Usable ace")]
    last_im = None

    for ax, (usable_ace, subtitle) in zip(axes, subtitles):
        arr = values_to_array(values, usable_ace)
        last_im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_xticks(range(len(DEALER_SHOWING)), DEALER_SHOWING)
        ax.set_yticks(range(len(PLAYER_SUMS)), PLAYER_SUMS)

    if title:
        fig.suptitle(title)

    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes), shrink=0.85)

    return fig, axes


def plot_value_difference(
    values_a: dict[BlackjackState, float],
    values_b: dict[BlackjackState, float],
    title: str = "Value difference",
    vmin: float | None = None,
    vmax: float | None = None,
):
    diff_values: dict[BlackjackState, float] = {}
    all_states = set(values_a) | set(values_b)
    for state in all_states:
        diff_values[state] = values_a.get(state, 0.0) - values_b.get(state, 0.0)
    return plot_value_function(diff_values, title=title, cmap="coolwarm", vmin=vmin, vmax=vmax)
