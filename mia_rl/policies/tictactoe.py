from __future__ import annotations

import random
from typing import Callable

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

# Policy type: a callable that takes (env, state) and returns an action.
Policy = Callable[[TicTacToeEnv, TicTacToeState], TicTacToeAction]


def random_action(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
    """Choose a uniformly random legal action. Used as a baseline opponent."""
    return random.choice(env.available_actions(state))


def human_policy(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
    """Ask to pick a valid cell via stdin (1-based, matching the rendered board)."""
    available_0 = env.available_actions(state)
    available_1 = [a + 1 for a in available_0]  # display as 1–9
    while True:
        try:
            choice = int(input(f"Your turn! Choose a cell {available_1}: "))
        except ValueError:
            print("Please enter a number.")
            continue
        action = choice - 1  # convert back to 0-based
        if action in available_0:
            return action
        print(f"Cell {choice} is not available. Try again.")
