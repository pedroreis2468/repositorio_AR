from __future__ import annotations

import numpy as np

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

# Each of the 9 cells is encoded as a 3-dim one-hot vector (from the current
# player's perspective): [my piece, opponent's piece, empty].
# Total feature dimension: 9 × 3 = 27.
STATE_FEATURE_DIM: int = 27


def encode_state(board: TicTacToeState, current_player: int) -> np.ndarray:
    """Encode a board as a 27-dim one-hot vector from `current_player`'s perspective.

    For each cell the encoding is:
        [1, 0, 0]  if the cell contains current_player's mark
        [0, 1, 0]  if the cell contains the opponent's mark
        [0, 0, 1]  if the cell is empty

    Using a perspective-relative encoding means the same policy weights work
    regardless of whether the agent is playing as X (+1) or O (-1).

    Args:
        board: length-9 tuple of ints (0 = empty, 1 = X, -1 = O).
        current_player: +1 or -1, identifies whose turn it is.

    Returns:
        np.ndarray of shape (27,), dtype float32.
    """
    phi = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    for i, cell in enumerate(board):
        if cell == current_player:
            phi[i * 3 + 0] = 1.0   # my piece
        elif cell == -current_player:
            phi[i * 3 + 1] = 1.0   # opponent's piece
        else:
            phi[i * 3 + 2] = 1.0   # empty
    return phi
