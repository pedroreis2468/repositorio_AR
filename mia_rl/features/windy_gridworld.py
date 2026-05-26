from __future__ import annotations

import numpy as np

from mia_rl.envs.windy_gridworld import (
    ACTIONS,
    WindyGridworldAction,
    WindyGridworldEnv,
    WindyGridworldState,
)

# ── Tile-coding features for semi-gradient SARSA control ────────────────────
# Tile coding uses LOCAL, OVERLAPPING binary features:
# each "tile" is active for a small neighbourhood of states, so a weight
# update only changes Q-values for a few nearby states rather than
# propagating globally.  This matches the approach recommended in
# Sutton & Barto Chapter 9 for linear function approximation on grid tasks.
#
# Setup (default 7×10 grid):
#   4 tilings, each with 2×2 tiles, offset by (dr, dc) ∈ {0,1}²
#   Tiles per tiling : ceil((7+1)/2) × ceil((10+1)/2) = 4 × 6 = 24
#   State feature dim:  4 × 24 = 96
#   Action feature dim: 96 × 4 actions = 384   (action-specific block encoding)
#
# Each feature is scaled by 1/N_TILINGS (= 0.25) so that the sum of active
# features is always 1 — this keeps the effective step size constant regardless
# of alpha.

_N_TILINGS     = 4
_TILE_SIZE_ROW = 2
_TILE_SIZE_COL = 2
_TILE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))

# Number of tiles per dimension — computed for the default 7×10 grid.
# Maximum (row+dr) = 6+1 = 7  →  tile_row ∈ [0, 3]  →  4 tile rows
# Maximum (col+dc) = 9+1 = 10 →  tile_col ∈ [0, 5]  →  6 tile cols
_N_TILE_ROWS      = (7  + _TILE_SIZE_ROW - 1 + 1) // _TILE_SIZE_ROW   # = 4
_N_TILE_COLS      = (10 + _TILE_SIZE_COL - 1 + 1) // _TILE_SIZE_COL   # = 6
_TILES_PER_TILING = _N_TILE_ROWS * _N_TILE_COLS                       # = 24

# Public dimension constants
TILE_STATE_DIM        = _N_TILINGS * _TILES_PER_TILING     # = 96 (per state)
STATE_ACTION_FEATURE_DIM = TILE_STATE_DIM * len(ACTIONS)   # = 384 (per (state, action))


def tile_features(state: WindyGridworldState) -> np.ndarray:
    """96-dim tile-coded feature vector for the 7×10 Windy Gridworld.

    4 overlapping tilings each contribute exactly one active entry (= 1/4).
    All other entries are 0.  The result is a sparse vector in R^96.
    """
    row, col = state
    phi = np.zeros(TILE_STATE_DIM, dtype=np.float64)
    for t, (dr, dc) in enumerate(_TILE_OFFSETS):
        tile_row = (row + dr) // _TILE_SIZE_ROW
        tile_col = (col + dc) // _TILE_SIZE_COL
        tile_idx = tile_row * _N_TILE_COLS + tile_col
        phi[t * _TILES_PER_TILING + tile_idx] = 1.0 / _N_TILINGS
    return phi


def state_action_features(
    state: WindyGridworldState,
    action: WindyGridworldAction,
    env: WindyGridworldEnv,
    actions: tuple[WindyGridworldAction, ...] = ACTIONS,
) -> np.ndarray:
    """384-dim action-specific block encoding of tile-coded state features.

    phi(s, a) places the 96-dim tile feature vector for state s in the block
    corresponding to action a; all other blocks are zero.  The approximation is

        q_hat(s, a) = w_a · phi_tile(s)

    where w_a  is the slice of the weight vector for action a.  This is the
    standard tile-coding + linear FA approach from Sutton & Barto Chapter 9.

    The `env` argument is accepted for API compatibility but not used because
    tile widths are fixed constants for the 7×10 grid.
    """
    phi = np.zeros(STATE_ACTION_FEATURE_DIM, dtype=np.float64)
    action_idx = list(actions).index(action)
    # place state features in the block for the current action
    # 0 -> 96, 1 -> 96:192, 2 -> 192:288, 3 -> 288:384
    phi[action_idx * TILE_STATE_DIM : (action_idx + 1) * TILE_STATE_DIM] = tile_features(state)
    return phi
