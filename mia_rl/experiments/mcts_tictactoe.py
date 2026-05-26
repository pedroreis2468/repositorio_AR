from __future__ import annotations

from mia_rl.agents.planning.mcts import MCTSAgent
from mia_rl.envs.tictactoe import (
    TicTacToeAction,
    TicTacToeEnv,
    TicTacToeState,
    _winner,
)
from mia_rl.policies.tictactoe import Policy, random_action


# ── Policy wrapper ────────────────────────────────────────────────────────────


def make_mcts_policy(agent: MCTSAgent) -> Policy:
    """Wrap an MCTSAgent as a ``Policy`` callable compatible with ``play_game``.

    A fresh MCTS search is run from scratch on every move call; no tree is
    reused between moves.

    Args:
        agent: configured MCTSAgent (n_simulations, c already set).

    Returns:
        A ``Policy`` callable: ``(env, state) → action``.
    """

    def policy(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
        return agent.select_action(state, env.current_player)  # full MCTS search

    return policy


# ── Evaluation helpers ────────────────────────────────────────────────────────


def evaluate_vs_random(
    env: TicTacToeEnv,
    agent: MCTSAgent,
    n_games: int = 200,
    as_player: int = 1,
) -> tuple[float, float, float]:
    """Evaluate MCTS win/draw/loss rates against a uniform-random opponent.

    Args:
        env:       TicTacToe environment instance.
        agent:     MCTSAgent to evaluate.
        n_games:   number of evaluation games.
        as_player: +1 → MCTS plays X (first mover); -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` — three fractions summing to 1.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == as_player:
                action = agent.select_action(state, env.current_player)  # MCTS move
            else:
                action = random_action(env, state)  # random opponent
            state, _, done = env.step(action)

        winner = _winner(state)
        if winner == as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / n_games
    draw_rate = draws / n_games
    loss_rate = losses / n_games

    return win_rate, draw_rate, loss_rate


def evaluate_mcts_vs_reinforce(
    env: TicTacToeEnv,
    mcts_agent: MCTSAgent,
    reinforce_policy: Policy,
    n_games: int = 200,
    mcts_as_player: int = 1,
) -> tuple[float, float, float]:
    """Pit MCTS against a trained REINFORCE policy.

    Args:
        env:              TicTacToe environment instance.
        mcts_agent:       MCTSAgent (planning, no learning).
        reinforce_policy: trained REINFORCE policy from ``make_reinforce_policy``.
        n_games:          number of games to play.
        mcts_as_player:   +1 → MCTS plays X; -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` for MCTS.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == mcts_as_player:
                action = mcts_agent.select_action(state, env.current_player)  # MCTS
            else:
                action = reinforce_policy(env, state)  # REINFORCE
            state, _, done = env.step(action)

        winner = _winner(state)
        if winner == mcts_as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / n_games
    draw_rate = draws / n_games
    loss_rate = losses / n_games

    return win_rate, draw_rate, loss_rate
