from __future__ import annotations

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState, _winner
from mia_rl.policies.tictactoe import Policy, human_policy


def play_game(
    env: TicTacToeEnv,
    policy_x: Policy,
    policy_o: Policy,
    render: bool = True,
) -> int:
    """Play one full game between two policies, optionally rendering each step.

    Args:
        env: the TicTacToeEnv instance.
        policy_x: callable (env, state) -> action for player X (+1).
        policy_o: callable (env, state) -> action for player O (-1).
        render: if True, print the board after every move.

    Returns:
        1 if X wins, -1 if O wins, 0 for a draw.
    """
    state = env.reset()
    if render:
        print("Initial board:")
        env.render(state)
        print()

    while not env.is_terminal(state):
        player_label = "X" if env.current_player == 1 else "O"

        policy = policy_x if env.current_player == 1 else policy_o
        action = policy(env, state)

        state, reward, done = env.step(action)

        if render:
            print(f"Player {player_label} plays cell {action + 1}:")
            env.render(state)
            print()

    result = _winner(state)
    if render:
        if result == 1:
            print("X wins!")
        elif result == -1:
            print("O wins!")
        else:
            print("Draw!")
    return result


def play_game_vs_human(
    env: TicTacToeEnv,
    agent_policy: Policy,
    human_plays: int = -1,
) -> int:
    """Play one game between an agent policy and a human player.

    Args:
        env: the TicTacToeEnv instance.
        agent_policy: callable (env, state) -> action for the agent.
        human_plays: which player the human controls: 1 for X, -1 for O (default).

    Returns:
        1 if X wins, -1 if O wins, 0 for a draw.
    """
    if human_plays not in (1, -1):
        raise ValueError("human_plays must be 1 (X) or -1 (O).")

    policy_x = human_policy if human_plays == 1 else agent_policy
    policy_o = human_policy if human_plays == -1 else agent_policy

    return play_game(env, policy_x, policy_o, render=True)
