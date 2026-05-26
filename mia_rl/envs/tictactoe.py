from __future__ import annotations

from mia_rl.core.base import Environment

# ── Type aliases ────────────────────────────────────────────────────────────
# The board is a 9-tuple of ints (one per cell, row-major):
#   0 = empty, 1 = player X, -1 = player O
# Actions are integers 0-8 identifying the cell to mark.
TicTacToeState  = tuple[int, ...]   # length-9
TicTacToeAction = int               # 0 … 8

# Indices of every winning line (rows, columns, diagonals)
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def _winner(board: TicTacToeState) -> int:
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):
    """Two-player Tic-Tac-Toe environment.

    Conventions:
    - Player X always goes first (represented as +1 in the board).
    - Player O is represented as -1.
    - `current_player` alternates between 1 (X) and -1 (O) each step.
    - The state is a length-9 tuple representing all 9 cells row-major:
        indices  0 1 2
                 3 4 5
                 6 7 8
    - `step()` applies the current player's move, then switches turns.
    - Episode ends when a player wins or the board is full (draw).
    - Rewards from the perspective of the player who just moved:
        +1  for winning
        -1  for losing (opponent wins — not possible in one step, included for completeness)
         0  otherwise (ongoing or draw)

    For self-play, call `reset()` at the start of each game and alternate
    calling `step()` for player X and player O.
    """

    def __init__(self) -> None:
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1  # X starts

    def reset(self) -> TicTacToeState:
        """Reset the board to an empty state and set X as the first player."""
        self.board = (0,) * 9
        self.current_player = 1  # X starts
        return self.board

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        """Return the indices of all empty cells in `state`."""
        return [i for i, cell in enumerate(state) if cell == 0]

    def is_terminal(self, state: TicTacToeState) -> bool:
        """Return True if the game is over (win or draw)."""
        if _winner(state) != 0:
            return True
        return 0 not in state  # No empty cells left

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        """Place the current player's mark on cell `action` and advance the game."""
        # 1. Validação da jogada
        if self.board[action] != 0:
            raise ValueError(f"Action {action} is invalid: cell already occupied.")

        # 2. Construção do novo tabuleiro (usando fatiamento de tuplo)
        board_list = list(self.board)
        board_list[action] = self.current_player
        new_board = tuple(board_list)

        # 3. Verificar vencedor e estado terminal
        winner = _winner(new_board)
        done = (winner != 0) or (0 not in new_board)

        # 4. Recompensa para o jogador que acabou de mover
        # Se o vencedor é o current_player, recompensa = 1.0
        reward = 1.0 if winner == self.current_player else 0.0

        # 5. Atualização de estado e troca de turno
        self.board = new_board
        self.current_player *= -1  # Alterna entre 1 e -1

        return new_board, reward, done

    def render(self, state: TicTacToeState | None = None) -> None:
        """Print a human-readable board to stdout."""
        s = state if state is not None else self.board
        res = []
        for i, val in enumerate(s):
            if val == 1:
                res.append("X")
            elif val == -1:
                res.append("O")
            else:
                res.append(str(i + 1)) # 1-based index para o utilizador

        # Formatação das linhas
        for i in range(0, 9, 3):
            print(f" {res[i]} | {res[i+1]} | {res[i+2]} ")
            if i < 6:
                print("---+---+---")