"""Simple policies for Connect 4."""

import numpy as np
import random

# Constants for Connect 4
RED_DISK = 1
YELLOW_DISK = -1
NO_DISK = 0
PROTAGONIST_TURN = RED_DISK
OPPONENT_TURN = YELLOW_DISK


def copy_env(env, mute_env=True):
    new_env = env.__class__(
        board_size=env.board_size,
        sudden_death_on_invalid_move=env.sudden_death_on_invalid_move,
        mute=mute_env)
    new_env.reset()
    return new_env


class EasyPolicy(object):
    """Slightly more difficult policy for Connect 4."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state.copy()

        # Simple heuristic: play in the middle column if available
        middle_col = len(board_state[0]) // 2
        if middle_col in possible_moves:
            return middle_col

        # Otherwise, play randomly
        ix = random.randint(0, len(possible_moves) - 1)
        action = possible_moves[ix]
        return action


class HardPolicy(object):
    """Slightly more difficult policy for Connect 4."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state.copy()

        # Check if we can win in the next move
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 2  # assume we're playing as 2
                    break
            if self.check_win(temp_board, 2):
                return move

        # Check if opponent can win in the next move and block them
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 1  # assume opponent is playing as 1
                    break
            if self.check_win(temp_board, 1):
                return move

        # Look for opportunities to create two possible wins at the same time
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 2  # assume we're playing as 2
                    break
            if self.count_possible_wins(temp_board, 2) > 1:
                return move

        # Look for opportunities to block opponent's two possible wins at the same time
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 1  # assume opponent is playing as 1
                    break
            if self.count_possible_wins(temp_board, 1) > 1:
                return move

        # Simple heuristic: play in the middle column if available
        middle_col = len(board_state[0]) // 2
        if middle_col in possible_moves:
            return middle_col

        # Otherwise, play randomly
        ix = random.randint(0, len(possible_moves) - 1)
        action = possible_moves[ix]
        return action

    def count_possible_wins(self, board, player):
        count = 0
        for col in range(len(board[0])):
            temp_board = [row[:] for row in board]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][col] == 0:
                    temp_board[i][col] = player
                    break
            if self.check_win(temp_board, player):
                count += 1
        return count

    def check_win(self, board, player):
        # Check horizontal locations for win
        for c in range(len(board[0]) - 3):
            for r in range(len(board)):
                if board[r][c] == player and board[r][c + 1] == player and board[r][c + 2] == player and board[r][
                    c + 3] == player:
                    return True

        # Check vertical locations for win
        for c in range(len(board[0])):
            for r in range(len(board) - 3):
                if board[r][c] == player and board[r + 1][c] == player and board[r + 2][c] == player and board[r + 3][
                    c] == player:
                    return True

        # Check positively sloped diagonals
        for c in range(len(board[0]) - 3):
            for r in range(len(board) - 3):
                if board[r][c] == player and board[r + 1][c + 1] == player and board[r + 2][c + 2] == player and \
                        board[r + 3][c + 3] == player:
                    return True

        # Check negatively sloped diagonals
        for c in range(len(board[0]) - 3):
            for r in range(3, len(board)):
                if board[r][c] == player and board[r - 1][c + 1] == player and board[r - 2][c + 2] == player and \
                        board[r - 3][c + 3] == player:
                    return True

        return False


class MediumPolicy(object):
    """Slightly more difficult policy for Connect 4."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state.copy()

        # Check if we can win in the next move
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 2  # assume we're playing as 2
                    break
            if self.check_win(temp_board, 2):
                return move

        # Check if opponent can win in the next move and block them
        for move in possible_moves:
            temp_board = [row[:] for row in board_state]
            for i in range(len(temp_board) - 1, -1, -1):
                if temp_board[i][move] == 0:
                    temp_board[i][move] = 1  # assume opponent is playing as 1
                    break
            if self.check_win(temp_board, 1):
                return move

        # Simple heuristic: play in the middle column if available
        middle_col = len(board_state[0]) // 2
        if middle_col in possible_moves:
            return middle_col

        # Otherwise, play randomly
        ix = random.randint(0, len(possible_moves) - 1)
        action = possible_moves[ix]
        return action

    def check_win(self, board, player):
        # Check horizontal locations for win
        for c in range(len(board[0]) - 3):
            for r in range(len(board)):
                if board[r][c] == player and board[r][c + 1] == player and board[r][c + 2] == player and board[r][
                    c + 3] == player:
                    return True

        # Check vertical locations for win
        for c in range(len(board[0])):
            for r in range(len(board) - 3):
                if board[r][c] == player and board[r + 1][c] == player and board[r + 2][c] == player and board[r + 3][
                    c] == player:
                    return True

        # Check positively sloped diagonals
        for c in range(len(board[0]) - 3):
            for r in range(len(board) - 3):
                if board[r][c] == player and board[r + 1][c + 1] == player and board[r + 2][c + 2] == player and \
                        board[r + 3][c + 3] == player:
                    return True

        # Check negatively sloped diagonals
        for c in range(len(board[0]) - 3):
            for r in range(3, len(board)):
                if board[r][c] == player and board[r - 1][c + 1] == player and board[r - 2][c + 2] == player and \
                        board[r - 3][c + 3] == player:
                    return True

        return False
