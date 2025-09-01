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


class EasyPolicy:
    """Random policy with a bias towards the center of the board for Connect 4."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        self.env = env.env if hasattr(env, 'env') else env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_size = self.env.board_size
        center_column = board_size // 2

        # Bias towards center
        best_move = min(possible_moves, key=lambda move: abs(move - center_column))
        return best_move


class MediumPolicy:
    """More difficult policy for Connect 4."""

    def __init__(self):
        self.env = None
        self.prev_moves = []

    def reset(self, env):
        self.env = env.env if hasattr(env, 'env') else env
        self.prev_moves = []

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_size = self.env.board_size
        center_column = board_size // 2

        def score_move(move):
            # Center bias
            center_bias = abs(move - center_column)
            # Defensive bias: penalize proximity to previous moves
            defensive_bias = sum(1 for prev_move in self.prev_moves if abs(move - prev_move) < 2)
            return center_bias + defensive_bias

        best_move = min(possible_moves, key=score_move)
        self.prev_moves.append(best_move)
        return best_move


class HardPolicy:
    """Even more difficult policy for Connect 4."""

    def __init__(self):
        self.env = None
        self.prev_moves = []
        self.winning_lines = []

    def reset(self, env):
        self.env = env.env if hasattr(env, 'env') else env
        self.prev_moves = []
        self.winning_lines = self.precompute_winning_lines()

    def precompute_winning_lines(self):
        # Generate all possible winning line combinations based on the board size
        board_size = self.env.board_size
        lines = []
        for i in range(board_size):
            for j in range(board_size):
                # Horizontal
                if j <= board_size - 4:
                    lines.append([(i, j + k) for k in range(4)])
                # Vertical
                if i <= board_size - 4:
                    lines.append([(i + k, j) for k in range(4)])
                # Diagonal (\)
                if i <= board_size - 4 and j <= board_size - 4:
                    lines.append([(i + k, j + k) for k in range(4)])
                # Anti-diagonal (/)
                if i >= 3 and j <= board_size - 4:
                    lines.append([(i - k, j + k) for k in range(4)])
        return lines

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state
        board_size = self.env.board_size
        center_column = board_size // 2

        def score_move(move):
            # Center bias
            center_bias = abs(move - center_column)
            # Defensive bias: penalize proximity to previous moves
            defensive_bias = sum(1 for prev_move in self.prev_moves if abs(move - prev_move) < 2)
            # Winning potential
            winning_bias = sum(1 for line in self.winning_lines if (move in [col for _, col in line]))
            return center_bias + defensive_bias - winning_bias

        best_move = min(possible_moves, key=score_move)
        self.prev_moves.append(best_move)
        return best_move
