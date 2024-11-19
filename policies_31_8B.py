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
    """Random policy with a bias towards the center of the board for Connect 4."""

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

        # Calculate the center of the board
        board_size = self.env.board_size
        center_column = board_size // 2

        # Create a list of possible moves with a bias towards the center
        biased_moves = []
        for move in possible_moves:
            # Assign a score to each move based on its distance from the center
            score = abs(move - center_column)
            biased_moves.append((move, score))

        # Sort the biased moves by their score
        biased_moves.sort(key=lambda x: x[1])

        # Choose a random move from the biased list
        ix = random.randint(0, len(biased_moves) - 1)
        action = biased_moves[ix][0]
        return action


class MediumPolicy(object):
    """More difficult policy for Connect 4."""

    def __init__(self):
        self.env = None
        self.prev_moves = []

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env
        self.prev_moves = []

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state.copy()

        # Calculate the center of the board
        board_size = self.env.board_size
        center_column = board_size // 2

        # Create a list of possible moves with a bias towards the center
        biased_moves = []
        for move in possible_moves:
            # Assign a score to each move based on its distance from the center
            score = abs(move - center_column)
            # Also assign a score based on the opponent's previous moves
            prev_score = 0
            for prev_move in self.prev_moves:
                if abs(move - prev_move) < 2:
                    prev_score += 1
            biased_moves.append((move, score + prev_score))

        # Sort the biased moves by their score
        biased_moves.sort(key=lambda x: x[1], reverse=True)

        # Choose a random move from the biased list
        ix = random.randint(0, len(biased_moves) - 1)
        action = biased_moves[ix][0]

        # Store the chosen move as the opponent's previous move
        self.prev_moves.append(action)

        return action


class HardPolicy(object):
    """Even more difficult policy for Connect 4."""

    def __init__(self):
        self.env = None
        self.prev_moves = []
        self.winning_lines = []

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env
        self.prev_moves = []
        self.winning_lines = []

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        board_state = self.env.board_state.copy()

        # Calculate the center of the board
        board_size = self.env.board_size
        center_column = board_size // 2

        # Create a list of possible moves with a bias towards the center
        biased_moves = []
        for move in possible_moves:
            # Assign a score to each move based on its distance from the center
            score = abs(move - center_column)
            # Also assign a score based on the opponent's previous moves
            prev_score = 0
            for prev_move in self.prev_moves:
                if abs(move - prev_move) < 2:
                    prev_score += 1
            # Also assign a score based on the number of potential winning lines
            winning_score = 0
            for line in self.winning_lines:
                if move in line:
                    winning_score += 1
            biased_moves.append((move, score + prev_score + winning_score))

        # Sort the biased moves by their score
        biased_moves.sort(key=lambda x: x[1], reverse=True)

        # Choose a random move from the biased list
        ix = random.randint(0, len(biased_moves) - 1)
        action = biased_moves[ix][0]

        # Store the chosen move as the opponent's previous move
        self.prev_moves.append(action)

        # Update the potential winning lines
        self.winning_lines = []
        for i in range(board_size):
            for j in range(board_size):
                if board_state[i][j] == 1:
                    # Check horizontal line
                    if j < board_size - 3:
                        line = [i, j, j + 1, j + 2, j + 3]
                        self.winning_lines.append(line)
                    # Check vertical line
                    if i < board_size - 3:
                        line = [i, j, i + 1, j, i + 2, j, i + 3, j]
                        self.winning_lines.append(line)
                    # Check diagonal line (top-left to bottom-right)
                    if i < board_size - 3 and j < board_size - 3:
                        line = [i, j, i + 1, j + 1, i + 2, j + 2, i + 3, j + 3]
                        self.winning_lines.append(line)
                    # Check diagonal line (bottom-left to top-right)
                    if i > 2 and j < board_size - 3:
                        line = [i, j, i - 1, j + 1, i - 2, j + 2, i - 3, j + 3]
                        self.winning_lines.append(line)

        return action


