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
    """Policy for Connect 4 that helps a player learn good strategies."""

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
        board_size = self.env.board_size
        player_turn = self.env.player_turn

        # Prioritize playing in the middle column
        middle_column = board_size // 2
        if middle_column in possible_moves:
            return middle_column

        # Then prioritize playing in the column with the most empty slots
        max_empty_slots = 0
        best_column = None
        for column in possible_moves:
            empty_slots = sum(board_state[:, column] == 0)
            if empty_slots > max_empty_slots:
                max_empty_slots = empty_slots
                best_column = column

        if best_column is not None:
            return best_column

        # Finally, play in a random column
        return random.choice(possible_moves)


class MediumPolicy(object):
    """Policy for Connect 4 that helps a player learn good strategies."""

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
        board_size = self.env.board_size
        player_turn = self.env.player_turn

        # Prioritize playing in the middle column
        middle_column = board_size // 2
        if middle_column in possible_moves:
            return middle_column

        # Then prioritize playing in the column with the most empty slots
        max_empty_slots = 0
        best_column = None
        for column in possible_moves:
            empty_slots = sum(board_state[:, column] == 0)
            if empty_slots > max_empty_slots:
                max_empty_slots = empty_slots
                best_column = column

        if best_column is not None:
            return best_column

        # Next, prioritize playing in a column that will create a "fork"
        for column in possible_moves:
            # Check if playing in this column will create a fork
            if self._will_create_fork(board_state, column, player_turn, board_size):
                return column

        # Finally, play in a random column
        return random.choice(possible_moves)

    def _will_create_fork(self, board_state, column, player_turn, board_size):
        # Check if playing in this column will create a fork
        # This is a simplified implementation and may not cover all cases
        for row in range(board_size):
            if board_state[row, column] == 0:
                # Check if playing in this column will create a fork
                # in the current row
                if self._is_fork(board_state, row, column, player_turn, board_size):
                    return True

                # Check if playing in this column will create a fork
                # in the next row
                if row < board_size - 1 and self._is_fork(board_state, row + 1, column, player_turn, board_size):
                    return True

        return False

    def _is_fork(self, board_state, row, column, player_turn, board_size):
        # Check if playing in this column will create a fork
        # in the current row
        if board_state[row, column] == 0:
            # Check if there are two possible winning lines
            # in the current row
            winning_lines = self._get_winning_lines(board_state, row, column, player_turn, board_size)
            return len(winning_lines) > 1

        return False

    def _get_winning_lines(self, board_state, row, column, player_turn, board_size):
        # Get all possible winning lines in the current row
        winning_lines = []
        for i in range(board_size):
            if board_state[row, i] == player_turn:
                winning_lines.append(i)
        return winning_lines


class HardPolicy:
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
        board_size = self.env.board_size
        player_turn = self.env.player_turn

        best_score = -float('inf')
        best_move = None

        for move in possible_moves:
            new_board_state = self.env.board_state.copy()
            new_board_state[move, 0] = player_turn
            new_board_state = self._update_board_state(new_board_state, move, board_size)

            score = self._minimax(new_board_state, -player_turn, board_size, 3, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, board_state, player_turn, board_size, depth, alpha, beta):
        if self._is_game_over(board_state, board_size):
            return self._evaluate_board_state(board_state, board_size)

        if depth == 0:
            return self._evaluate_board_state(board_state, board_size)

        if player_turn == 1:
            best_score = -float('inf')
            for move in self._get_possible_moves(board_state, board_size):
                new_board_state = self._update_board_state(board_state, move, board_size)
                score = self._minimax(new_board_state, -player_turn, board_size, depth - 1, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in self._get_possible_moves(board_state, board_size):
                new_board_state = self._update_board_state(board_state, move, board_size)
                score = self._minimax(new_board_state, -player_turn, board_size, depth - 1, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def _is_game_over(self, board_state, board_size):
        for i in range(board_size):
            if board_state[i, 0] == 1:
                return True
            if board_state[i, board_size - 1] == -1:
                return True
        return False

    def _evaluate_board_state(self, board_state, board_size):
        score = 0
        for i in range(board_size):
            if board_state[i, 0] == 1:
                score += 1
            if board_state[i, board_size - 1] == -1:
                score -= 1
        return score

    def _get_possible_moves(self, board_state, board_size):
        possible_moves = []
        for i in range(board_size):
            if board_state[i, 0] == 0:
                possible_moves.append(i)
        return possible_moves

    def _update_board_state(self, board_state, move, board_size):
        new_board_state = board_state.copy()
        new_board_state[move, 0] = 1
        return new_board_state
