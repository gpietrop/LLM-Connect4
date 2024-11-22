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
    """Easy policy for Connect 4."""
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

        # Prioritize blocking the opponent's winning line
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            if self.check_win(temp_board, -player_turn):
                return move

        # Try to create a winning line
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            if self.check_win(temp_board, player_turn):
                return move

        # Play in the center column
        center_move = board_size // 2
        if center_move in possible_moves:
            return center_move

        # Play randomly
        ix = random.randint(0, len(possible_moves)-1)
        action = possible_moves[ix]
        return action

    def drop_piece(self, board, move, player):
        for row in reversed(board):
            if row[move] == 0:
                row[move] = player
                return board
        return board

    def check_win(self, board, player):
        # Check horizontal locations for win
        for row in board:
            for col in range(len(row) - 3):
                if row[col] == player and row[col+1] == player and row[col+2] == player and row[col+3] == player:
                    return True

        # Check vertical locations for win
        for col in range(len(board[0])):
            for row in range(len(board) - 3):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and board[row+3][col] == player:
                    return True

        # Check positively sloped diagonals
        for row in range(len(board) - 3):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and board[row+3][col+3] == player:
                    return True

        # Check negatively sloped diagonals
        for row in range(3, len(board)):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and board[row-3][col+3] == player:
                    return True

        return False


class HardPolicy():
    """Hard policy for Connect 4."""

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

        # Look ahead 2 moves to block opponent's win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            for next_move in self.get_possible_moves(temp_board):
                next_temp_board = temp_board.copy()
                next_temp_board = self.drop_piece(next_temp_board, next_move, player_turn)
                if self.check_win(next_temp_board, -player_turn):
                    return move

        # Look ahead 2 moves to create a win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            for next_move in self.get_possible_moves(temp_board):
                next_temp_board = temp_board.copy()
                next_temp_board = self.drop_piece(next_temp_board, next_move, -player_turn)
                if self.check_win(next_temp_board, player_turn):
                    return move

        # Block opponent's potential two-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            if self.check_two_way_win(temp_board, -player_turn):
                return move

        # Create a potential two-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            if self.check_two_way_win(temp_board, player_turn):
                return move

        # Block opponent's potential three-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            if self.check_three_way_win(temp_board, -player_turn):
                return move

        # Create a potential three-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            if self.check_three_way_win(temp_board, player_turn):
                return move

        # Play in the center column
        center_move = board_size // 2
        if center_move in possible_moves:
            return center_move

        # Play in a column that could potentially block an opponent's win
        for move in possible_moves:
            if self.check_column_block(board_state, move, -player_turn):
                return move

        # Play in a column that could potentially create a win
        for move in possible_moves:
            if self.check_column_win(board_state, move, player_turn):
                return move

        # Play randomly
        ix = random.randint(0, len(possible_moves)-1)
        action = possible_moves[ix]
        return action

    def drop_piece(self, board, move, player):
        for row in reversed(board):
            if row[move] == 0:
                row[move] = player
                return board
        return board

    def get_possible_moves(self, board):
        possible_moves = []
        for col in range(len(board[0])):
            if board[0][col] == 0:
                possible_moves.append(col)
        return possible_moves

    def check_win(self, board, player):
        # Check horizontal locations for win
        for row in board:
            for col in range(len(row) - 3):
                if row[col] == player and row[col+1] == player and row[col+2] == player and row[col+3] == player:
                    return True

        # Check vertical locations for win
        for col in range(len(board[0])):
            for row in range(len(board) - 3):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and board[row+3][col] == player:
                    return True

        # Check positively sloped diagonals
        for row in range(len(board) - 3):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and board[row+3][col+3] == player:
                    return True

        # Check negatively sloped diagonals
        for row in range(3, len(board)):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and board[row-3][col+3] == player:
                    return True

        return False

    def check_two_way_win(self, board, player):
        # Check horizontal locations for two-way win
        for row in board:
            for col in range(len(row) - 2):
                if row[col] == player and row[col+1] == player and row[col+2] == player and (col > 0 and row[col-1] == 0 or col < len(row) - 3 and row[col+3] == 0):
                    return True

        # Check vertical locations for two-way win
        for col in range(len(board[0])):
            for row in range(len(board) - 2):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and (row > 0 and board[row-1][col] == 0 or row < len(board) - 3 and board[row+3][col] == 0):
                    return True

        # Check positively sloped diagonals for two-way win
        for row in range(len(board) - 2):
            for col in range(len(board[0]) - 2):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and (row > 0 and col > 0 and board[row-1][col-1] == 0 or row < len(board) - 3 and col < len(board[0]) - 3 and board[row+3][col+3] == 0):
                    return True

        # Check negatively sloped diagonals for two-way win
        for row in range(2, len(board)):
            for col in range(len(board[0]) - 2):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and (row < len(board) - 1 and col > 0 and board[row+1][col-1] == 0 or row > 2 and col < len(board[0]) - 3 and board[row-3][col+3] == 0):
                    return True

        return False

    def check_three_way_win(self, board, player):
        # Check horizontal locations for three-way win
        for row in board:
            for col in range(len(row) - 1):
                if row[col] == player and row[col+1] == player and (col > 0 and row[col-1] == 0 and (col < len(row) - 2 and row[col+2] == 0 or col < len(row) - 3 and row[col+3] == 0)):
                    return True

        # Check vertical locations for three-way win
        for col in range(len(board[0])):
            for row in range(len(board) - 1):
                if board[row][col] == player and board[row+1][col] == player and (row > 0 and board[row-1][col] == 0 and (row < len(board) - 2 and board[row+2][col] == 0 or row < len(board) - 3 and board[row+3][col] == 0)):
                    return True

        # Check positively sloped diagonals for three-way win
        for row in range(len(board) - 1):
            for col in range(len(board[0]) - 1):
                if board[row][col] == player and board[row+1][col+1] == player and (row > 0 and col > 0 and board[row-1][col-1] == 0 and (row < len(board) - 2 and col < len(board[0]) - 2 and board[row+2][col+2] == 0 or row < len(board) - 3 and col < len(board[0]) - 3 and board[row+3][col+3] == 0)):
                    return True

        # Check negatively sloped diagonals for three-way win
        for row in range(1, len(board)):
            for col in range(len(board[0]) - 1):
                if board[row][col] == player and board[row-1][col+1] == player and (row < len(board) - 1 and col > 0 and board[row+1][col-1] == 0 and (row > 1 and col < len(board[0]) - 2 and board[row-2][col+2] == 0 or row > 2 and col < len(board[0]) - 3 and board[row-3][col+3] == 0)):
                    return True

        return False

    def check_column_block(self, board, col, player):
        for row in reversed(board):
            if row[col] == player:
                return True
            elif row[col] != 0:
                return False
        return False

    def check_column_win(self, board, col, player):
        for row in reversed(board):
            if row[col] == player:
                return True
            elif row[col] == 0:
                return False
        return False


class MediumPolicy():
    """Medium policy for Connect 4."""

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

        # Prioritize blocking the opponent's winning line
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            if self.check_win(temp_board, -player_turn):
                return move

        # Try to create a winning line
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            if self.check_win(temp_board, player_turn):
                return move

        # Block opponent's potential two-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, -player_turn)
            if self.check_two_way_win(temp_board, -player_turn):
                return move

        # Create a potential two-way win
        for move in possible_moves:
            temp_board = board_state.copy()
            temp_board = self.drop_piece(temp_board, move, player_turn)
            if self.check_two_way_win(temp_board, player_turn):
                return move

        # Play in the center column
        center_move = board_size // 2
        if center_move in possible_moves:
            return center_move

        # Play in a column that could potentially block an opponent's win
        for move in possible_moves:
            if self.check_column_block(board_state, move, -player_turn):
                return move

        # Play randomly
        ix = random.randint(0, len(possible_moves)-1)
        action = possible_moves[ix]
        return action

    def drop_piece(self, board, move, player):
        for row in reversed(board):
            if row[move] == 0:
                row[move] = player
                return board
        return board

    def check_win(self, board, player):
        # Check horizontal locations for win
        for row in board:
            for col in range(len(row) - 3):
                if row[col] == player and row[col+1] == player and row[col+2] == player and row[col+3] == player:
                    return True

        # Check vertical locations for win
        for col in range(len(board[0])):
            for row in range(len(board) - 3):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and board[row+3][col] == player:
                    return True

        # Check positively sloped diagonals
        for row in range(len(board) - 3):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and board[row+3][col+3] == player:
                    return True

        # Check negatively sloped diagonals
        for row in range(3, len(board)):
            for col in range(len(board[0]) - 3):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and board[row-3][col+3] == player:
                    return True

        return False

    def check_two_way_win(self, board, player):
        # Check horizontal locations for two-way win
        for row in board:
            for col in range(len(row) - 2):
                if row[col] == player and row[col+1] == player and row[col+2] == player and (col > 0 and row[col-1] == 0 or col < len(row) - 3 and row[col+3] == 0):
                    return True

        # Check vertical locations for two-way win
        for col in range(len(board[0])):
            for row in range(len(board) - 2):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and (row > 0 and board[row-1][col] == 0 or row < len(board) - 3 and board[row+3][col] == 0):
                    return True

        # Check positively sloped diagonals for two-way win
        for row in range(len(board) - 2):
            for col in range(len(board[0]) - 2):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and (row > 0 and col > 0 and board[row-1][col-1] == 0 or row < len(board) - 3 and col < len(board[0]) - 3 and board[row+3][col+3] == 0):
                    return True

        # Check negatively sloped diagonals for two-way win
        for row in range(2, len(board)):
            for col in range(len(board[0]) - 2):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and (row < len(board) - 1 and col > 0 and board[row+1][col-1] == 0 or row > 2 and col < len(board[0]) - 3 and board[row-3][col+3] == 0):
                    return True

        return False

    def check_column_block(self, board, col, player):
        for row in reversed(board):
            if row[col] == player:
                return True
            elif row[col] != 0:
                return False
        return False
