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


class RandomPolicy(object):
    """Random policy for Connect 4."""

    def __init__(self, seed=20):
        self.rnd = np.random.RandomState(seed=seed)
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def seed(self, seed):
        self.rnd = np.random.RandomState(seed=seed)

    def get_action(self, obs):
        if self.env is None:
            print("Environment is None in RandomPolicy")  ### BUG
            return None
        possible_moves = self.env.possible_moves
        if len(possible_moves) == 0:
            return None
        ix = self.rnd.randint(0, len(possible_moves))
        action = possible_moves[ix]
        return action


class MinimaxPolicy(object):
    """Minimax policy for Connect 4."""

    def __init__(self, max_depth=10):
        self.env = None
        self.max_depth = max_depth

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def minimax(self, env, depth, maximizing_player, obs, done):
        if depth == 0 or done:
            return self.evaluate_state(env), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in env.possible_moves:
                new_env = copy_env(env)
                new_env.set_board_state(obs)
                new_env.board_state.resize((6, 6))
                obs, _, done, _ = new_env.step(move)
                my_eval, _ = self.minimax(new_env, depth - 1, False, obs, done)
                if my_eval > max_eval:
                    max_eval = my_eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in env.possible_moves:
                new_env = copy_env(env)
                new_env.set_board_state(obs)
                new_env.board_state.resize((6, 6))
                obs, _, done, _ = new_env.step(move)
                my_eval, _ = self.minimax(new_env, depth - 1, True, obs, done)
                if my_eval < min_eval:
                    min_eval = my_eval
                    best_move = move
            return min_eval, best_move

    def evaluate_state(self, env):
        env.board_state.resize((6, 6))
        board = env.board_state
        # Constants for scores
        SCORE_4 = 1000
        SCORE_3 = 100
        SCORE_2 = 10
        SCORE_1 = 1

        def count_score(n, player):
            if n == 4:
                return SCORE_4 if player == -1 else -SCORE_4
            if n == 3:
                return SCORE_3 if player == -1 else -SCORE_3
            elif n == 2:
                return SCORE_2 if player == -1 else -SCORE_2
            elif n == 1:
                return SCORE_1 if player == -1 else -SCORE_1
            return 0

        def evaluate_direction(x, y, dx, dy):
            player = board[x][y]
            if player == 0:
                return 0

            score = 0
            count = 0

            for i in range(4):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < 6 and 0 <= ny < 6:
                    if board[nx][ny] == player:
                        count += 1
                    else:
                        break
                else:
                    break

            score += count_score(count, player)
            return score

        total_score = 0

        # Check all directions from each cell
        for x in range(6):
            for y in range(6):
                total_score += evaluate_direction(x, y, 1, 0)  # Horizontal
                total_score += evaluate_direction(x, y, 0, 1)  # Vertical
                total_score += evaluate_direction(x, y, 1, 1)  # Diagonal (down-right)
                total_score += evaluate_direction(x, y, -1, 1)  # Diagonal (up-right)

        return total_score

    def get_action(self, obs):
        self.env.set_board_state(obs)
        self.env.board_state.resize((6, 6))
        _, move = self.minimax(self.env, self.max_depth, True, obs, False)
        return move