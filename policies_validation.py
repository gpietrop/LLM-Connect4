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
    def __init__(self, depth=1):
        self.depth = depth
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        best_score = -float('inf')
        best_action = None
        for action in self.env.possible_moves:
            # Apply the action to get the new state
            new_env = copy_env(self.env)
            new_env.step(action)
            # Call the minimax function
            score = self.minimax(new_env, self.depth, False)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def minimax(self, env, depth, is_maximizing_player):
        # print(env.terminated)
        if depth == 0 or env.terminated:
            return env.calculate_reward(-1)[0]

        if is_maximizing_player:
            best_score = -float('inf')
            for action in env.possible_moves:
                new_env = copy_env(env)
                new_env.step(action)
                score = self.minimax(new_env, depth - 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for action in env.possible_moves:
                new_env = copy_env(env)
                new_env.step(action)
                score = self.minimax(new_env, depth - 1, True)
                best_score = min(best_score, score)
            return best_score
