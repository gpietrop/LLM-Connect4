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

    def __init__(self, seed=0):
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
            print("Environment is None in RandomPolicy")   ### BUG
            return None
        possible_moves = self.env.possible_moves
        if len(possible_moves) == 0:
            return None
        ix = self.rnd.randint(0, len(possible_moves))
        action = possible_moves[ix]
        return action


class GreedyPolicy(object):
    """Greedy policy for Connect 4."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        # if self.env is None:
        #     print("Environment is None in RandomPolicy")   ### BUG
        #     return 1
        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # Evaluate each possible move by its immediate benefit.
        possible_moves = self.env.possible_moves
        best_score = -1
        best_move = None

        for move in possible_moves:
            new_env.reset()
            # print("obs", obs)
            new_env.set_board_state(obs)  # Assuming set_board_state sets up the board from the observation
            new_env.board_state.resize((6, 6))
            # print("new state", new_env.board_state.resize((6, 6)))
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            # Choose the move with the best immediate reward.
            if reward > best_score:
                best_score = reward
                best_move = move

        new_env.close()

        return best_move # if best_move is not None else np.random.choice(possible_moves)


class ImprovedGreedyPolicy(object):
    """Improved Greedy policy for Connect 4 with defensive and lookahead strategies."""

    def __init__(self):
        self.env = None
        self.lookahead_depth = 1  # Adjust this for deeper lookahead
        self.randomness_factor = 0.1  # 10% chance to make a random move

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        if self.env is None:
            return np.random.choice(self.env.possible_moves)

        if random.random() < self.randomness_factor:
            # Occasionally make a random move
            return np.random.choice(self.env.possible_moves)

        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # Evaluate each possible move
        possible_moves = self.env.possible_moves
        best_score = -float('inf')
        best_move = None

        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(obs)
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            if reward > best_score:
                best_score = reward
                best_move = move

            # Look-ahead for opponent's response
            opponent_perspective = 1 if my_perspective == 0 else 0
            new_env.set_player_turn(opponent_perspective)
            opponent_moves = new_env.possible_moves
            for opp_move in opponent_moves:
                new_env.reset()
                new_env.set_board_state(obs)
                new_env.set_player_turn(opponent_perspective)
                _, opp_reward, _, _ = new_env.step(opp_move)

                # If the opponent can win next turn, block that move
                if opp_reward > 0:
                    best_move = opp_move
                    best_score = 0  # Blocking move has priority

        new_env.close()

        return best_move if best_move is not None else np.random.choice(possible_moves)
