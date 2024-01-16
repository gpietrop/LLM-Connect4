"""Simple policies for Connect 4."""

import numpy as np

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
            new_env.set_board_state(obs)  # Assuming set_board_state sets up the board from the observation
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            # Choose the move with the best immediate reward.
            if reward > best_score:
                best_score = reward
                best_move = move

        new_env.close()
        return best_move # if best_move is not None else np.random.choice(possible_moves)

