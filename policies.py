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
        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # Evaluate each possible move by its immediate benefit.
        possible_moves = self.env.possible_moves
        best_score = -100
        best_moves = []

        for move in possible_moves:
            new_env.reset()
            # print("obs", obs)
            new_env.set_board_state(obs)  # Assuming set_board_state sets up the board from the observation
            new_env.board_state.resize((6, 6))
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            # Choose the move with the best immediate reward.

            if reward[0] > best_score:
                best_score = reward[0]
                # print(best_score)
                best_moves = [move]
            elif reward[0] == best_score:
                best_moves.append(move)

        new_env.close()

        return random.choice(best_moves)  # if best_move is not None else np.random.choice(possible_moves)


class IntermediateGreedyPolicy(object):
    """Intermediate policy for Connect 4 with basic lookahead and defensive strategies."""

    def __init__(self):
        self.env = None
        self.randomness_factor = 0.05  # 5% chance to make a random move
        self.blocking_chance = 0.5

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):

        if random.random() < self.randomness_factor:
            return np.random.choice(self.env.possible_moves)

        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        possible_moves = self.env.possible_moves
        best_score = -100
        best_move = None

        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(obs)
            new_env.board_state.resize((6, 6))  # Assuming a 6x6 board
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            # Simple defensive check
            opponent_perspective = RED_DISK
            # Switches to the opponent's perspective and simulates the same move to evaluate its consequences.
            new_env.reset()
            new_env.set_board_state(obs)
            new_env.board_state.resize((6, 6))
            new_env.set_player_turn(opponent_perspective)

            _, opp_reward, _, _ = new_env.step(move)

            # If the opponent can win with this move, immediately return this move to block the opponent.
            # print(opp_reward)
            if opp_reward[0] > 0.6 and random.random() < self.blocking_chance:  # mean the opposite win that turn
                # return opp_move
                best_score = 100
                best_move = opp_move

            # if opp_reward[1] == 1 and random.random() < self.blocking_chance:
            #    return move  # Prioritize blocking over other strategies

            # Evaluate for best move
            if reward[0] > best_score:
                best_score = reward[0]
                best_move = move  # return move

        new_env.close()
        return best_move if best_move is not None else np.random.choice(possible_moves)


class ImprovedGreedyPolicy(object):
    """Improved Greedy policy for Connect 4 with defensive and lookahead strategies."""

    def __init__(self):
        self.env = None
        self.lookahead_depth = 1  # Adjust this for deeper lookahead
        self.randomness_factor = 0.0  # 10% chance to make a random move

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):

        if random.random() < self.randomness_factor:
            # Occasionally make a random move
            return np.random.choice(self.env.possible_moves)

        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # Evaluate each possible move
        possible_moves = self.env.possible_moves
        best_score = -100  # float('inf')
        best_move = None

        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(obs)
            new_env.board_state.resize((6, 6))
            new_env.set_player_turn(my_perspective)
            _, reward, _, _ = new_env.step(move)

            if reward[0] > best_score:
                best_score = reward[0]
                best_move = move

            # Look-ahead for opponent's response

            opponent_perspective = RED_DISK
            new_env.set_player_turn(opponent_perspective)
            opponent_moves = new_env.possible_moves
            for opp_move in opponent_moves:
                new_env.reset()
                new_env.set_board_state(obs)
                new_env.board_state.resize((6, 6))
                new_env.set_player_turn(opponent_perspective)
                # print("debug")
                _, opp_reward, _, _ = new_env.step(opp_move)

                # If the opponent can win next turn, block that move
                # print(opp_reward)
                if opp_reward[0] > 0.6:  # mean the opposite win that turn
                    # return opp_move
                    best_score = 100
                    best_move = opp_move
                    # Blocking move has priority  FORSE QUA METTO RETURN MOVE??? PRIMA ERA 5 E CE LA FCEVA

        new_env.close()

        return best_move if best_move is not None else np.random.choice(possible_moves)
