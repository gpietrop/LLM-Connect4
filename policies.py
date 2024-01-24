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
                best_move = move

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

