import gym
from gym import spaces
import numpy as np

from policies import RandomPolicy, GreedyPolicy

# Constants for Connect 4 on a 6x6 board
NO_DISK = 0
RED_DISK = 1
YELLOW_DISK = -1


class Connect4Env(gym.Env):
    """Wrapper for Connect4BaseEnv."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self,
                 red_policy=None,
                 yellow_policy=None,  # Try GreedyPolicy for Connect 4
                 protagonist=RED_DISK,
                 board_size=6,  # Set board size to 6x6
                 initial_rand_steps=0,
                 seed=0,
                 sudden_death_on_invalid_move=True,
                 render_in_step=False,
                 num_disk_as_reward=False,
                 possible_actions_in_obs=False):

        # Create the inner environment.
        self.board_size = board_size
        self.num_disk_as_reward = num_disk_as_reward
        self.env = Connect4BaseEnv(
            num_disk_as_reward=self.num_disk_as_reward,
            sudden_death_on_invalid_move=sudden_death_on_invalid_move,
            possible_actions_in_obs=possible_actions_in_obs,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_in_step = render_in_step
        self.initial_rand_steps = initial_rand_steps
        self.rand_seed = seed
        self.rnd = np.random.RandomState(seed=self.rand_seed)
        self.max_rand_steps = 0
        self.rand_step_cnt = 0

        # Initialize policies.
        self.protagonist = protagonist
        if self.protagonist == RED_DISK:
            self.opponent = yellow_policy
        else:
            self.opponent = red_policy

    def seed(self, seed=None):
        self.rand_seed = seed if seed is not None else self.rand_seed
        self.rnd = np.random.RandomState(seed=self.rand_seed)
        if self.opponent is not None and hasattr(self.opponent, 'seed'):
            self.opponent.seed(self.rand_seed)

    def reset(self):
        obs = self.env.reset()
        self.max_rand_steps = self.rnd.randint(0, self.initial_rand_steps // 2 + 1) * 2
        self.rand_step_cnt = 0

        # This provides the opponent a chance to get env.possible_moves.
        if hasattr(self.opponent, 'reset'):
            try:
                self.opponent.reset(self)
            except TypeError:
                pass

        if self.env.player_turn == self.protagonist:
            return obs
        else:
            action = self.opponent.get_action(obs)
            obs, _, done, _ = self.env.step(action)
            if done:
                return self.reset()
            else:
                return obs

    def set_opponent_policy(self, new_policy):
        self.opponent = new_policy
        # If the new policy has a 'seed' method, initialize it with the current seed.
        if hasattr(self.opponent, 'seed'):
            self.opponent.seed(self.rand_seed)
        # If the new policy has a 'reset' method, call it to initialize.
        if hasattr(self.opponent, 'reset'):
            self.opponent.reset(self)

    def step(self, action):
        assert self.env.player_turn == self.protagonist

        if self.rand_step_cnt < self.max_rand_steps:
            action = self.rnd.choice(self.env.possible_moves)
            self.rand_step_cnt += 1

        obs, reward, done, _ = self.env.step(action)

        if self.render_in_step:
            self.render()

        if done or self.env.player_turn == self.protagonist:
            return obs, reward, done, None

        while not done and self.env.player_turn != self.protagonist:
            opponent_move = self.opponent.get_action(obs)
            obs, _, done, _ = self.env.step(opponent_move)
            if self.render_in_step:
                self.render()

        return obs, reward, done, None

    def render(self, mode='human', close=False):
        if close:
            return
        self.env.render(mode=mode, close=close)

    def close(self):
        self.env.close()

    @property
    def player_turn(self):
        return self.env.player_turn

    @property
    def possible_moves(self):
        return self.env.possible_moves


class Connect4BaseEnv(gym.Env):
    """Connect 4 base environment on a 6x6 board."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self, sudden_death_on_invalid_move=True, num_disk_as_reward=False, possible_actions_in_obs=False,
                 mute=True, board_size=6):
        self.board_size = board_size
        self.sudden_death_on_invalid_move = sudden_death_on_invalid_move
        self.num_disk_as_reward = num_disk_as_reward
        self.mute = mute
        self.possible_actions_in_obs = possible_actions_in_obs

        self.board_state = self._reset_board()
        self.viewer = None

        self.player_turn = RED_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = []

        self.action_space = spaces.Discrete(self.board_size)
        self.observation_space = spaces.Box(
            low=NO_DISK, high=YELLOW_DISK, shape=(self.board_size, self.board_size), dtype=np.int8)

    def _reset_board(self):
        return np.zeros([self.board_size] * 2, dtype=int)

    def reset(self):
        self.board_state = self._reset_board()
        self.player_turn = RED_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = self._get_possible_actions()
        # obs = self.board_state.flatten()
        return self.get_observation()  # obs

    def _get_possible_actions(self):
        # print(self.board_state)
        return [col for col in range(self.board_size) if self.board_state[0][col] == NO_DISK]

    def _is_valid_move(self, action):
        return self.board_state[0][action] == NO_DISK

    def _drop_disk(self, column):
        row = self.board_size - 1
        while row >= 0 and self.board_state[row][column] != NO_DISK:
            row -= 1
        if row >= 0:
            self.board_state[row][column] = self.player_turn
            return True
        return False

    def _check_winner(self):
        # Check rows, columns, and diagonals for a win
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board_state[row][col] != NO_DISK:
                    # Check horizontal
                    if col <= self.board_size - 4 and all(
                            self.board_state[row][col + i] == self.board_state[row][col] for i in range(4)):
                        return self.board_state[row][col]
                    # Check vertical
                    if row <= self.board_size - 4 and all(
                            self.board_state[row + i][col] == self.board_state[row][col] for i in range(4)):
                        return self.board_state[row][col]
                    # Check diagonal (down-right)
                    if row <= self.board_size - 4 and col <= self.board_size - 4 and all(
                            self.board_state[row + i][col + i] == self.board_state[row][col] for i in range(4)):
                        return self.board_state[row][col]
                    # Check diagonal (up-right)
                    if row >= 3 and col <= self.board_size - 4 and all(
                            self.board_state[row - i][col + i] == self.board_state[row][col] for i in range(4)):
                        return self.board_state[row][col]
        return NO_DISK

    def set_board_state(self, board_state):
        """Sets the board state for Connect 4 game."""

        # Check if the board state has more than 2 dimensions.
        if np.ndim(board_state) > 2:
            raise ValueError("Board state dimension is higher than expected")

        # Set the new board state
        self.board_state = np.array(board_state)

    def set_player_turn(self, turn):
        """Sets the player's turn for the Connect 4 game."""

        if turn not in [RED_DISK, YELLOW_DISK]:
            raise ValueError("Invalid player turn. Must be RED_DISK or YELLOW_DISK.")

        self.player_turn = turn

        # Update possible moves for the new player turn
        self.possible_moves = self._get_possible_actions()

    def get_observation(self):
        """Returns the current state of the Connect 4 board."""

        if self.possible_actions_in_obs:
            # Include a grid of possible moves (1 if the move is possible in that column, 0 otherwise)
            grid_of_possible_moves = np.zeros(self.board_size, dtype=bool)
            for move in self.possible_moves:
                grid_of_possible_moves[move] = True
            return np.array([self.board_state, grid_of_possible_moves])
        else:
            return np.array(self.board_state)

    def _is_progressing_towards_win_3(self, player):
        # Define the length of the line that we consider as progress (e.g., 2 or 3)
        progress_line_length = 3

        # Check horizontally
        for row in range(self.board_size):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row, col + i] == player for i in range(progress_line_length)):
                    return True

        # Check vertically
        for col in range(self.board_size):
            for row in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row + i, col] == player for i in range(progress_line_length)):
                    return True

        # Check diagonally (downward)
        for row in range(self.board_size - progress_line_length + 1):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row + i, col + i] == player for i in range(progress_line_length)):
                    return True

        # Check diagonally (upward)
        for row in range(progress_line_length - 1, self.board_size):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row - i, col + i] == player for i in range(progress_line_length)):
                    return True

        return False

    def _is_progressing_towards_win_2(self, player):
        # Define the length of the line that we consider as progress (e.g., 2 or 3)
        progress_line_length = 2

        # Check horizontally
        for row in range(self.board_size):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row, col + i] == player for i in range(progress_line_length)):
                    return True

        # Check vertically
        for col in range(self.board_size):
            for row in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row + i, col] == player for i in range(progress_line_length)):
                    return True

        # Check diagonally (downward)
        for row in range(self.board_size - progress_line_length + 1):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row + i, col + i] == player for i in range(progress_line_length)):
                    return True

        # Check diagonally (upward)
        for row in range(progress_line_length - 1, self.board_size):
            for col in range(self.board_size - progress_line_length + 1):
                if all(self.board_state[row - i, col + i] == player for i in range(progress_line_length)):
                    return True

        return False

    def calculate_reward(self, winner):
        reward = 0
        is_winner = 0
        if winner == -self.player_turn:
            reward = 1  # Won the game
            is_winner = 1
        elif winner == NO_DISK:
            if self._is_progressing_towards_win_2(self.player_turn):  # it was self.player_turn
                if self._is_progressing_towards_win_3(self.player_turn):  # it was self.player_turn
                    reward = 0.6  # Making progress towards win
                else:
                    reward = 0.3
            if self._is_progressing_towards_win_3(-self.player_turn):  # it was -self.player_turn
                # print(self.player_turn)
                reward = -0.3  # Opponent is making progress
        else:
            # print(f"we are {self.player_turn} and we lose")
            reward = 0  # Lost the game
        return reward, is_winner

    def calculate_simple_reward(self, winner):
        # it was: reward = 1 if winner == self.player_turn else 0
        if winner == self.player_turn:
            reward = 1  # we win
        else:
            reward = 0  # no winner
        return reward

    def step(self, action):
        if self.terminated:
            raise ValueError('Game has terminated!')
        # fare attenzione qua, si finiscono pochi giochi
        if not self._is_valid_move(action):
            if self.sudden_death_on_invalid_move:
                self.terminated = True
                self.winner = -self.player_turn
                return self.board_state, (-1, 0), True, {}
            else:
                return self.board_state, (-1, 0), False, {}

        self._drop_disk(action)
        winner = self._check_winner()
        # print(winner)

        if winner != NO_DISK:
            self.terminated = True
            self.winner = winner

        self.player_turn = YELLOW_DISK if self.player_turn == RED_DISK else RED_DISK
        self.possible_moves = self._get_possible_actions()

        reward, is_winner = self.calculate_reward(winner)
        total_reward = reward, is_winner

        return self.board_state.flatten(), total_reward, self.terminated, {}

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            for row in self.board_state:
                print(' '.join(['.' if x == NO_DISK else 'R' if x == RED_DISK else 'Y' for x in row]))
            print('-' * (self.board_size * 2 - 1))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
