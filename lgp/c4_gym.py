import numpy as np
import gymnasium
from gym_connect4.envs import Connect4Env
from gymnasium.core import ObsType, ActType


class Connect4Env(gymnasium.Env):
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        # self.board[-1, 0] = 2
        # self.board[-1, 1] = 2
        # self.board[-1, 2] = 2
        # print(self.board)
        self.action_size = 7  # 7 columns in Connect 4
        self.observation_size = 6 * 7  # Board size (6 rows x 7 columns)
        self.current_player = 1  # Start with player 1

    @classmethod
    def make(cls) -> Connect4Env:

        return cls()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Make the move
        if self.is_valid_move(action):
            # print("Valid move, updating board...")
            for row in range(5, -1, -1):  # Start from the bottom row
                if self.board[row][action] == 0:
                    self.board[row][action] = self.current_player
                    break

            # Check for a win or draw
            if self.check_win(self.current_player):
                reward = 1
                terminated = True
                info['winner'] = self.current_player
            elif self.is_draw():
                terminated = True
                info['winner'] = None

            # Debug print to show game status
            # print(f"Game terminated: {terminated}, Winner: {info.get('winner')}")

            # Switch player
            self.current_player = 3 - self.current_player
        else:
            # print("Invalid move made.")  # Debug print for invalid move
            reward = -1  # Invalid move penalty
            terminated = True

        return self.board.flatten(), reward, terminated, truncated, info

    def reset(self) -> tuple[ObsType, dict]:
        self.board = np.zeros((6, 7), dtype=int)
        # self.board[-1, 0] = 2
        # self.board[-1, 1] = 2
        # self.board[-1, 2] = 2
        self.current_player = 1
        return self.board.flatten(), {}

    def is_valid_move(self, column):
        return self.board[0][column] == 0

    def check_win(self, player):
        # Check horizontal lines
        for row in range(6):
            for col in range(4):
                if all(self.board[row, col:col + 4] == player):
                    return True

        # Check vertical lines
        for col in range(7):
            for row in range(3):
                if all(self.board[row:row + 4, col] == player):
                    return True

        # Check diagonal (down-right) and (down-left)
        for row in range(3):
            for col in range(4):
                if all([self.board[row + i][col + i] == player for i in range(4)]) or \
                        all([self.board[row + i][col + 3 - i] == player for i in range(4)]):
                    return True

        return False

    def is_draw(self):
        return all(self.board[0] != 0)

    def render(self):
        # Optional: Implement a method to visualize the game board
        pass

    def close(self):
        # Optional: Implement any cleanup necessary
        pass

    # Additional methods as needed for game logic
