import gym
import numpy as np

# Assuming policies.py and your Connect4Env class are in the same directory
from policies import GreedyPolicy
from c4_gym import Connect4Env, RED_DISK, YELLOW_DISK


def render_board(board_state):
    # Adjusting the function to handle 2D board state
    if len(board_state.shape) == 1:  # If the board state is flattened
        board_size = int(np.sqrt(len(board_state)))  # Assuming a square board
        board_state = board_state.reshape((board_size, board_size))

    symbols = {0: '.', 1: 'R', -1: 'Y'}
    for row in board_state:
        print(' '.join(symbols[cell] for cell in row))
    print('-' * (len(board_state[0]) * 2 - 1))


def get_human_move(possible_moves):
    move = None
    while move not in possible_moves:
        try:
            move = int(input("Enter your move (0-5): "))
            if move not in possible_moves:
                print(f"Invalid move. Possible moves: {possible_moves}")
        except ValueError:
            print("Please enter a number.")
    return move


def main():
    # Initialize the environment
    env = Connect4Env(yellow_policy=GreedyPolicy ())  # Or any other policy you wish to play against
    obs = env.reset()
    done = False

    while not done:
        render_board(obs)

        # Player's turn
        if env.player_turn == RED_DISK:
            move = get_human_move(env.possible_moves)
        else:
            # Environment's turn
            move = env.opponent.get_action(obs)

        obs, reward, done, _ = env.step(move)

    render_board(obs)
    if env.env.winner == RED_DISK:
        print("Congratulations, you won!")
    elif env.env.winner == YELLOW_DISK:
        print("You lost. Better luck next time!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
