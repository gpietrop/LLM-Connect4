import numpy as np
from lgp.cgpax.encoding import genome_to_lgp_program
from lgp.c4_gym import Connect4Env  # Replace with your actual import


def load_best_genome(filename='best_genome2.npy'):
    return np.load(filename)


def is_valid_move(self, column):
    # A move is valid if the top row of the chosen column is empty (0)
    return self.board[0][column] == 0


def play_against_best(config):
    env = Connect4Env()
    best_genome = load_best_genome()
    best_strategy = genome_to_lgp_program(best_genome, config)  # Convert genome to strategy

    register_array = np.zeros(config["n_registers"])

    while True:
        env.render()  # Display the board (implement this method in your Connect4Env)

        # Player's move
        valid_move = False
        while not valid_move:
            try:
                action = int(input("Your move (0-6): "))  # Get player input
                if 0 <= action <= 6:
                    state, reward, done, truncated, info = env.step(action)  # Make player's move
                    valid_move = True
                else:
                    print("Invalid move. Please enter a number between 0 and 6.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        if done:
            env.render()
            print("Congratulations, you won!" if info.get('winner') == 1 else "You lost.")
            break

        # AI's move
        ai_output = best_strategy(state, register_array)
        print(ai_output)
        ai_action = ai_output[1]  # Extract the action from the output
        print(f"AI action: {ai_action}")

        ai_action = np.argmax(ai_action)  # Convert array to a single action
        print(f"Processed AI action: {ai_action}")

        if 0 <= ai_action <= 6:
            state, reward, done, truncated, info = env.step(ai_action)
        else:
            print("AI generated an invalid move.")

    if done:
        env.render()
        print("AI wins!" if info.get('winner') == 2 else "It's a draw!")


if __name__ == '__main__':
    # Load or define the config used during the training of the genetic program
    config = {
        "n_rows": 20,
        "n_extra_registers": 5,
        "seed": 0,
        "n_constants": 10,
        "n_in": 42,
        "n_out": 1,
        "n_registers": 42,
        "n_individuals": 50,
        "solver": "lgp",
        "p_mut_lhs": 0.3,
        "p_mut_rhs": 0.1,
        "p_mut_functions": 0.1,
        "n_generations": 10,
        "selection": {
            "elite_size": 1,
            "type": "tournament",
            "tour_size": 2
        },
        "survival": "truncation",
        "crossover": False,
    }
    play_against_best(config)
