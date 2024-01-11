import gym
import gym_connect4
import random
import warnings

warnings.filterwarnings('ignore', module='gym')

env = gym.make('Connect4-v0')  # Initialize the environment
env.reset()  # Reset the environment to get it into a clean state

env.board[0, 0] = 0  # Customize the board as you prefer
env.render()
agents = ['Agent1()', 'Agent2()']
game_over = False

while not game_over:
    action_dict = {}
    for agent_id, agent in enumerate(agents):
        valid_actions = [c for c in range(env.board.shape[1]) if env.board[c, -1] == -1]   # Determine the valid actions (columns not full)
        if valid_actions:  # Choose a valid action
            action = random.choice(valid_actions)
        else:
            break  # No valid actions available

        action_dict[agent_id] = action
        print(type(action))

        obses, rewards, game_over, info = env.step(action)
        env.render()

