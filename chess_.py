import gym
import chess_gym

env = gym.make("Chess-v0")
env.reset()

terminal = False

while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, info = env.step(action)
    env.render()

env.close()

