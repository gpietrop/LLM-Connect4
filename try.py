""""
Consider a RL environment from the gym library https://www.gymlibrary.dev/index.html
1) there is all the documentation to get access to the environments
2) Script python written by Steven
3) Once that the script is build, there are A LOT of different dataset which can be used
"""

import gym

""""
We must choose a problem considering the following requirements:  
+ Development time -> same for all the dataset (?) as the pipeline is the same
- Execution time
- Game complexity
- Controller complexity
- Training difficulty of LLM
+ Available benchmarks -> automatically solved 

I have considered two possibilities: 
1) classic RL problems
  - lower "game" complexity -> too easy? 
  - lower controller complexity 
  - how can an LLM provide more difficult domain? not trivial  
2) ATARI games 
  - game complexity which can vary 
  - need to find an environment where we can vary the difficulties
"""

"""
1) Mountain Car
and how to manually change the starting point
"""

import gym

# Create the Mountain Car environment
env = gym.make('MountainCar-v0')

# Reset the environment to get the initial state
initial_state = env.reset()

# Manually set the starting state
# For instance, setting the position to -0.5 and the velocity to 0
desired_state = [-0.5, 1]
env.env.state = desired_state
