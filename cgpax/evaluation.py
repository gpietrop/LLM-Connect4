from typing import Callable, Dict
import gym
import copy
import numpy as np
import jax.numpy as jnp
from cgpax.encoding import genome_to_lgp_program

from policies import GreedyPolicy


def copy_env(env, mute_env=True):
    new_env = env.__class__(
        board_size=env.board_size,
        sudden_death_on_invalid_move=env.sudden_death_on_invalid_move,
        mute=mute_env)
    new_env.reset()
    return new_env


def _evaluate_program(program: Callable, program_state_size: int, env: gym.Env,
                      episode_length: int = 42) -> Dict:
    obs = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    dead_time = episode_length
    final_percentage = 0.
    for i in range(episode_length):
        inputs = jnp.asarray(obs.copy().flatten())

        new_program_state, actions = program(inputs, program_state)

        selected_action = jnp.argmax(actions) % 6

        # Iterate through the actions to find a suitable one
        sorted_indices = jnp.argsort(actions)[::-1]  # This sorts in descending order
        for index in sorted_indices:
            test_env = copy_env(env.env)
            test_env.set_board_state(obs)
            test_env.board_state.resize((6, 6))

            # Test the action in the copied environment
            _, rew, _, _ = test_env.step(index % 6)

            if rew[0] >= -5:
                selected_action = index % 6
                break

        obs, total_reward, done, info = env.step(selected_action)  # Make sure action is an integer

        final_percentage = total_reward[1]
        cumulative_reward += total_reward[0]

        if done:
            # print(obs.reshape((6, 6)))
            break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "dead_time": dead_time,
        "final_percentage": final_percentage
    }


def evaluate_lgp_genome(genome: jnp.ndarray,
                        config: Dict,
                        env: gym.Env,
                        episode_length: int = 42,
                        new_policy=None,
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    env.set_opponent_policy(new_policy=new_policy)

    total_reward = []
    min_final_percentage = 1.

    for _ in range(1):
        performances = inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env,
                                       episode_length)

        total_reward.append(performances['reward'])
        # print(performances['reward'])
        min_final_percentage = min(min_final_percentage, performances['final_percentage'])

    # print(total_reward)
    mean_reward = np.median(total_reward)
    # print(mean_reward)
    return {
        'reward': mean_reward,
        'done': performances['done'],  # Assuming 'done' and 'dead_time' don't change
        'dead_time': performances['dead_time'],
        'final_percentage': min_final_percentage
    }
