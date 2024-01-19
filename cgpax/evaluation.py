from typing import Callable, Dict
import gym
import jax.numpy as jnp
from cgpax.encoding import genome_to_lgp_program

from policies import GreedyPolicy


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
        # print(selected_action)
        obs, total_reward, done, info = env.step(selected_action)  # Make sure action is an integer
        # print(obs.reshape((6, 6)))

        final_percentage = total_reward[1]
        cumulative_reward += total_reward[0]
        if done:
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

    total_reward = 0.
    min_final_percentage = 1.

    for _ in range(5):
        performances = inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)

        total_reward += performances['reward']
        min_final_percentage = min(min_final_percentage, performances['final_percentage'])

    mean_reward = np.median(total_reward)

    return {
        'reward': mean_reward,
        'done': performances['done'],  # Assuming 'done' and 'dead_time' don't change
        'dead_time': performances['dead_time'],
        'final_percentage': min_final_percentage
    }