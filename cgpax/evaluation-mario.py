from typing import Callable, Dict

import gymnasium
import jax.numpy as jnp

from cgpax.encoding import genome_to_cgp_program, genome_to_lgp_program


def _evaluate_program(program: Callable, program_state_size: int, env: gymnasium.Env,
                      episode_length: int = 1000) -> Dict:
    obs, info = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    dead_time = episode_length
    final_percentage = 0
    for i in range(episode_length):
        inputs = jnp.asarray(obs)
        new_program_state, actions = program(inputs, program_state)
        boolean_actions = (actions > 0).tolist()
        obs, reward, done, truncated, info = env.step(boolean_actions)
        final_percentage = reward
        if not done:
            cumulative_reward -= (1. - reward)
        else:
            cumulative_reward -= (1. - reward) * (episode_length - i)
            done_time = i
            if truncated:
                dead_time = i
            break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "dead_time": dead_time,
        "final_percentage": final_percentage
    }


def evaluate_cgp_genome(genome: jnp.ndarray, config: Dict, env: gymnasium.Env,
                        episode_length: int = 1000,
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    return inner_evaluator(genome_to_cgp_program(genome, config), config["buffer_size"], env, episode_length)


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, env: gymnasium.Env,
                        episode_length: int = 1000,
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    return inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)
