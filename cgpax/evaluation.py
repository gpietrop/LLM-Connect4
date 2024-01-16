from typing import Callable, Dict
import gym
import jax.numpy as jnp
from cgpax.encoding import genome_to_lgp_program


def _evaluate_program(program: Callable, program_state_size: int, env: gym.Env,
                      episode_length: int = 100) -> Dict:
    obs = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    dead_time = episode_length
    final_percentage = 0
    for i in range(episode_length):
        inputs = jnp.asarray(obs)
        new_program_state, actions = program(inputs, program_state)
        selected_action = jnp.argmax(actions) % 6
        print("env: ", env)
        obs, reward, done, info = env.step(selected_action)  # Make sure action is an integer
        final_percentage = reward
        if not done:
            cumulative_reward -= (1. - reward)
        else:
            cumulative_reward -= (1. - reward) * (episode_length - i)
            done_time = i
            break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "dead_time": dead_time,
        "final_percentage": final_percentage
    }


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, env: gym.Env,
                        episode_length: int = 100,  # Adjusted episode length for Connect 4
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    val_ = inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)
    # print(val_)
    return val_
