from typing import Callable, Dict
import gym
import jax.numpy as jnp
from cgpax.encoding import genome_to_lgp_program


def _evaluate_program(program: Callable, program_state_size: int, env: gym.Env,
                      episode_length: int = 42) -> Dict:
    obs = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    dead_time = episode_length
    final_percentage = 0
    for i in range(episode_length):
        inputs = jnp.asarray(obs.copy().flatten())

        new_program_state, actions = program(inputs, program_state)
        # print(actions)
        selected_action = jnp.argmax(actions) % 6
        # print(selected_action)
        obs, reward, done, info = env.step(selected_action)  # Make sure action is an integer
        # print(obs.reshape((6, 6)))
        # print("reward:", reward)
        # print(env._check_winner())
        # print("game over?", done)
        # print("winner", )
        final_percentage = reward

        cumulative_reward += reward
        if done:
            break
        # if not done:
        #     cumulative_reward -= (1. - reward)
        # else:
        #     cumulative_reward -= (1. - reward) * (episode_length - i)
        #     done_time = i
        #    break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "dead_time": dead_time,
        "final_percentage": final_percentage
    }


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, env: gym.Env,
                        episode_length: int = 42,  # Adjusted episode length for Connect 4
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    val_ = inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)
    # print(val_)
    return val_
