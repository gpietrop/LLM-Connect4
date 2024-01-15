from typing import Callable, Dict
import gymnasium
import jax.numpy as jnp
from cgpax.encoding import genome_to_lgp_program


def _evaluate_program(program: Callable, program_state_size: int, env: gymnasium.Env,
                      episode_length: int = 1) -> Dict:  # Adjusted episode length for Connect 4
    obs, info = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    # print(episode_length)
    for i in range(episode_length):
        # print(obs)
        inputs = jnp.asarray(obs)
        inputs = inputs * 10000
        program_state, actions = program(inputs, program_state)
        # print(program_state)
        # print(actions)
        # Convert the action to a valid column choice (0 to 6)
        action = jnp.argmax(actions) % 7
        # print("before")
        # print(action)
        # print(env.board)
        obs, reward, done, _, info = env.step(action)
        # print(obs)
        cumulative_reward += reward

        if done:
            done_time = i
            break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "winner": info.get('winner')
    }


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, env: gymnasium.Env,
                        episode_length: int = 1,  # Adjusted episode length for Connect 4
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    val_ = inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)
    # print(val_)
    return val_
