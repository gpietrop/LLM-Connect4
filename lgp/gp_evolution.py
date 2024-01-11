import time
from typing import Tuple

from jax import random
import jax.numpy as jnp
from lgp.cgpax.evaluation import evaluate_lgp_genome
from lgp.cgpax.individual import generate_population
from lgp.cgpax.run_utils import update_config_with_env_data, compute_masks, compute_weights_mutation_function, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection
from mario_gym import MarioEnv


# TODO find how many time steps are used in the mario game in java
# in java the best agent has 20 * 1000 as timer, and every tick decreases the timer by 30 (~700 time steps)
def evaluate_genomes(genomes_array: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fitnesses_list = []
    percentages_list = []
    dead_times_list = []
    for genome_array in genomes_array:
        result = evaluate_lgp_genome(genome_array, config, mario_env, episode_length=1000)
        fitnesses_list.append(result["reward"])
        percentages_list.append(result["final_percentage"])
        dead_times_list.append(result["dead_time"])
    return jnp.asarray(fitnesses_list), jnp.asarray(percentages_list), jnp.asarray(dead_times_list)


def log_data(generation: int, fitnesses: jnp.ndarray, percentages: jnp.ndarray, dead_times: jnp.ndarray,
             eval_time: float) -> None:
    max_fitness = max(fitnesses)
    max_percentage = max(percentages)
    max_dead_time = max(dead_times)
    print(f"{generation},{max_fitness:.4f},{max_percentage:.4f},{max_dead_time},{eval_time:.4f}")


if __name__ == '__main__':

    config = {
        "n_rows": 20,
        "n_extra_registers": 5,
        "seed": 0,
        "n_individuals": 100,
        "solver": "lgp",
        "p_mut_lhs": 0.5,
        "p_mut_rhs": 0.5,
        "p_mut_functions": 0.1,
        "n_generations": 1000,
        "selection": {
            "elite_size": 1,
            "type": "tournament",
            "tour_size": 2
        },
        "survival": "truncation",
        "crossover": False,

    }

    rnd_key = random.PRNGKey(config["seed"])

    mario_env = MarioEnv.make()
    update_config_with_env_data(config, mario_env)

    genome_mask, mutation_mask = compute_masks(config)
    weights_mutation_function = compute_weights_mutation_function(config)
    select_parents = compile_parents_selection(config)
    select_survivals = compile_survival_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, weights_mutation_function)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key,
                                  weights_mutation_function=weights_mutation_function)

    for _generation in range(config["n_generations"]):
        start_eval = time.process_time()
        fitnesses, percentages, dead_times = evaluate_genomes(genomes)
        end_eval = time.process_time()
        eval_time = end_eval - start_eval
        log_data(_generation, fitnesses, percentages, dead_times, eval_time)

        # select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        parents = select_parents(genomes, fitnesses, select_key)

        # compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        if config.get("crossover", False):
            parents1, parents2 = jnp.split(parents, 2)
            rnd_key, *xover_keys = random.split(rnd_key, len(parents1) + 1)
            offspring1, offspring2 = crossover_genomes(parents1, parents2, jnp.array(xover_keys))
            new_parents = jnp.concatenate((offspring1, offspring2))
        else:
            new_parents = parents
        offspring_matrix = mutate_genomes(new_parents, mutate_keys)
        offspring = jnp.reshape(offspring_matrix, (-1, offspring_matrix.shape[-1]))

        # select survivals
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitnesses, survival_key)

        # update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))
