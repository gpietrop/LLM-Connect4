import os
import time
from typing import Tuple, Dict
from multiprocessing import Pool
import numpy as np

import yaml
from jax import random
import jax

jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from cgpax.evaluation import evaluate_lgp_genome
from cgpax.individual import generate_population
from cgpax.run_utils import update_config_with_env_data, compute_masks, compute_weights_mutation_function, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection
from cgpax.utils import CSVLogger
from c4_gym import Connect4Env  # Make sure to import the correct class
from tqdm import tqdm


def evaluate_genomes(genomes_array: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fitnesses_list = []
    percentages_list = []
    dead_times_list = []

    args = [(genome, config, Connect4Env(num_disk_as_reward=True), 42) for genome in genomes_array]

    with Pool(processes=os.cpu_count() - 1) as pool:
        for result in pool.starmap(evaluate_lgp_genome, args):
            fitnesses_list.append(result["reward"])
            percentages_list.append(result["final_percentage"])
            dead_times_list.append(result["dead_time"])

    return jnp.asarray(fitnesses_list), jnp.asarray(percentages_list), jnp.asarray(dead_times_list)


if __name__ == '__main__':
    config = {
        "n_rows": 20,
        "n_extra_registers": 5,
        "seed": 100,
        "n_individuals": 1,
        "solver": "lgp",
        "p_mut_lhs": 0.0,
        "p_mut_rhs": 0.0,
        "p_mut_functions": 0.0,
        "n_generations": 10,
        "selection": {
            "elite_size": 0,
            "type": "tournament",
            "tour_size": 2
        },
        "survival": "truncation",
        "crossover": False,
        "run_name": "connect4_trial"
    }

    run_name = f"{config['run_name']}_{config['seed']}"
    os.makedirs(f"results/{run_name}", exist_ok=True)

    rnd_key = random.PRNGKey(config["seed"])

    # Initialize the Connect 4 environment
    connect4_env = Connect4Env(num_disk_as_reward=True)
    update_config_with_env_data(config, connect4_env)

    # Compute masks and compile various functions for genetic programming
    genome_mask, mutation_mask = compute_masks(config)
    weights_mutation_function = compute_weights_mutation_function(config)
    select_parents = compile_parents_selection(config)
    select_survivals = compile_survival_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, weights_mutation_function)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key,
                                  weights_mutation_function=weights_mutation_function)

    csv_logger = CSVLogger(
        filename=f"results/{run_name}/metrics.csv",
        header=["generation", "max_fitness", "max_percentage", "max_dead_time", "eval_time"]
    )

    # Evolution loop
    for _generation in tqdm(range(config["n_generations"]), position=1, desc="Generations"):
        start_eval = time.process_time()
        # print(genomes)
        fitnesses, percentages, dead_times = evaluate_genomes(genomes)
        # print(fitnesses, percentages, dead_times)
        end_eval = time.process_time()
        eval_time = end_eval - start_eval

        # Log metrics
        metrics = {
            "generation": _generation,
            "max_fitness": max(fitnesses),
            "max_percentage": max(percentages),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time
        }
        print("\n mean fitness", np.mean(fitnesses))
        csv_logger.log(metrics)

        # Parent selection
        rnd_key, select_key = random.split(rnd_key, 2)
        parents = select_parents(genomes, fitnesses, select_key)

        # Compute offspring
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

        # Survival selection
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitnesses, survival_key)

        # Update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))

    # Save final results
    jnp.save(f"results/{run_name}/genotypes.npy", genomes)
    jnp.save(f"results/{run_name}/fitnesses.npy", fitnesses)
    with open(f"results/{run_name}/config.yaml", "w") as file:
        yaml.dump(config, file)

    # Render and evaluate the best genome
    connect4_env.render()
    best_genome = genomes[jnp.argmax(fitnesses)]
    jnp.save(f"results/{run_name}/best_genome.npy", best_genome)
    evaluate_lgp_genome(best_genome, config, connect4_env, episode_length=42)



