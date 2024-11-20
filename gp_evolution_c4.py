import os
import time
import argparse
from typing import Tuple, Dict
from multiprocessing import Pool
import numpy as np

import yaml
from jax import random
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

from cgpax.evaluation import evaluate_lgp_genome
from cgpax.individual import generate_population
from cgpax.run_utils import update_config_with_env_data, compute_masks, compute_weights_mutation_function, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection
from cgpax.utils import CSVLogger

from c4_gym import Connect4Env
from tqdm import tqdm
# from policies import GreedyPolicy, IntermediateGreedyPolicy, ImprovedGreedyPolicy


def evaluate_genomes(genomes_array: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fitnesses_list = []
    percentages_list = []
    dead_times_list = []

    args = [(genome, config, Connect4Env(num_disk_as_reward=True), 42, new_policy) for genome in genomes_array]

    with Pool(processes=os.cpu_count() - 1) as pool:
        for result in pool.starmap(evaluate_lgp_genome, args):
            fitnesses_list.append(result["reward"])
            percentages_list.append(result["final_percentage"])
            dead_times_list.append(result["dead_time"])

    return jnp.asarray(fitnesses_list), jnp.asarray(percentages_list), jnp.asarray(dead_times_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run GP evolution experiments.')
    parser.add_argument('--seed', type=int, help='Seed for the experiment', default=1)
    parser.add_argument('--n_individuals', type=int, help='Seed for the experiment', default=10)
    parser.add_argument('--n_generations', type=int, help='Seed for the experiment', default=10)
    parser.add_argument('--policy_version', type=str, choices=['original', '31_8B', '31_405B', '31_8B_GPT'],
                        help='Policy version to use', default='original')
    parser.add_argument('--greedy', type=int, help='Percentage for greedy strategy', default=100)
    parser.add_argument('--greedy_intermediate', type=int, help='Percentage for intermediate greedy strategy',
                        default=0)
    parser.add_argument('--greedy_improved', type=int, help='Percentage for improved greedy strategy', default=0)
    parser.add_argument('--adaptive', type=bool, help='Adaptive change of policy', default=False)

    args = parser.parse_args()

    if args.policy_version == 'original':
        from policies import GreedyPolicy, IntermediateGreedyPolicy, ImprovedGreedyPolicy
    if args.policy_version == '31_8B':
        from policies_31_8B import EasyPolicy as GreedyPolicy
        from policies_31_8B import MediumPolicy as IntermediateGreedyPolicy
        from policies_31_8B import HardPolicy as ImprovedGreedyPolicy
    if args.policy_version == '31_8B_GPT':
        from policies_31_8B import EasyPolicy as GreedyPolicy
        from policies_31_8B import MediumPolicy as IntermediateGreedyPolicy
        from policies_31_8B import HardPolicy as ImprovedGreedyPolicy
    if args.policy_version == '31_405B':
        from policies_31_405B import EasyPolicy as GreedyPolicy
        from policies_31_405B import MediumPolicy as IntermediateGreedyPolicy
        from policies_31_405B import HardPolicy as ImprovedGreedyPolicy


    config = {
        "n_rows": 20,
        "n_extra_registers": 5,
        "seed": args.seed,
        "n_individuals": args.n_individuals,
        "solver": "lgp",
        "p_mut_lhs": 0.3,
        "p_mut_rhs": 0.1,
        "p_mut_functions": 0.1,
        "n_generations": args.n_generations,
        "yellow_strategy": {
            "greedy": args.greedy,
            "greedy_intermediate": args.greedy_intermediate,
            "greedy_improved": args.greedy_improved
        },
        "selection": {
            "elite_size": 5,
            "type": "tournament",
            "tour_size": 2
        },
        "survival": "truncation",
        "crossover": False,
        "run_name": "connect4_trial",
        "adaptive": args.adaptive
    }

    dir_name = f'results/{args.policy_version}/results_{config["n_individuals"]}_{config["n_generations"]}_{config["adaptive"]}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    run_name = f"{config['run_name']}_{config['seed']}"
    os.makedirs(f"{dir_name}/{run_name}", exist_ok=True)

    rnd_key = random.PRNGKey(config["seed"])

    # Initialize the Connect 4 environment
    connect4_env = Connect4Env(num_disk_as_reward=True)
    update_config_with_env_data(config, connect4_env)

    # Load the opponent strategy config and calculate the number of gen for each policy
    yellow_strategy = config["yellow_strategy"]

    n_generations = config["n_generations"]

    n_greedy_generations = int(n_generations * yellow_strategy["greedy"] / 100)
    n_greedy_intermediate_generations = int(n_generations * yellow_strategy["greedy_intermediate"] / 100)
    n_greedy_improved_generations = n_generations - (n_greedy_generations + n_greedy_intermediate_generations)

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
        filename=f"{dir_name}/{run_name}/res_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.csv",
        header=["generation", "max_fitness", "mean_fitness",
                "max_dead_time", "eval_time", "percentage"]
    )

    # Evolution loop
    for _generation in tqdm(range(n_greedy_generations), position=1, desc="Generations (Greedy Policy)"):
        start_eval = time.process_time()

        # set the opponent policy
        new_policy = GreedyPolicy()
        fitnesses, percentages, dead_times = evaluate_genomes(genomes)
        end_eval = time.process_time()
        eval_time = end_eval - start_eval

        # Log metrics
        metrics = {
            "generation": _generation,
            "max_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time,
            "percentage": np.mean(percentages)
        }
        csv_logger.log(metrics)
        # print(np.mean(percentages))
        # print(f"\n best fitness {max(fitnesses)} \t percentage win: {np.round(np.mean(percentages), 2) * 100}%")

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

        # if change policy flag is set to True: change the policy when the best individual win
        if config["adaptive"] and np.mean(percentages) > 1:
            # transfer the number of evaluation left to the other policy
            n_greedy_intermediate_generations = n_greedy_improved_generations + (n_greedy_generations - n_generations)
            break

    for _generation in tqdm(range(n_greedy_intermediate_generations), position=1,
                            desc="Generations (Greedy Intermediate Policy)"):
        start_eval = time.process_time()

        # set the opponent policy
        new_policy = GreedyPolicy()
        fitnesses, percentages, dead_times = evaluate_genomes(genomes)
        end_eval = time.process_time()
        eval_time = end_eval - start_eval

        # Log metrics
        metrics = {
            "generation": _generation + n_greedy_generations,
            "max_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time,
            "percentage": np.mean(percentages)
        }
        csv_logger.log(metrics)
        # print(np.mean(percentages))
        # print(f"\n best fitness {max(fitnesses)} \t percentage win: {np.round(np.mean(percentages), 2) * 100}%")

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

        # if change policy flag is set to True: change the policy when the best individual win
        if config["adaptive"] and np.mean(percentages) > 1:
            # transfer the number of evaluation left to the other policy
            n_greedy_improved_generations = n_greedy_improved_generations + (n_greedy_intermediate_generations - n_generations)
            break

    for _generation in tqdm(range(n_greedy_improved_generations), position=1,
                            desc="Generations (Greedy Improved Policy)"):
        start_eval = time.process_time()

        # set the opponent policy
        new_policy = ImprovedGreedyPolicy()
        fitnesses, percentages, dead_times = evaluate_genomes(genomes)
        end_eval = time.process_time()
        eval_time = end_eval - start_eval

        # Log metrics
        metrics = {
            "generation": _generation + n_greedy_generations + n_greedy_intermediate_generations,
            "max_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time,
            "percentage": np.mean(percentages)
        }
        csv_logger.log(metrics)
        # print(f"\n best fitness {max(fitnesses)} \t percentage win: {np.round(np.mean(percentages), 2) * 100}")

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
    jnp.save(f"{dir_name}/{run_name}/genotypes_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.npy",
             genomes)
    jnp.save(f"{dir_name}/{run_name}/fitnesses_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.npy",
             fitnesses)
    with open(f"{dir_name}/{run_name}/config_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.yaml",
              "w") as file:
        yaml.dump(config, file)
    # Save final results also on a .txt format
    np.savetxt(f"{dir_name}/{run_name}/genotypes_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.txt",
               np.array(genomes), fmt='%s')
    np.savetxt(f"{dir_name}/{run_name}/fitnesses.txt_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}",
               np.array(fitnesses), fmt='%s')

    # Render and evaluate the best genome
    # connect4_env.render()
    best_genome = genomes[jnp.argmax(fitnesses)]
    jnp.save(f"{dir_name}/{run_name}/best_genome_{yellow_strategy['greedy']}_{yellow_strategy['greedy_intermediate']}_{yellow_strategy['greedy_improved']}.npy",
             best_genome)

    policy = GreedyPolicy()
    evaluate_lgp_genome(best_genome, config, connect4_env, episode_length=42, new_policy=policy)
