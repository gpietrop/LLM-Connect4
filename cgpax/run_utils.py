from functools import partial
from typing import Callable, Tuple, Dict, Union

import jax.numpy as jnp
from jax import vmap, jit, random

from cgpax.functions import available_functions, constants
from cgpax.individual import mutate_genome_n_times, mutate_genome_n_times_stacked, compute_cgp_genome_mask, \
    compute_cgp_mutation_prob_mask, compute_lgp_genome_mask, compute_lgp_mutation_prob_mask, \
    levels_back_transformation_function, lgp_one_point_crossover_genomes
from cgpax.selection import truncation_selection, tournament_selection, fp_selection, composed_selection
from cgpax.utils import identity


def update_config_with_data(config: Dict, observation_space_size: int, action_space_size: int) -> None:
    config["n_functions"] = len(available_functions)
    config["n_constants"] = len(constants) if config.get("use_input_constants", True) else 0

    config["n_in_env"] = observation_space_size
    config["n_in"] = config["n_in_env"] + config["n_constants"]
    config["n_out"] = action_space_size

    if config["solver"] == "cgp":
        config["buffer_size"] = config["n_in"] + config["n_nodes"]
        config["genome_size"] = 4 * config["n_nodes"] + config["n_out"]
        levels_back = config.get("levels_back")
        if levels_back is not None and levels_back < config["n_in"]:
            config["levels_back"] = config["n_in"]
    else:
        config["n_registers"] = config["n_in"] + config["n_extra_registers"] + config["n_out"]
        config["genome_size"] = 5 * config["n_rows"]


def update_config_with_env_data(config: Dict, env) -> None:
    update_config_with_data(config, env.observation_size, env.action_size)


def compute_parallel_runs_indexes(n_individuals: int, n_parallel_runs: int, n_elites: int = 1) -> jnp.ndarray:
    indexes = jnp.zeros((n_parallel_runs, n_individuals))
    for run_idx in range(n_parallel_runs):
        for elite_idx in range(n_elites):
            indexes = indexes.at[run_idx, elite_idx].set(run_idx * n_elites + elite_idx)
        for ind_idx in range(n_individuals - n_elites):
            indexes = indexes.at[run_idx, ind_idx + n_elites].set(
                n_elites * n_parallel_runs + ind_idx + (n_individuals - n_elites) * run_idx)
    return indexes.astype(int)


def compile_crossover(config: Dict) -> Union[Callable, None]:
    if config.get("crossover", False) and config["solver"] == "lgp":
        vmap_crossover = vmap(lgp_one_point_crossover_genomes, in_axes=(0, 0, 0))
        return jit(vmap_crossover)
    else:
        return None


def compile_mutation(config: Dict, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                     weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                     genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = identity,
                     n_mutations_per_individual: int = 1) -> Callable:
    if config.get("mutation", "standard") == "standard":
        partial_multiple_mutations = partial(mutate_genome_n_times, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask,
                                             weights_mutation_function=weights_mutation_function,
                                             genome_transformation_function=genome_transformation_function)
    else:
        partial_multiple_mutations = partial(mutate_genome_n_times_stacked, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask,
                                             weights_mutation_function=weights_mutation_function,
                                             genome_transformation_function=genome_transformation_function)

    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    return jit(vmap_multiple_mutations)


def compile_survival_selection(config: Dict) -> Union[Callable, None]:
    if config["survival"] == "parents":
        return None
    elif config["survival"] == "truncation":
        return jit(partial(truncation_selection, n_elites=config["selection"]["elite_size"]))
    elif config["survival"] == "tournament":
        return jit(partial(tournament_selection, n_elites=config["selection"]["elite_size"],
                           tour_size=config["selection"]["tour_size"]))
    else:
        return jit(partial(fp_selection, n_elites=config["selection"]["elite_size"]))


def compile_parents_selection(config: Dict, n_parents: int = 0) -> Callable:
    if n_parents == 0:
        n_parents = config["n_individuals"] - config["selection"]["elite_size"]
    if config["selection"]["type"] == "truncation":
        partial_selection = partial(truncation_selection, n_elites=n_parents)
    elif config["selection"]["type"] == "tournament":
        partial_selection = partial(tournament_selection, n_elites=n_parents,
                                    tour_size=config["selection"]["tour_size"])
    else:
        partial_selection = partial(fp_selection, n_elites=n_parents)
    inner_selection = jit(partial_selection)
    if config.get("n_parallel_runs", 1) == 1:
        return inner_selection
    else:
        def _composite_selection(genomes, fitness_values, select_key):
            parents_list = []
            for run_idx in config["runs_indexes"]:
                rnd_key, sel_key = random.split(select_key, 2)
                current_parents = composed_selection(genomes, fitness_values, sel_key, run_idx, inner_selection)
                parents_list.append(current_parents)
            parents_matrix = jnp.array(parents_list)
            return jnp.reshape(parents_matrix, (-1, parents_matrix.shape[-1]))

        return _composite_selection


def compute_masks(config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if config["solver"] == "cgp":
        genome_mask = compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
        mutation_mask = compute_cgp_mutation_prob_mask(config, config["n_out"])
    else:
        genome_mask = compute_lgp_genome_mask(config, config["n_in"])
        mutation_mask = compute_lgp_mutation_prob_mask(config)
    return genome_mask, mutation_mask


def compute_weights_mutation_function(config: Dict) -> Callable[[random.PRNGKey], jnp.ndarray]:
    sigma = config.get("weights_sigma", 0.0)
    length = config.get("n_rows", config.get("n_nodes"))

    def _gaussian_function(rnd_key: random.PRNGKey) -> jnp.ndarray:
        return random.normal(key=rnd_key, shape=[length]) * sigma

    return _gaussian_function


def compute_genome_transformation_function(config: Dict) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if config["solver"] == "cgp" and config.get("levels_back") is not None:
        return levels_back_transformation_function(config["n_in"], config["n_nodes"])
    else:
        return identity
