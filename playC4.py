import os
import random

import jax.numpy as jnp
import numpy as np
from cgpax.utils import readable_lgp_program_from_genome
import yaml
from cgpax.evaluation import evaluate_lgp_genome
from c4_gym import Connect4Env
from policies import MinimaxPolicy


def validation_policy(seeds=1, n_generations=100, n_individuals=50, ep1=0, ep2=0, ep3=0, ep4=100):
    rewards = []
    victories = []

    for seed in range(1, seeds + 1):
        fitnesses = jnp.load(f"/home/alfa/Desktop/LLM-Connect4-MinMax/results/results_50_100_False"
                                           f"/connect4_trial_{seed}/fitnesses_{ep1}_{ep2}_{ep3}_{ep4}.npy")
        genomes = jnp.load(f"/home/alfa/Desktop/LLM-Connect4-MinMax/results/results_50_100_False"
                                         f"/connect4_trial_{seed}/genotypes_{ep1}_{ep2}_{ep3}_{ep4}.npy")
        with open(f"/home/alfa/Desktop/LLM-Connect4-MinMax/results/results_50_100_False/connect4_trial_{seed}/config_{ep1}_{ep2}_{ep3}_{ep4}.yaml") as f:
            config = yaml.load(f, Loader=yaml.loader.FullLoader)
            config["seed"] = random.randint(0, 100)

        best_genome = genomes[jnp.argmax(fitnesses)]

        othello_env = Connect4Env(num_disk_as_reward=False)

        othello_env.render()
        othello_env.render_in_step = True

        new_policy = MinimaxPolicy()
        reward = evaluate_lgp_genome(best_genome, config, othello_env, episode_length=42, new_policy=new_policy)

        rewards.append(reward["reward"])
        victories.append(reward["final_percentage"])

    mean_reward = np.mean(rewards)
    total_victories = np.sum(victories)

    return mean_reward, total_victories


seeds = 1

e1, e2, e3 = 0, 0, 0
mean_reward, total_victories = validation_policy(seeds=seeds, ep1=e1, ep2=e2, ep3=e3)
print(f"{e1}-{e2}-{e3} reward: {mean_reward}; victories: {total_victories}")





