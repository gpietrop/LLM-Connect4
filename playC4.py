import os

import jax.numpy as jnp
from cgpax.utils import readable_lgp_program_from_genome
import yaml
from cgpax.evaluation import evaluate_lgp_genome
from c4_gym import Connect4Env
from policies import GreedyPolicy, IntermediateGreedyPolicy, ImprovedGreedyPolicy, RandomPolicy

ep1 = 0
ep2 = 0
ep3 = 100
seed = 12

fitnesses = jnp.load(os.getcwd() + f"/results/results_50_100_False/connect4_trial_{seed}/fitnesses_{ep1}_{ep2}_{ep3}.npy")
genomes = jnp.load(os.getcwd() + f"/results/results_50_100_False/connect4_trial_{seed}/genotypes_{ep1}_{ep2}_{ep3}.npy")
with open(f"results/results_50_100_False/connect4_trial_{seed}/config_{ep1}_{ep2}_{ep3}.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.FullLoader)
# print(config)
# config["seed"] = 999
# print(config["seed"])
best_genome = genomes[jnp.argmax(fitnesses)]

othello_env = Connect4Env(num_disk_as_reward=False)

othello_env.render()
othello_env.render_in_step = True

new_policy = IntermediateGreedyPolicy()
reward = evaluate_lgp_genome(best_genome, config, othello_env, episode_length=42, new_policy=new_policy)

print(reward)
# print(readable_lgp_program_from_genome(best_genome, config))
# print(readable_lgp_program_from_genome(genomes[1], config))
# print(best_genome)
# print(fitnesses)
# Collapse



















