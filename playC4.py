import os

import jax.numpy as jnp
from cgpax.utils import readable_lgp_program_from_genome
import yaml
from cgpax.evaluation import evaluate_lgp_genome
from c4_gym import Connect4Env
from policies import GreedyPolicy, IntermediateGreedyPolicy, ImprovedGreedyPolicy, RandomPolicy

fitnesses = jnp.load(os.getcwd() + "/results/results_50_100_False/connect4_trial_1/fitnesses_0_50_50.npy")
genomes = jnp.load(os.getcwd() + "/results/results_50_100_False/connect4_trial_1/genotypes_0_50_50.npy")
with open("results/results_50_100_False/connect4_trial_1/config_0_50_50.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.FullLoader)
print(config)

best_genome = genomes[jnp.argmax(fitnesses)]

othello_env = Connect4Env(num_disk_as_reward=False)

othello_env.render()
othello_env.render_in_step = True

new_policy = RandomPolicy()
reward = evaluate_lgp_genome(best_genome, config, othello_env, episode_length=1000, new_policy=new_policy)

print(reward)
# print(readable_lgp_program_from_genome(best_genome, config))
# print(readable_lgp_program_from_genome(genomes[1], config))
# print(best_genome)
# print(fitnesses)
# Collapse



















