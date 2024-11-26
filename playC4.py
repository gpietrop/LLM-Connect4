import os

import jax.numpy as jnp
from cgpax.utils import readable_lgp_program_from_genome
import yaml
from cgpax.evaluation import evaluate_lgp_genome
from c4_gym import Connect4Env
from policies import GreedyPolicy, IntermediateGreedyPolicy, ImprovedGreedyPolicy, RandomPolicy

llm = "31_405B_NEW"
exp_name = "results_50_100_False"
seed = 1
curriculum = "10_10_80"

fitnesses = jnp.load(os.getcwd() + f"/results/{llm}/{exp_name}/connect4_trial_{seed}/fitnesses_{curriculum}.npy")
genomes = jnp.load(os.getcwd() + f"/results/{llm}/{exp_name}/connect4_trial_{seed}/genotypes_{curriculum}.npy")
with open(f"results/{llm}/{exp_name}/connect4_trial_{seed}/config_{curriculum}.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.FullLoader)
print(config)

best_genome = genomes[jnp.argmax(fitnesses)]

c4_env = Connect4Env(num_disk_as_reward=False)

c4_env.render()
c4_env.render_in_step = True

new_policy = IntermediateGreedyPolicy()
reward = evaluate_lgp_genome(best_genome, config, c4_env, episode_length=1000, new_policy=new_policy)

print(reward)
# print(readable_lgp_program_from_genome(best_genome, config))
# print(readable_lgp_program_from_genome(genomes[1], config))
# print(best_genome)
# print(fitnesses)
# Collapse



















