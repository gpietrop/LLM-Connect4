import os
import jax.numpy as jnp
from cgpax.utils import readable_lgp_program_from_genome
import yaml
from cgpax.evaluation import evaluate_lgp_genome, evaluate_cgp_genome
from c4_gym import Connect4Env
from policies_validation import MinimaxPolicy
from policies import RandomPolicy, GreedyPolicy

llm = "original"  # "31_405B_NEW"
exp_name = "results_50_100_False"
gp_model = "lgp"
curriculum = "0_0_100"
num_runs = 30
episode_length = 50

# Base directory based on gp_model
base_dir = os.getcwd() + ("/results_cgp" if gp_model == "cgp" else "/results")

# Counters for victories, losses, and draws
victories = 0
losses = 0
draws = 0

# Loop through each run
for seed in range(30, num_runs + 1):
    try:
        # Load fitnesses and genomes for the current seed
        fitnesses = jnp.load(f"{base_dir}/{llm}/{exp_name}/connect4_trial_{seed}/fitnesses_{curriculum}.npy")
        genomes = jnp.load(f"{base_dir}/{llm}/{exp_name}/connect4_trial_{seed}/genotypes_{curriculum}.npy")

        # Load configuration
        config_path = f"{base_dir}/{llm}/{exp_name}/connect4_trial_{seed}/config_{curriculum}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.loader.FullLoader)
            print(f"Run {seed}: Config = {config}")

        # Select the best genome
        best_genome = genomes[jnp.argmax(fitnesses)]

        # Set up the environment and policy
        c4_env = Connect4Env(num_disk_as_reward=False)
        # c4_env.render()
        # c4_env.render_in_step = True

        # new_policy = MinimaxPolicy()
        new_policy = RandomPolicy()

        # Evaluate the genome against Minimax
        if gp_model == "lgp":
            results = evaluate_lgp_genome(best_genome, config, c4_env, episode_length=episode_length, new_policy=new_policy)
        if gp_model == "cgp":
            results = evaluate_cgp_genome(best_genome, config, c4_env, episode_length=episode_length,
                                          new_policy=new_policy)

        reward = results["reward"]
        perc = results["final_percentage"]
        # Record the result
        if reward > 0 and perc == 1.0:
            victories += 1
        elif -5 < reward < 3 and perc == 0:
            losses += 1
        else:
            draws += 1

        print(f"Run {seed}: Reward = {reward}, Perc: = {perc}")
    except Exception as e:
        print(f"Error in run {seed}: {e}")

# Calculate percentages
total_games = victories + losses + draws
victory_percentage = (victories / total_games) * 100 if total_games > 0 else 0

# Print results
print(f"Total Runs: {num_runs}")
print(f"Victories: {victories}, Losses: {losses}, Draws: {draws}")
print(f"Victory Percentage: {victory_percentage:.2f}%")
