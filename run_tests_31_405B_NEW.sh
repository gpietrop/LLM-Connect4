#!/bin/bash

n_generations=100
n_individuals=51
pol="31_405B_NEW"

for seed in {16..30}; do
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 0 --greedy_improved 0 --greedy_expert 100 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 10 --greedy_improved 10 --greedy_expert 70 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 20 --greedy_intermediate 20 --greedy_improved 20 --greedy_expert 40 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 20 --greedy_intermediate 0 --greedy_improved 0 --greedy_expert 80 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 20 --greedy_improved 0 --greedy_expert 80 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 0 --greedy_improved 20 --greedy_expert 80 --policy_version "$pol"
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 20 --greedy_improved 20 --greedy_expert 50 --policy_version "$pol"
done

