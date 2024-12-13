#!/bin/bash

n_generations=100
n_individuals=50
pol="original"
python_script="gp_evolution_c4.py"

for seed in {31..32}; do
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 0 --greedy_improved 100 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 0 --greedy_improved 90 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 25 --greedy_intermediate 0 --greedy_improved 75 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 50 --greedy_intermediate 0 --greedy_improved 50 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 10 --greedy_improved 90 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 25 --greedy_improved 75 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 50 --greedy_improved 50 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 10 --greedy_improved 80 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 20 --greedy_improved 70 --policy_version "$pol"
        python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 33 --greedy_intermediate 33 --greedy_improved 33 --policy_version "$pol"
        # Add more lines for other combinations if needed
done

