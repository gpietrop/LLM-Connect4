#!/bin/bash

n_generations=100
n_individuals=50

for seed in {1..30}; do
	python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 0 --greedy_improved 100
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 0 --greedy_improved 90
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 25 --greedy_intermediate 0 --greedy_improved 75
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 50 --greedy_intermediate 0 --greedy_improved 50
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 10 --greedy_improved 90
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 25 --greedy_improved 75
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 0 --greedy_intermediate 50 --greedy_improved 50
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 10 --greedy_improved 80
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 10 --greedy_intermediate 20 --greedy_improved 70
        python gp_evolution_c4.py --seed "$seed" --n_individuals "$n_individuals" --n_generations $n_generations --greedy 33 --greedy_intermediate 33 --greedy_improved 33
        # Add more lines for other combinations if needed
done

