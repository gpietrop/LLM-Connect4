#!/bin/bash

# Loop over the desired seed values
for seed in {1..30}; do
    # Run the Python script with the specified seed and strategy configuration
    python gp_evolution_c4.py --seed "$seed" --greedy 0 --greedy_intermediate 0 --greedy_improved 100
    python gp_evolution_c4.py --seed "$seed" --greedy 10 --greedy_intermediate 0 --greedy_improved 90
    python gp_evolution_c4.py --seed "$seed" --greedy 25 --greedy_intermediate 0 --greedy_improved 75    
    python gp_evolution_c4.py --seed "$seed" --greedy 50 --greedy_intermediate 0 --greedy_improved 50
    python gp_evolution_c4.py --seed "$seed" --greedy 75 --greedy_intermediate 0 --greedy_improved 25
    python gp_evolution_c4.py --seed "$seed" --greedy 0 --greedy_intermediate 10 --greedy_improved 90
    python gp_evolution_c4.py --seed "$seed" --greedy 0 --greedy_intermediate 25 --greedy_improved 75
    python gp_evolution_c4.py --seed "$seed" --greedy 0 --greedy_intermediate 50 --greedy_improved 50
    python gp_evolution_c4.py --seed "$seed" --greedy 0 --greedy_intermediate 75 --greedy_improved 25
    python gp_evolution_c4.py --seed "$seed" --greedy 10 --greedy_intermediate 10 --greedy_improved 80
    python gp_evolution_c4.py --seed "$seed" --greedy 10 --greedy_intermediate 30 --greedy_improved 60
    python gp_evolution_c4.py --seed "$seed" --greedy 33 --greedy_intermediate 33 --greedy_improved 33
    # Add more lines for other combinations
done
