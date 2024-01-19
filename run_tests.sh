#!/bin/bash

# Loop over the desired seed values
for seed in {1..5}; do
    # Loop over the different configurations for 'greedy' and 'greedy_improved'
    for greedy in 25 50 75; do
        greedy_improved=$((100-greedy))

        # Run the Python script with the specified seed and strategy configuration
        python gp_evolution_c4.py --seed "$seed" --greedy "$greedy" --greedy_improved "$greedy_improved"
    done
done
