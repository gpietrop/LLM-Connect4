#!/bin/bash

n_generations=200
n_individuals=51
pol="31_405B_NEW"
python_scripts=("cgp_evolution_c4.py" "lgp_evolution_c4.py")  # List of Python scripts
greedy_configs=(
    "0 0 0 100"
    "10 10 10 70"
    "20 20 20 40"
    "20 0 0 80"
    "0 20 0 80"
    "0 0 20 80"
    "10 20 20 50"
)
adaptive_values=(True False)  # Adaptive flag values

for python_script in "${python_scripts[@]}"; do
    for config in "${greedy_configs[@]}"; do
        for adaptive in "${adaptive_values[@]}"; do
            for seed in {1..31}; do
                IFS=' ' read -r -a greedy <<< "$config"
                python $python_script --seed "$seed" --n_individuals "$n_individuals" --n_generations "$n_generations" \
                    --policy_version "$pol" --greedy "${greedy[0]}" --greedy_intermediate "${greedy[1]}" \
                    --greedy_improved "${greedy[2]}" --greedy_expert "${greedy[3]}" --adaptive "$adaptive"
            done
        done
    done
done
