#!/bin/bash

# Maximum number of parallel processes
MAX_JOBS=4
current_jobs=0

# Function to wait for processes to finish
wait_for_jobs() {
    while [ $(jobs -p | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# Loop over the desired seed values
for seed in {1..10}; do
    # Loop over the different configurations for 'greedy' and 'greedy_improved'
    for greedy in 25 50 75; do
        greedy_improved=$((100-greedy))

        # Run the Python script with the specified seed and strategy configuration in the background
        python gp_evolution_c4.py --seed "$seed" --greedy "$greedy" --greedy_improved "$greedy_improved" &

        # Check if the number of parallel jobs reached the limit
        ((current_jobs++))
        if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
            wait_for_jobs
            current_jobs=0
        fi
    done
done

# Wait for all background jobs to finish
wait

