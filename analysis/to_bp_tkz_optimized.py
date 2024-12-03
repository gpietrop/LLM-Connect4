import os
import pandas as pd
from collections import defaultdict


def process_csv_files(input_dir, output_file, num_policy_elements, selected_seeds=None, min_generations=98):
    grouped_results = defaultdict(list)  # Group results by policies

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.startswith("res_") and file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                seed_dir = os.path.basename(root)
                seed = extract_seed(seed_dir, selected_seeds)
                if seed is None:
                    continue

                df = pd.read_csv(file_path)
                if df['generation'].max() < min_generations:
                    print(f"Skipping {file_name} (incomplete)")
                    continue

                policies = extract_policies(file_name, num_policy_elements)
                start_generation = sum(policies[:-1])
                non_zero_percentage = df[df['generation'] > start_generation]
                non_zero_percentage = non_zero_percentage[non_zero_percentage['percentage'] > 0]

                if not non_zero_percentage.empty:
                    first_win_generation = non_zero_percentage['generation'].iloc[0]
                    grouped_results[policies].append(first_win_generation)
                else:
                    print(f"No winning found in {file_name}")

    write_grouped_results(grouped_results, output_file)


def extract_seed(seed_dir, selected_seeds):
    if "connect4_trial_" in seed_dir:
        try:
            seed = int(seed_dir.split("connect4_trial_")[-1])
            if selected_seeds is not None and seed not in selected_seeds:
                return None
            return seed
        except ValueError:
            return None
    return None


def extract_policies(file_name, num_elements):
    parts = file_name.split("_")[1:num_elements + 1]
    return tuple(int(p.replace(".csv", "")) for p in parts)


def write_grouped_results(grouped_results, output_file):
    with open(output_file, "w") as file:
        header = "\t".join([f"res_{'_'.join(map(str, k))}" for k in sorted(grouped_results.keys())])
        file.write(header + "\n")
        max_entries = max(len(values) for values in grouped_results.values())
        for i in range(max_entries):
            row = [str(grouped_results[k][i]) if i < len(grouped_results[k]) else "" for k in
                   sorted(grouped_results.keys())]
            file.write("\t".join(row) + "\n")


selected_seeds = range(1, 31)  # Specify the seeds you want to include
llm_model = "31_405B_NEW"
gp_model = "lgp"
n_individuals = 101
n_generations = 100
if gp_model == "lgp":
    input_directory = os.path.join(os.getcwd(),
                                   f'../results/{llm_model}/results_{n_individuals}_{n_generations}_False/')
if gp_model == "cgp":
    input_directory = os.path.join(os.getcwd(),
                                   f'../results_cgp/{llm_model}/results_{n_individuals}_{n_generations}_False/')

# Construct the output file path
output_txt = os.path.abspath(os.path.join(os.getcwd(), f"../tkz/{llm_model}_{gp_model}_{n_individuals}_c4-bp.txt"))

# Usage example with parameter for number of policies (elements in file name before '.csv')
process_csv_files(input_directory, output_txt, 4, selected_seeds=selected_seeds)
