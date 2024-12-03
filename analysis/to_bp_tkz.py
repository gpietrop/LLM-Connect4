import os
import pandas as pd
from collections import defaultdict


def process_csv_files_3_policies(input_dir, output_file, flag, selected_seeds=None, min_generations=98):
    grouped_results = defaultdict(list)  # Group results by x, y, z

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.startswith("res_") and file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Extract seed from the directory name
                seed_dir = os.path.basename(root)
                seed = None
                if "connect4_trial_" in seed_dir:
                    try:
                        seed = int(seed_dir.split("connect4_trial_")[-1])
                    except ValueError:
                        seed = None

                # Skip file if the seed is not in the selected seeds
                if selected_seeds is not None and seed not in selected_seeds:
                    continue

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the number of generations is sufficient
                if df['generation'].max() < min_generations:
                    print(f"Skipping incomplete run: {file_name} (less than {min_generations} generations)")
                    continue  # Skip this file if it's incomplete

                # Extract x, y, z from the file name
                parts = file_name.split("_")
                x = int(parts[1])
                y = int(parts[2])
                z = int(parts[3].replace(".csv", ""))  # Remove the extension

                # Locate the first percentage of winning after generation x + y
                start_generation = x + y
                df_after_start = df[df['generation'] > start_generation]
                non_zero_percentage = df_after_start[df_after_start['percentage'] > 0]

                # Check if there are any non-zero percentages
                if not non_zero_percentage.empty:
                    first_win_generation = non_zero_percentage['generation'].iloc[0]
                    if flag:
                        first_win_generation -= (x + y)
                    grouped_results[(x, y, z)].append(first_win_generation)
                else:
                    print(f"No winning found in file: {file_name}")

    # Write the grouped results to a .txt file
    with open(output_file, "w") as output:
        # Write headers
        header = "\t".join([f"res_{x}_{y}_{z}" for x, y, z in sorted(grouped_results.keys())])
        output.write(header + "\n")

        # Write rows (align columns by filling missing entries with blanks)
        max_entries = max(len(values) for values in grouped_results.values())
        for i in range(max_entries):
            row = []
            for key in sorted(grouped_results.keys()):
                row.append(str(grouped_results[key][i]) if i < len(grouped_results[key]) else "")
            output.write("\t".join(row) + "\n")


def process_csv_files_4_policies(input_dir, output_file, flag, selected_seeds=None, min_generations=98):
    grouped_results = defaultdict(list)  # Group results by x, y, z

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.startswith("res_") and file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Extract seed from the directory name
                seed_dir = os.path.basename(root)
                seed = None
                if "connect4_trial_" in seed_dir:
                    try:
                        seed = int(seed_dir.split("connect4_trial_")[-1])
                    except ValueError:
                        seed = None

                # Skip file if the seed is not in the selected seeds
                if selected_seeds is not None and seed not in selected_seeds:
                    continue

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the number of generations is sufficient
                if df['generation'].max() < min_generations:
                    print(f"Skipping incomplete run: {file_name} (less than {min_generations} generations)")
                    continue  # Skip this file if it's incomplete

                # Extract x, y, z from the file name
                parts = file_name.split("_")
                x = int(parts[1])
                y = int(parts[2])
                e = int(parts[3])
                z = int(parts[4].replace(".csv", ""))  # Remove the extension

                # Locate the first percentage of winning after generation x + y
                start_generation = x + y + e
                df_after_start = df[df['generation'] > start_generation]
                non_zero_percentage = df_after_start[df_after_start['percentage'] > 0]

                # Check if there are any non-zero percentages
                if not non_zero_percentage.empty:
                    first_win_generation = non_zero_percentage['generation'].iloc[0]
                    if flag:
                        first_win_generation -= (x + y + e)
                    grouped_results[(x, y, e, z)].append(first_win_generation)
                else:
                    print(f"{seed:} No winning found in file: {file_name}")

    # Write the grouped results to a .txt file
    with open(output_file, "w") as output:
        # Write headers
        header = "\t".join([f"res_{x}_{y}_{e}_{z}" for x, y, e, z in sorted(grouped_results.keys())])
        output.write(header + "\n")

        # Write rows (align columns by filling missing entries with blanks)
        max_entries = max(len(values) for values in grouped_results.values())
        for i in range(max_entries):
            row = []
            for key in sorted(grouped_results.keys()):
                row.append(str(grouped_results[key][i]) if i < len(grouped_results[key]) else "")
            output.write("\t".join(row) + "\n")


# Usage
selected_seeds = range(30)  # Replace with actual seed values
n_individuals = 50
n_generations = 100
gp_model = "lgp"  # This should be set or determined earlier in your script
llm_model = "31_405B_NEW"  # 31_405B_NEW
flag_after_iteration_diff_policy = 0

# Determine input directory based on the type of model
if gp_model == "lgp":
    input_directory = os.path.join(os.getcwd(), f'../results/{llm_model}/results_{n_individuals}_{n_generations}_False/')
elif gp_model == "cgp":
    input_directory = os.path.join(os.getcwd(), f'../results_cgp/{llm_model}/results_{n_individuals}_{n_generations}_False/')

# Construct the output file path
output_txt = os.path.abspath(os.path.join(os.getcwd(), f"../tkz/{llm_model}_{gp_model}_{n_individuals}_c4-bp.txt"))
if flag_after_iteration_diff_policy:
    output_txt = os.path.abspath(os.path.join(os.getcwd(), f"../tkz_relative/{llm_model}_{gp_model}_{n_individuals}_c4-bp.txt"))

# Apply different processing functions based on the last digit of n_individuals
if str(n_individuals).endswith("1"):
    process_csv_files_4_policies(input_directory, output_txt, flag_after_iteration_diff_policy, selected_seeds=selected_seeds)
else:
    process_csv_files_3_policies(input_directory, output_txt, flag_after_iteration_diff_policy, selected_seeds=selected_seeds)