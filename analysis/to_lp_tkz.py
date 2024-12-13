import os
import pandas as pd


def convert_csv_to_txt(input_dir, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.startswith("res_") and file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Read the CSV file
                try:
                    df = pd.read_csv(file_path)

                    # Check if both 'generation' and 'fitness' columns exist
                    if 'generation' in df.columns and 'max_fitness' in df.columns:
                        # Extract required columns
                        output_df = df[['generation', 'max_fitness']]

                        # Define output file name with .txt extension
                        output_file_name = file_name.replace(".csv", ".txt")
                        output_file_path = os.path.join(output_dir, output_file_name)

                        # Save to .txt file with tab-separated values
                        output_df.to_csv(output_file_path, sep="\t", index=False, header=True)
                        print(f"Saved: {output_file_path}")
                    else:
                        print(f"Skipping {file_name}: Missing required columns")

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")


# Usage
selected_seeds = range(30)  # Replace with actual seed values
n_individuals = 50
n_generations = 100
gp_models = ["lgp", "cgp"]  # This should be set or determined earlier in your script
llm_models = ["31_405B_NEW", "original"]  # 31_405B_NEW

for selected_seed in selected_seeds:
    for gp_model in gp_models:
        for llm_model in llm_models:
            # Determine input directory based on the type of model
            if gp_model == "lgp":
                input_directory = os.path.join(os.getcwd(),
                                               f'../results/{llm_model}/results_{n_individuals}_{n_generations}_False'
                                               f'/connect4_trial_{selected_seed}')
            elif gp_model == "cgp":
                input_directory = os.path.join(os.getcwd(),
                                               f'../results_cgp/{llm_model}/results_{n_individuals}_{n_generations}_False'
                                               f'/connect4_trial_{selected_seed}')

            # Construct the output file path
            output_dir = os.path.abspath(os.path.join(os.getcwd(), f"../tkz/lp/{selected_seed}/{llm_model}_{gp_model}_{n_individuals}"))

            convert_csv_to_txt(input_directory, output_dir)
