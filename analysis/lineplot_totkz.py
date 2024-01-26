import os
import glob

import pandas as pd


def process_and_save_csv(general_path, file):

    input_file = general_path + file + ".csv"
    df = pd.read_csv(input_file)

    # Check if 'max_fitness' column exists
    if 'max_fitness' not in df.columns:
        raise ValueError("Column 'max_fitness' not found in the CSV file.")

    # Modify values in 'max_fitness' column as per the requirement
    df['max_fitness'] = df['max_fitness'].apply(lambda x: -10 if x < -10 else x)

    # Check if 'generation' column exists
    if 'generation' not in df.columns:
        raise ValueError("Column 'generation' not found in the CSV file.")

    # Select only the 'generation' and 'max_fitness' columns
    output_df = df[['generation', 'max_fitness']]

    # Save the new DataFrame to a CSV file
    if not os.path.exists(general_path + "_csv_paper"):
        os.mkdir(general_path + "_csv_paper")

    output_file = general_path + "_csv_paper/" + file + ".txt"
    output_df.to_csv(output_file, sep="\t", index=False)

    return output_file


def process_csv_files(general_path):
    # Using glob to find all .csv files in the folder
    csv_files = glob.glob(os.path.join(general_path, '*.csv'))
    file_names = [os.path.basename(file) for file in csv_files]

    # Iterate through each file and apply the provided function
    for file in file_names:

        process_and_save_csv(general_path, file[:-4])
        print(f"Processed: {file}")


seed = 2
general_path = os.getcwd() + f"/../results/results_50_100_False/connect4_trial_{seed}/"
process_csv_files(general_path)
