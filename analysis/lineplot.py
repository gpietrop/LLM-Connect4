import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Setting up the aesthetics for the plot
sns.set(style="whitegrid")

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def lp_fitness(my_seed):
    # Path to the directory containing the metrics files
    directory_path = os.path.join(os.getcwd(), f'../results/connect4_trial_{my_seed}')
    metrics_files = glob.glob(os.path.join(directory_path, 'res_*.csv'))

    # Setting up the plot
    plt.figure(figsize=(10, 6))

    # Loop through each file and plot its data
    for file_path in metrics_files:
        data = pd.read_csv(file_path)
        label = os.path.basename(file_path).replace('.csv', '')  # Use file name as label
        # sns.lineplot(x='generation', y='mean_fitness', data=data, label=f'{label} - Mean Fitness')
        sns.lineplot(x='generation', y='max_fitness', data=data, label=f'{label} - Max Fitness')

    # Adding title and labels
    plt.title('Mean Fitness and Max Fitness Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Save and show the plot
    plt.savefig(os.path.join(directory_path, 'lp_fitness.png'))
    plt.show()
    plt.close()


def lp_fitness_statistical(seeds, n_generations=100, n_individuals=10, adaptive=False):
    # Initialize a dictionary to store data from each file across seeds
    aggregated_data = {}

    # Iterate over each seed
    for my_seed in seeds:
        directory_path = os.path.join(os.getcwd(), f'../results/results_{n_individuals}_{n_generations}_{adaptive}/connect4_trial_{my_seed}')
        metrics_files = glob.glob(os.path.join(directory_path, 'res_*.csv'))

        # Loop through each file
        for file_path in metrics_files:
            file_name = os.path.basename(file_path)
            data = pd.read_csv(file_path)

            # Aggregate data for the same file across different seeds
            if file_name not in aggregated_data:
                aggregated_data[file_name] = []
            aggregated_data[file_name].append(data)

    # Set style for more sophisticated look
    sns.set(style="whitegrid", context="talk", font_scale=1)
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    # Setting up the plot with a smaller figure size
    plt.figure(figsize=(10, 6))

    # Process and plot the aggregated data
    for file_name, data_list in aggregated_data.items():
        # Concatenate data from different seeds and compute the mean
        concatenated_data = pd.concat(data_list)
        mean_data = concatenated_data.groupby('generation').mean().reset_index()
        # label = file_name.replace('.csv', '')  # Use file name as label
        match = re.match(r'res_(\d+)_(\d+).csv', file_name)
        if match:
            label = f'{match.group(1)} - {match.group(2)}'
        else:
            label = file_name.replace('.csv', '')
        sns.lineplot(x='generation', y='max_fitness', data=mean_data, label=f'{label}')

    # Adding title and labels with improved font size and boldness
    plt.title('Mean Max Fitness Across Generations for Different Seeds', fontsize=15)
    plt.ylim(bottom=-2)
    plt.ylim(top=12)
    plt.xlabel('Generation', fontsize=13)
    plt.ylabel('Fitness', fontsize=13)
    plt.legend()

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Improving the layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(os.path.join(os.getcwd(), f'../results/results_{n_individuals}_{n_generations}_{adaptive}/lp_fitness_mean.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # lp_fitness_statistical([1, 2, 3, 4, 5, 6, 7, 8, 9])
    lp_fitness_statistical([9], n_generations=100, n_individuals=50)
