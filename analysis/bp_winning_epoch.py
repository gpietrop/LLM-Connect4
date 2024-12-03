import os
import math
import re
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_first_nonzero_percentage_with_median(llm_model, seeds, gp_model, n_generations=100, n_individuals=25, adaptive=False):
    # Initialize a list to store the first index where percentage > 0 after a certain threshold
    nonzero_indices = []
    # Dictionary to count valid nonzero indices for each file
    valid_nonzero_counts = {}
    # Dictionary to track total occurrences of each file
    file_totals = {}

    # Iterate over each seed
    for my_seed in seeds:
        if gp_model == "lgp":
            directory_path = os.path.join(os.getcwd(),
                                          f'../results/{llm_model}/results_{n_individuals}_{n_generations}_{adaptive}/connect4_trial_{my_seed}')
        else:
            directory_path = os.path.join(os.getcwd(),
                                          f'../results_cgp/{llm_model}/results_{n_individuals}_{n_generations}_{adaptive}/connect4_trial_{my_seed}')
        metrics_files = glob.glob(os.path.join(directory_path, 'res_*.csv'))

        # print(f"Seed: {my_seed}, Found files: {metrics_files}")  # Debug: Check found files

        # Count occurrences of each file
        for file_path in metrics_files:
            file_name = os.path.basename(file_path)
            file_totals[file_name] = file_totals.get(file_name, 0) + 1

            data = pd.read_csv(file_path)

            # Extract the first two numbers from the file name and sum them
            numbers_in_filename = [int(num) for num in re.findall(r'\d+', file_name)]
            threshold = sum(numbers_in_filename[:2]) + 1

            # Find the first non-zero index after the threshold
            nonzero_index = data.index[(data['percentage'] > 0.0) & (data.index > threshold)].min()
            if math.isnan(nonzero_index):
                nonzero_index = None

            if nonzero_index is not None:
                # Store the file name and the first nonzero index after the threshold
                nonzero_indices.append((file_name, nonzero_index))
                # Count the valid nonzero indices
                valid_nonzero_counts[file_name] = valid_nonzero_counts.get(file_name, 0) + 1

    # Convert the list to a DataFrame if there are valid nonzero indices
    if nonzero_indices:
        boxplot_data = pd.DataFrame(nonzero_indices, columns=['File', 'First Nonzero Index After Threshold'])
    else:
        print("No valid nonzero indices found. Skipping plot generation.")
        return  # Exit the function if no data is available for plotting

    # Plotting
    sns.set(style="whitegrid", context="talk", font_scale=1)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='File', y='First Nonzero Index After Threshold', data=boxplot_data)
    plt.xticks(rotation=45)

    # Get the maximum y-value of the boxplot to position the text above the plot area
    y_max = boxplot_data['First Nonzero Index After Threshold'].max()
    y_limit = ax.get_ylim()[1]
    y_text_offset = (y_limit - y_max) * 0.1  # Offset the text by 10% of the range above the max value

    # Annotate each box with the count of valid nonzero indices and their ratio
    # Annotate each box with the count of valid nonzero indices and their ratio
    for i, file in enumerate(boxplot_data['File'].unique()):
        count = valid_nonzero_counts.get(file, 0)
        total = file_totals.get(file, 0)  # Total number of occurrences for this file
        ratio = count / total if total > 0 else 0
        annotation = f'{count}/{total}\n({ratio:.2%})'  # Format as 'count/total (percentage%)'

        # Get the median value for this boxplot to position the annotation
        median_value = boxplot_data[boxplot_data['File'] == file]['First Nonzero Index After Threshold'].median()
        # Adjust the y position above the median
        y = median_value + y_text_offset
        ax.text(i, y, annotation, horizontalalignment='center', size='small', color='black', weight='semibold')

    # Adjust the y-limit to accommodate the text annotations
    ax.set_ylim(0, y_limit + y_text_offset * 4)  # Increase offset for annotations

    # Adjust plot size and spacing
    plt.gcf().set_size_inches(12, 8)  # Increase figure size
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add spacing for the title

    plt.title('Distribution of First Epoch of Winning\nAgainst Difficult Policy', fontsize=14, pad=20)
    plt.xlabel('Evaluation Strategy', fontsize=13)
    plt.ylabel('First Nonzero Index After Threshold', fontsize=13)
    plt.tight_layout()

    plt.savefig(
        os.path.join(os.getcwd(), f'../results/bp_{gp_model}_{llm_model}_{n_generations}_{n_individuals}.png'))
    plt.show()
    plt.close()


# Example usage
seeds = range(30)  # Replace with actual seed values
llm_model = "original"  # "31_405B_NEW"
gp_model = "cgp"
boxplot_first_nonzero_percentage_with_median(llm_model, seeds, gp_model, n_generations=100, n_individuals=25)
