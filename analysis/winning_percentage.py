import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os


def histogram_final_percentage(seeds):
    # Initialize a dictionary to store the count of final percentages greater than 0.01
    final_percentage_counts = {}

    # Iterate over each seed
    for my_seed in seeds:
        directory_path = os.path.join(os.getcwd(), f'../results/connect4_trial_{my_seed}')
        metrics_files = glob.glob(os.path.join(directory_path, 'res_*.csv'))

        # Loop through each file
        for file_path in metrics_files:
            file_name = os.path.basename(file_path)
            data = pd.read_csv(file_path)
            final_percentage = data['percentage'].iloc[-1]

            # Count if final percentage is greater than 0.01
            if file_name not in final_percentage_counts:
                final_percentage_counts[file_name] = 0
            if final_percentage > 0.01:
                final_percentage_counts[file_name] += 1

    # Convert counts to a DataFrame
    histogram_data = pd.DataFrame(list(final_percentage_counts.items()), columns=['File', 'Count > 0.01'])

    # Plotting
    sns.set(style="whitegrid", context="talk", font_scale=1)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='File', y='Count > 0.01', data=histogram_data)
    plt.xticks(rotation=45)
    plt.title('Count of Final Success Percentage > 0.01 for Different Evaluation Strategies', fontsize=15)
    plt.xlabel('Evaluation Strategy', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.close()


# Example usage
seeds = [1, 2, 3, 4, 5, 6, 7]  # Replace with your actual seeds
histogram_final_percentage(seeds)
