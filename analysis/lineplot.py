import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Setting up the aesthetics for the plot
sns.set(style="whitegrid")


def lp_fitness(my_seed):
    file_path = os.getcwd() + f'/../results/connect4_trial_{my_seed}/metrics.csv'
    data = pd.read_csv(file_path)

    # Display the first few rows of the dataframe to understand its structure
    data.head()

    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='generation', y='mean_fitness', data=data, label='Mean Fitness', color='blue')
    sns.lineplot(x='generation', y='max_fitness', data=data, label='Max Fitness', color='green')

    # Adding title and labels
    plt.title('Mean Fitness and Max Fitness Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Display the plot
    plt.savefig(os.getcwd() + f'/../results/connect4_trial_{my_seed}/lp_fitness.png')

    plt.show()
    plt.close()


if __name__ == '__main__':
    lp_fitness(2)
    # lp_fitness(2)
