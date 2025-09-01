import os

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt

def find_best_median_column(file_path):
    # Load the file, handle missing values
    df = pd.read_csv(file_path, sep='\t', engine='python')

    # Convert all columns to numeric, coerce errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Compute the median for each column, ignoring NaN
    medians = df.median()

    # Find the column with the lowest median
    best_column = medians.idxmin()
    best_median = medians.min()

    return best_column, best_median


def compare_to_best_column(file_path, alpha=0.05):
    # Load and clean data
    df = pd.read_csv(file_path, sep='\t', engine='python')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')

    # Compute medians
    medians = df.mean()

    # Print all medians
    print("Medians of all columns:")
    for col, median in medians.items():
        print(f"  {col}: {median}")

    # Plot the boxplot
    plt.figure(figsize=(12, 6))
    df.boxplot(grid=True, rot=45)
    plt.title("Boxplot of All Columns")
    plt.ylabel("Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()

    # Find the column with the lowest median
    medians = df.median()
    best_col = medians.idxmin()
    best_values = df[best_col].dropna()

    # Compare each column to the best one
    p_values = {}
    equivalent = []
    non_equivalent = []

    for col in df.columns:
        if col == best_col:
            continue
        values = df[col].dropna()

        # Perform Mann–Whitney U test
        stat, p = mannwhitneyu(best_values, values, alternative='two-sided')
        p_values[col] = p
        if p > alpha:
            equivalent.append(col)
        else:
            non_equivalent.append(col)

    # Print results
    print(f"Best column (lowest median): {best_col}\n")
    print("P-values compared to best column:")
    for col, p in p_values.items():
        print(f"  {col}: p = {p:.4f}")

    print("\nStatistically equivalent columns (p > 0.05):")
    for col in equivalent:
        print(f"  {col}")

    print("\nStatistically non-equivalent columns (p ≤ 0.05):")
    for col in non_equivalent:
        print(f"  {col}")

    return best_col, equivalent, non_equivalent, p_values


# Example usage
selected_seeds = range(30)  # Replace with actual seed values
n_individuals = 101
n_generations = 100
gp_model = "cgp"  # This should be set or determined earlier in your script
llm_model = "31_405B_NEW"  # 31_405B_NEW
flag_after_iteration_diff_policy = 1

# Construct the output file path
file_path = os.path.abspath(os.path.join(os.getcwd(), f"../data/{llm_model}_{gp_model}_{n_individuals}_c4-bp.txt"))

# file_path = '/mnt/data/31_8B_NEW_lgp_50_c4-bp.txt'
best_col, best_val = find_best_median_column(file_path)
print(file_path)
print(f"Best column: {best_col}, Median: {best_val}")

# Example usage
compare_to_best_column(file_path)
