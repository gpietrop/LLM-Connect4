import pandas as pd

n_individuals = 50
n_generations = 100
adaptive = False
file_path = f'../results/results_{n_individuals}_{n_generations}_{adaptive}/boxplot_convergence.csv'
data = pd.read_csv(file_path, delimiter=',')
# print(data.columns)


# Grouping the data by 'File' and collecting the 'First Nonzero Index After Threshold' values
grouped_data = data.groupby('File')['First Nonzero Index After Threshold'].apply(list)

# Ensure each list has 20 elements, filling missing values with NaN
grouped_data = grouped_data.apply(lambda x: x + [float('NaN')] * (20 - len(x)) if len(x) < 20 else x)

# Converting the lists into a DataFrame
formatted_data = pd.DataFrame(grouped_data.tolist(), index=grouped_data.index).reset_index()

# Renaming the columns to 'run_1', 'run_2', ..., 'run_20'
formatted_data.columns = ['File'] + [f'run_{i+1}' for i in range(20)]


# Saving the processed data to a new CSV file
output_file_path = f'../results/results_{n_individuals}_{n_generations}_{adaptive}/c4.csv'
formatted_data.to_csv(output_file_path, sep='\t', index=False)

# output_file_path

file_path = f'../results/results_{n_individuals}_{n_generations}_{adaptive}/c4.csv'
data = pd.read_csv(file_path, delimiter=',')

# Splitting the data into separate columns
data = data['File\trun_1\trun_2\trun_3\trun_4\trun_5\trun_6\trun_7\trun_8\trun_9\trun_10\trun_11\trun_12\trun_13\trun_14\trun_15\trun_16\trun_17\trun_18\trun_19\trun_20'].str.split('\t', expand=True)

# Rename the first column as 'Run'
data.rename(columns={0: 'Run'}, inplace=True)

# Remove the '.csv' from the run names
data['Run'] = data['Run'].str.replace('.csv', '')

# Transpose the data
data_transposed = data.set_index('Run').T

# Reset index to make it a column again and rename it appropriately
data_transposed.reset_index(inplace=True)
data_transposed.rename(columns={'index': 'Trial'}, inplace=True)

if 'Trial' in data_transposed.columns:
    data_transposed = data_transposed.drop(columns=['Trial'])

numeric_value_counts = data_transposed.apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().sum())
# data_transposed.loc['count'] = numeric_value_counts
# data_transposed["count"] = numeric_value_counts
sorted_columns_by_row_count = numeric_value_counts.sort_values(ascending=False).index
data_sorted = data_transposed[sorted_columns_by_row_count]

# Display the first few rows of the reformatted data
output_file_path = f'../results/results_{n_individuals}_{n_generations}_{adaptive}/c4-bp.txt'
data_sorted.to_csv(output_file_path, sep='\t', index=False)

