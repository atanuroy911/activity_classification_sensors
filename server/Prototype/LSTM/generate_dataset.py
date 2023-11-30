import numpy as np
import pandas as pd

# Function to generate a synthetic time series dataset
def generate_time_series_data(n_samples, sequence_length):
    X = np.zeros((n_samples, sequence_length))
    y = np.zeros((n_samples,))
    for i in range(n_samples):
        # Generate a random sequence
        sequence = np.random.random((sequence_length,))
        # Sum all values in the sequence and set it as the target value
        target = np.sum(sequence)
        X[i] = sequence
        y[i] = target
    return X, y

# Define the size of the dataset
n_samples = 10000
sequence_length = 10

# Generate the synthetic time series data
X_data, y_data = generate_time_series_data(n_samples, sequence_length)

# Split the data into training and test sets (80% for training, 20% for testing)
split_ratio = 0.8
split_index = int(n_samples * split_ratio)

X_train, y_train = X_data[:split_index], y_data[:split_index]
X_test, y_test = X_data[split_index:], y_data[split_index:]

column_names = [f"X{i}" for i in range(sequence_length)] + ["y"]

# Save the full combined dataset to a CSV file
full_data = np.column_stack((X_data, y_data))
df_full = pd.DataFrame(full_data, columns=column_names)
df_full.to_csv("full_dataset.csv", index=False)

# Save the training dataset to a CSV file
training_data = np.column_stack((X_train, y_train))
df_train = pd.DataFrame(training_data, columns=column_names)
df_train.to_csv("training_dataset.csv", index=False)

# Save the test dataset to a CSV file
test_data = np.column_stack((X_test, y_test))
df_test = pd.DataFrame(test_data, columns=column_names)
df_test.to_csv("test_dataset.csv", index=False)
