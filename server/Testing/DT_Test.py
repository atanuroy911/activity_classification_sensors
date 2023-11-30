import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data for testing
print("Loading the test data...")
test_data = pd.read_csv('unprocessed_testing_set.csv')  # Replace 'unprocessed_testing_set.csv' with your test dataset file

# Preprocess categorical columns using the same LabelEncoder as before
print("Preprocessing the test data...")
label_encoder = LabelEncoder()
for col in test_data.columns:
    if test_data[col].dtype == 'object':  # Check if the column contains string values
        test_data[col] = label_encoder.fit_transform(test_data[col])

# Extract features (X_test) and labels (y_test) from the test data
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Load the trained Decision Tree classifier from the pickle file
print("Loading the trained Decision Tree classifier...")
model_filename = 'trained_decision_tree_classifier.pkl'
with open(model_filename, 'rb') as file:
    dt_classifier = pickle.load(file)

# Make predictions on the test set using the trained classifier
print("Making predictions on the test set...")
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy score for classification evaluation
print("Calculating accuracy score...")
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy on test data:", accuracy)
