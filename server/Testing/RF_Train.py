import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the data
print("Loading the data...")
data = pd.read_csv('unprocessed_training_set.csv')  # Replace 'your_dataset.csv' with your dataset file

# Preprocess categorical columns using LabelEncoder
print("Preprocessing the data...")
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':  # Check if the column contains string values
        data[col] = label_encoder.fit_transform(data[col])

# Extract features (X) and labels (y) from the data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values  # Last 11 columns (one-hot encoded labels)

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
print("Initializing the Random Forest classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the Random Forest classifier
print("Training the Random Forest classifier...")
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy score for classification evaluation
print("Calculating accuracy score...")
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save the trained model to a file
model_filename = 'trained_random_forest_classifier.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf_classifier, file)

print("Training and saving process completed.")
