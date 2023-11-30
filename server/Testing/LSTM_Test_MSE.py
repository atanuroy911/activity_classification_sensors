import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Load the test data
test_data = pd.read_csv('preprocessed_testing_set.csv')  # Replace 'preprocessed_test_set.csv' with your test dataset file

# Extract features (X_test) and labels (y_test) from the test data
X_test = test_data.iloc[:, :-11].values  # All columns except the last 11
y_test = test_data.iloc[:, -11:].values  # Last 11 columns (one-hot encoded labels)

# Convert data to torch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

X_test = X_test.view(X_test.shape[0], 1, X_test.shape[1])

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state and cell state on the same device as the model's parameters
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(next(self.parameters()).device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(next(self.parameters()).device)

        # Propagate input through LSTM
        output, _ = self.lstm(x, (h_0, c_0))
        hn = output[:, -1, :]
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

# Initialize the model and load the trained weights
input_size = X_test.shape[2]  # Number of features
output_size = y_test.shape[1]  # Number of output classes (11 for one-hot encoded labels)
hidden_size = 128
num_layers = 1 #number of stacked lstm layers


model = LSTMModel(output_size, input_size, hidden_size, num_layers, X_test.shape[1])
model.load_state_dict(torch.load('trained_lstm_model.pth'))
model.eval()

# Make predictions on test data
y_pred_list = []
with torch.no_grad():
    for inputs in tqdm(X_test):  # Wrap the loop with tqdm for progress visualization
        inputs = inputs.unsqueeze(0)  # Add a batch dimension
        y_pred = model(inputs)
        y_pred_list.append(y_pred.squeeze(0).cpu().numpy())

y_pred = np.array(y_pred_list)

# Convert predictions and ground truth to numpy arrays
y_test = y_test.numpy()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) score:", r2)
