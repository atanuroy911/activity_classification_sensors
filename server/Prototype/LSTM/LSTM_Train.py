import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the dataset from the CSV file
df = pd.read_csv("training_dataset.csv")
X_train = df.drop(columns=["y"]).values
y_train = df["y"].values

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data to PyTorch tensors and move them to the GPU if available
X_train = torch.tensor(X_train[:, :, np.newaxis], dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Define hyperparameters
input_dim = X_train.shape[2]  # The number of features (1 in this case)
output_dim = 1
hidden_units = 50
epochs = 50
batch_size = 32

# Create and train the LSTM model
model = LSTMModel(input_dim, hidden_units, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_X, batch_y = X_train[i:i + batch_size], y_train[i:i + batch_size]
        output = model(batch_X)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_lstm_model.pth')
