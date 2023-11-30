import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Load the dataset from the CSV file
df = pd.read_csv("test_dataset.csv")
X_test = df.drop(columns=["y"]).values
y_test = df["y"].values

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data to PyTorch tensors and move them to the GPU if available
X_test = torch.tensor(X_test[:, :, np.newaxis], dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# Load the trained LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_dim = X_test.shape[2]  # The number of features (1 in this case)
output_dim = 1
hidden_units = 50

model = LSTMModel(input_dim, hidden_units, output_dim).to(device)
model.load_state_dict(torch.load('trained_lstm_model.pth'))
model.eval()

# Make predictions on test data
batch_size = 32
y_true = []
y_pred = []

# Loop through the test dataset in batches and make predictions
with torch.no_grad():
    for i in tqdm(range(0, len(X_test), batch_size)):
        batch_X, batch_y = X_test[i:i + batch_size], y_test[i:i + batch_size]
        output = model(batch_X).cpu().numpy()
        y_pred.extend(output.squeeze())
        y_true.extend(batch_y.cpu().numpy())

# Convert predictions and true labels to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
