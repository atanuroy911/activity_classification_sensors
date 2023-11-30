import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Load the data
data = pd.read_csv('preprocessed_training_set.csv')  # Replace 'your_dataset.csv' with your dataset file

# Extract features (X) and labels (y) from the data
X = data.iloc[:, :-11].values  # All columns except the last 11
y = data.iloc[:, -11:].values  # Last 11 columns (one-hot encoded labels)

# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)
y = y.to(device)

X = X.view(X.shape[0], 1, X.shape[1])

# # Define the LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]  # Select the last time step output for each sequence
#         output = self.fc(lstm_out)
#         return output

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


# Initialize the model and define hyperparameters
input_size = X.shape[2]  # Number of features
output_size = y.shape[1]  # Number of output classes (11 for one-hot encoded labels)
hidden_size = 128
learning_rate = 0.001
batch_size = 64
num_epochs = 100
num_layers = 1 #number of stacked lstm layers

# Create DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
# model = LSTMModel(input_size, hidden_size, output_size)
model = LSTMModel(output_size, input_size, hidden_size, num_layers, X.shape[1]) #our lstm class)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.to(device)

# Create a SummaryWriter to log training details
writer = SummaryWriter()

# Training the LSTM model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": total_loss / (batch_idx + 1)})
            pbar.update()

            # Log the loss to TensorBoard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss', loss.item(), step)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'trained_lstm_model.pth')

# Close the SummaryWriter
writer.close()
