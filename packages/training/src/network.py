import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class StockIndicatorNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout) 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.lstm1(x) 
        # out: (batch_size, seq_len, hidden_size)
        last_output = out[:, -1, :] 
        # last_output: (batch_size, hidden_size)
        last_output = self.dropout(last_output)  # Apply dropout
        prediction = self.fc(last_output)
        # prediction: (batch_size, output_size)
        return prediction

# Example usage
input_size = 1  # Single input feature
hidden_size = 50 
num_layers = 2  # Number of stacked LSTM layers
output_size = 1  # Predict the next value
dropout_rate = 0.2  # Dropout rate

model = StockIndicatorNN(input_size, hidden_size, num_layers, output_size, dropout_rate)

# --- Training Setup ---
# 1. Prepare Data
# Assuming you have your data in a NumPy array or Pandas DataFrame:
#   - 'train_X': Input sequences (e.g., shape: (num_samples, sequence_length, input_size))
#   - 'train_y': Target values (e.g., shape: (num_samples, output_size))

train_dataset = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 

test_dataset = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).float())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# 2. Define Loss Function and Optimizer
lossFunc = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)) 

# 3. Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = lossFunc(outputs, labels)

        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Deactivate gradient calculation for evaluation
            total_loss = 0.0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = lossFunc(outputs, labels)
                total_loss += loss.item()
            avg_test_loss = total_loss / len(test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {avg_test_loss:.4f}')


# --- After Training ---
# Save the trained model (optional)
torch.save(model.state_dict(), 'stock_indicator_model.pth')