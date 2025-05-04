import torch
import torch.nn as nn
import torch.nn.functional as F

class motionPPGNet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(motionPPGNet, self).__init__()
        # Conv1D blocks
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=40)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=40)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        # Final dense layer for output
        self.dense = nn.Linear(in_features=128, out_features=1)

       
    def forward(self, x):
        # x shape: (batch_size, n_features, n_timesteps)
        
        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        
        # Use the last output from LSTM for dense layer
        # For the second LSTM, we take the final hidden state
        # hidden shape: (1, batch_size, hidden_size)
        x = hidden.squeeze(0)
        
        # Final dense layer
        x = self.dense(x)
        
        return x
