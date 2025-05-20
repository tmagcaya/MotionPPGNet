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
        
        # Calculate the sequence length after conv and pooling
        l1 = n_timesteps - 40 + 1  # Conv1 output length
        l2 = l1 // 4              # Pool1 output length
        l3 = l2 - 40 + 1          # Conv2 output length
        self.seq_len_after_conv = l3 // 4  # Pool2 output length
        print(f"Sequence length after convolutions: {self.seq_len_after_conv}")
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        # Final dense layer for output
        self.dense = nn.Linear(in_features=128, out_features=1)
        
        # Store activations
        self.activations = {}
        
        # Initialize weights using Xavier/Glorot (similar to TensorFlow default)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
      
    def forward(self, x):
        # x shape: (batch_size, n_features, n_timesteps)
        
        # Convolutional layers
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = torch.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        x, (h1, c1) = self.lstm1(x)
        x, (h2, c2) = self.lstm2(x)
        
        # Use the final hidden state
        x = h2.squeeze(0)
        
        # Final dense layer
        x = self.dense(x)
        
        return x.squeeze(-1)  # Remove last dim if it's 1
