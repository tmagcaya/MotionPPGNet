import torch
import torch.nn as nn
import torch.nn.functional as F

class motionPPGNet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(motionPPGNet, self).__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=40)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=40)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.dropout2 = nn.Dropout(0.1)

        self.lstm1 = nn.LSTM(50, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


