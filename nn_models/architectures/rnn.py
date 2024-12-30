# RNN: using LSTM (using PyTorch)
from torch import nn

class RNNModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, output_size=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output from the last time step
        return out