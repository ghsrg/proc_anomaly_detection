from torch import nn
from torch_geometric.nn import GCNConv


class GNNModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=2):
        super(GNNModel, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)  # GCN Layer
        self.layer2 = GCNConv(hidden_dim, output_dim)  # Another GCN Layer
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.layer2(x, edge_index)
        return x
