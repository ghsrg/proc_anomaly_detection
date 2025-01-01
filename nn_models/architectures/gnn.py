from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Використовуємо GCNConv для обробки edge_index
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Перший графовий шар
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Другий графовий шар
        self.edge_processor = nn.Linear(1, hidden_dim)  # Для edge_attr (якщо вони числові)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward метод для GNN.

        :param x: Тензор вузлів (node features) [num_nodes, input_dim].
        :param edge_index: Тензор зв’язків (edges) [2, num_edges].
        :param edge_attr: Тензор атрибутів ребер (edge features) [num_edges, feature_dim].
        :return: Вихідний тензор для кожного вузла [num_nodes, output_dim].
        """
        # Якщо edge_attr є, обробляємо його
        if edge_attr is not None:
            edge_features = self.edge_processor(edge_attr)
        else:
            edge_features = None

        # Перший графовий шар
        x = self.activation(self.conv1(x, edge_index, edge_features))

        # Другий графовий шар
        x = self.conv2(x, edge_index, edge_features)

        return x



