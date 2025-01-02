import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH

logger = get_logger(__name__)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=None):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        # Якщо є edge_attr, додати обробку
        if edge_dim is not None:
            self.edge_processor = nn.Linear(edge_dim, hidden_dim)  # Лінійний шар для обробки атрибутів ребер
        else:
            self.edge_processor = None

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Перевірка вхідних даних
        #print( f"x min: {x.min()}, max: {x.max()}, contains nan: {torch.isnan(x).any()}, contains inf: {torch.isinf(x).any()}")

        if edge_attr is not None and self.edge_processor is not None:
            edge_features = self.edge_processor(edge_attr)
            print(
                f"Processed edge_attr: min {edge_features.min()}, max {edge_features.max()}, nan {torch.isnan(edge_features).any()}")

        # Перший графовий шар
        x = torch.relu(self.conv1(x, edge_index))
        #print(f"After conv1: min {x.min()}, max {x.max()}, contains nan: {torch.isnan(x).any()}")

        # Другий графовий шар
        x = torch.relu(self.conv2(x, edge_index))
       # print(f"After conv2: min {x.min()}, max {x.max()}, contains nan: {torch.isnan(x).any()}")

        # Глобальний пулінг
        x = self.global_pool(x, batch)
        #print(f"After pooling: min {x.min()}, max {x.max()}, contains nan: {torch.isnan(x).any()}")

        # Вихідний шар
        x = self.activation(self.fc(x))
       # print(f"After activation: min {x.min()}, max {x.max()}, contains nan: {torch.isnan(x).any()}")

        return x


def transform_graph(graph, label):
    """
    Transforms a NetworkX graph into a PyTorch Geometric Data object.

    :param graph: NetworkX graph.
    :param label: Label of the graph (0 - normal, 1 - anomalous).
    :return: PyTorch Geometric Data object.
    """
    node_map = {node: idx for idx, node in enumerate(graph.nodes())}

    # Node attributes
    numeric_attrs = graph.graph.get("numeric_attrs", [])
    node_features = []
    for _, node_data in graph.nodes(data=True):
        features = [
            float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
            for attr in numeric_attrs
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)
    if torch.isnan(x).any():
        print(f"Node features contain nan: {x}")

    # Edge attributes
    edges = [(node_map[edge[0]], node_map[edge[1]]) for edge in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_attr = None
    if graph.graph.get("edge_attrs"):
        edge_attr = torch.tensor(
            [
                [float(edge_data.get(attr, 0)) if isinstance(edge_data.get(attr), (int, float)) else 0.0
                 for attr in graph.graph["edge_attrs"]]
                for _, _, edge_data in graph.edges(data=True)
            ],
            dtype=torch.float
        )
        if torch.isnan(edge_attr).any():
            print(f"Edge attributes contain nan: {edge_attr}")

    # Global attributes
    global_features = []
    for node, node_data in graph.nodes(data=True):
        if node_data.get("type") == "startEvent":
            global_features = [
                float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                for attr in graph.graph.get("global_attrs", [])
            ]
            break

    global_features = torch.tensor(global_features, dtype=torch.float)
    if torch.isnan(global_features).any():
        print(f"Global features contain nan: {global_features}")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float),
        global_features=global_features
    )

    return data

def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Prepares data for GNN training.

    :param normal_graphs: Registry of normal graphs.
    :param anomalous_graphs: Registry of anomalous graphs.
    :param anomaly_type: Type of anomaly for training.
    :return: List of Data objects for GNN.
    """
    data_list = []

    def infer_graph_attributes(graph):
        """
        Infers numeric, global, and edge attributes for a graph.

        :param graph: NetworkX graph.
        :return: Lists of numeric_attrs, global_attrs, edge_attrs.
        """
        numeric_attrs = set()
        global_attrs = set()
        edge_attrs = set()

        for _, node_data in graph.nodes(data=True):
            for attr, value in node_data.items():
                if isinstance(value, (int, float)):
                    numeric_attrs.add(attr)
                elif node_data.get("type") == "startEvent":
                    global_attrs.add(attr)

        for _, _, edge_data in graph.edges(data=True):
            for attr, value in edge_data.items():
                if isinstance(value, (int, float)):
                    edge_attrs.add(attr)

        return list(numeric_attrs), list(global_attrs), list(edge_attrs)

    for graph_file in normal_graphs["graph_path"]:
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        data = transform_graph(graph, label=0)  # Мітка для нормальних графів
        data.y = torch.tensor([0], dtype=torch.float)  # Додаємо мітку графу
        data_list.append(data)

    for graph_file in anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]["graph_path"]:
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        data = transform_graph(graph, label=1)  # Мітка для аномальних графів
        data.y = torch.tensor([1], dtype=torch.float)  # Додаємо мітку графу
        data_list.append(data)

    return data_list



def train_epoch(model, data, optimizer, batch_size=32, loss_fn=None):
    """
    Виконує одну епоху навчання з розбиттям на пакети (batch).

    :param model: GNN модель.
    :param data: Дані для навчання (список об'єктів Data).
    :param optimizer: Оптимізатор для оновлення ваг моделі.
    :param batch_size: Розмір пакету для навчання.
    :param loss_fn: Функція втрат (опціонально, якщо None, використовується nn.BCELoss).
    :return: Середнє значення втрат за епоху.
    """
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size  # Обчислення кількості пакетів

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        # Підготовка даних для батчу
        x = torch.cat([item.x for item in batch], dim=0)
        edge_index = torch.cat([item.edge_index for item in batch], dim=1)
        batch_tensor = torch.cat(
            [torch.full((item.x.size(0),), idx, dtype=torch.long) for idx, item in enumerate(batch)], dim=0
        )
        y = torch.tensor([item.y.item() for item in batch], dtype=torch.float)  # Одна мітка на граф

        # **Додано для перевірки**
        print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
        # Скидання градієнтів
        optimizer.zero_grad()

        # Прогноз і обчислення втрат
        outputs = model(x, edge_index, batch_tensor).squeeze()  # Передати batch
       # print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")
       # print(f"Labels min: {y.min()}, max: {y.max()}")
        loss = loss_fn(outputs, y)

        # Зворотнє поширення і оновлення ваг
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.info(f"Середнє значення втрат за епоху: {avg_loss}")
    return avg_loss


def calculate_statistics(model, data):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель.
    :param data: Дані для оцінки (список об'єктів Data).
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for graph in data:
            x = graph.x
            edge_index = graph.edge_index
            batch = graph.batch
            label = graph.y.item()  # Мітка графа

            # Передбачення моделі
            output = model(x, edge_index, batch).item()
            predictions.append(output)
            labels.append(label)

    # Розрахунок метрик
    if hasattr(model, "requires_roc_auc") and model.requires_roc_auc:
        roc_auc = calculate_roc_auc(labels, predictions)
    else:
        roc_auc = None

    precision, recall = calculate_precision_recall(labels, predictions)

    stats = {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }

    return stats


def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    return Adam(model.parameters(), lr=learning_rate)
