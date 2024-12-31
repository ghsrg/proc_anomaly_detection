import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.file_utils_l import join_path
from src.config.config import NORMAL_GRAPH_PATH, ANOMALOUS_GRAPH_PATH

import networkx as nx
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc

logger = get_logger(__name__)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

def create_model():
    """
    Створює та повертає модель GNN.
    """
    input_dim = 16  # рекомендовані в літературі значення для базових порівнянь
    hidden_dim = 32
    output_dim = 1
    return GNN(input_dim, hidden_dim, output_dim)

def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Готує дані для навчання GNN.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Підготовлені дані у форматі, придатному для GNN.
    """
    data = []

    # Завантаження та підготовка нормальних графів
    # Обробка нормальних графів
    for graph_file in normal_graphs['graph_path']:
        full_path = join_path([NORMAL_GRAPH_PATH, graph_file])
        graph = nx.read_graphml(full_path)
        data.append((graph, 0))  # 0 для нормальних графів

    # Обробка аномальних графів
    for graph_file in anomalous_graphs[anomalous_graphs['params'].str.contains(anomaly_type)]['graph_path']:
        full_path = join_path([ANOMALOUS_GRAPH_PATH, graph_file])
        graph = nx.read_graphml(full_path)
        data.append((graph, 1))  # 1 для аномальних графів

    return data

def train_epoch(model, data, learning_rate=0.001):
    """
    Виконує одну епоху навчання.

    :param model: GNN модель.
    :param data: Дані для навчання.
    :param learning_rate: Рівень навчання.
    :return: Середнє значення втрат за епоху.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    model.train()
    total_loss = 0
    for graph, label in data:
        optimizer.zero_grad()
        output = model(graph)
        loss = loss_fn(output, torch.tensor([label], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data)
    logger.info(f"Середнє значення втрат за епоху: {avg_loss}")
    return avg_loss

def calculate_statistics(model, data):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель.
    :param data: Дані для оцінки.
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for graph, label in data:
            output = model(graph).item()
            predictions.append(output)
            labels.append(label)

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

