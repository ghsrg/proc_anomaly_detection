import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch.nn.functional import relu
from src.utils.logger import get_logger
from src.utils.file_utils import join_path, load_graph
from tqdm import tqdm
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH

logger = get_logger(__name__)


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None):
        super(CNN, self).__init__()

        # Шари для обробки вузлів і зв'язків
        self.conv1 = nn.Conv1d(in_channels=input_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Шар для обробки документних атрибутів
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Остаточний шар для класифікації
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Активації
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, node_features, edge_features, doc_features):
        """
        Прямий прохід через модель.

        :param node_features: Тензор вузлів [batch_size, num_nodes, node_dim].
        :param edge_features: Тензор зв'язків [batch_size, num_edges, edge_dim].
        :param doc_features: Тензор документа [batch_size, doc_dim].
        :param batch_size: Інформація про пакети (необов’язково).
        :return: Класифікація [batch_size, output_dim].
        """
        # Узгодження кількості вузлів і зв'язків
        max_len = max(node_features.size(1), edge_features.size(1))

        node_features_padded = torch.cat(
            [node_features, torch.zeros(node_features.size(0), max_len - node_features.size(1), node_features.size(2),
                                        device=node_features.device)],
            dim=1
        )

        edge_features_padded = torch.cat(
            [edge_features, torch.zeros(edge_features.size(0), max_len - edge_features.size(1), edge_features.size(2),
                                        device=edge_features.device)],
            dim=1
        )

        # Узгодження кількості атрибутів
        max_dim = max(node_features_padded.size(2), edge_features_padded.size(2))
        node_features_padded = torch.cat(
            [node_features_padded, torch.zeros(node_features_padded.size(0), node_features_padded.size(1),
                                               max_dim - node_features_padded.size(2), device=node_features.device)],
            dim=2
        )
        edge_features_padded = torch.cat(
            [edge_features_padded, torch.zeros(edge_features_padded.size(0), edge_features_padded.size(1),
                                               max_dim - edge_features_padded.size(2), device=edge_features.device)],
            dim=2
        )

        # Об'єднання вузлів і зв'язків
        x = torch.cat([node_features_padded, edge_features_padded], dim=2)  # [batch_size, max_len, max_dim]
        x = x.permute(0, 2, 1)  # [batch_size, max_dim, max_len]

        # Подальша обробка через CNN
        x = self.activation(self.conv1(x))
        x = self.pool(self.activation(self.conv2(x))).squeeze(2)  # [batch_size, hidden_dim]

        # Обробка документних атрибутів
        doc_emb = self.activation(self.doc_fc(doc_features))  # [batch_size, hidden_dim]

        # Об'єднання всіх ознак
        combined = torch.cat([x, doc_emb], dim=1)  # [batch_size, hidden_dim * 2]
        output = self.fc(combined)  # [batch_size, output_dim]
        return self.final_activation(output)


def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для CNN.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для CNN, розмірність входу (input_dim), розмірність документа (doc_dim).
    """
    data_list = []
    max_nodes = 0
    max_edges = 0

    def transform_graph(graph, node_attrs, edge_attrs, max_nodes, max_edges):
        """
        Перетворення графу на тензори вузлів і зв'язків із доповненням.

        :param graph: Граф із вузлами та зв'язками.
        :param node_attrs: Список атрибутів вузлів.
        :param edge_attrs: Список атрибутів зв'язків.
        :param max_nodes: Максимальна кількість вузлів.
        :param max_edges: Максимальна кількість зв'язків.
        :return: Тензори вузлів і зв'язків із доповненням.
        """
        node_features = []
        for _, node_data in graph.nodes(data=True):
            features = [float(node_data.get(attr, 0.0)) for attr in node_attrs]
            node_features.append(features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_features = []
        for _, _, edge_data in graph.edges(data=True):
            features = [float(edge_data.get(attr, 0.0)) for attr in edge_attrs]
            edge_features.append(features)
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # Доповнення вузлів і зв'язків
        padded_node_features = torch.zeros(max_nodes, node_features.size(1))
        padded_node_features[:node_features.size(0), :] = node_features

        padded_edge_features = torch.zeros(max_edges, edge_features.size(1))
        padded_edge_features[:edge_features.size(0), :] = edge_features

        return padded_node_features, padded_edge_features

    def transform_doc(doc_info, selected_doc_attrs):
        """
        Перетворення атрибутів документа на тензор.

        :param doc_info: Інформація про документ.
        :param selected_doc_attrs: Вибрані атрибути документа.
        :return: Тензор атрибутів документа.
        """
        return torch.tensor([doc_info.get(attr, 0.0) for attr in selected_doc_attrs], dtype=torch.float)

    selected_node_attrs = ["type", "DURATION_", "START_TIME_", "END_TIME_", "active_executions", "SEQUENCE_COUNTER_",
                           "overdue_work", "duration_work"]
    selected_edge_attrs = ["DURATION_", "taskaction_code", "overdue_work"]
    selected_doc_attrs = ["PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate", "CategoryL1", "CategoryL2",
                          "CategoryL3", "ClassSSD", "Company_SO"]

    input_dim = len(selected_node_attrs)  # Розмірність вузлів
    edge_dim = len(selected_edge_attrs)  # Розмірність зв'язків
    doc_dim = len(selected_doc_attrs)  # Розмірність атрибутів документа

    # Обробка нормальних графів
    #for idx, row in normal_graphs.iterrows():
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка нормальних графів", total=len(normal_graphs)):
        graph_file = row["graph_path"]  # Шлях до файлу графу
        doc_info = row.get("doc_info", {})  # Інформація про документ
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])  # Повний шлях до графа
        graph = load_graph(full_path)  # Завантаження графу

        max_nodes = max(max_nodes, len(graph.nodes))
        max_edges = max(max_edges, len(graph.edges))

        node_features, edge_features = transform_graph(graph, selected_node_attrs, selected_edge_attrs, max_nodes, max_edges)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        data = {
            "node_features": node_features,
            "edge_features": edge_features,
            "doc_features": doc_features,
            "label": torch.tensor([0], dtype=torch.float)  # Нормальні дані
        }
        data_list.append(data)

    # Обробка аномальних графів
    #for idx, row in anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)].iterrows():
    filtered_anomalous_graphs = anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]
    for idx, row in tqdm(filtered_anomalous_graphs.iterrows(), desc="Обробка аномальних графів",
                         total=len(filtered_anomalous_graphs)):
        graph_file = row["graph_path"]  # Шлях до файлу графу
        doc_info = row.get("doc_info", {})  # Інформація про документ
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])  # Повний шлях до графа
        graph = load_graph(full_path)  # Завантаження графу

        max_nodes = max(max_nodes, len(graph.nodes))
        max_edges = max(max_edges, len(graph.edges))

        node_features, edge_features = transform_graph(graph, selected_node_attrs, selected_edge_attrs, max_nodes, max_edges)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        data = {
            "node_features": node_features,
            "edge_features": edge_features,
            "doc_features": doc_features,
            "label": torch.tensor([1], dtype=torch.float)  # Аномальні дані
        }
        data_list.append(data)

    return data_list, input_dim, doc_dim

def train_epoch(model, train_data, optimizer, batch_size):
    """
    Тренування моделі на одному епоху для CNN.

    :param model: Нейронна мережа (CNN).
    :param train_data: Дані для тренування (список словників).
    :param optimizer: Оптимізатор.
    :param batch_size: Розмір батчу.
    :return: Середня втрата за епоху.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()  # Функція втрат для бінарної класифікації

    # Поділ даних на батчі
    #for i in range(0, len(train_data), batch_size):
    for i in tqdm(range(0, len(train_data), batch_size), desc="Поділ на батчі", unit="батч", leave=False, dynamic_ncols=True, mininterval=5):

        batch = train_data[i:i + batch_size]

        # Формування батчу
        node_features = [data["node_features"] for data in batch]  # Список вузлових атрибутів
        edge_features = [data["edge_features"] for data in batch]  # Список зв'язкових атрибутів
        doc_features = torch.stack([data["doc_features"] for data in batch])  # [batch_size, doc_dim]
        labels = torch.stack([data["label"] for data in batch])  # [batch_size, 1]

        # Вирівнювання розмірів вузлів через пакування (padding)
        max_nodes = max([nf.size(0) for nf in node_features])
        max_edges = max([ef.size(0) for ef in edge_features])

        node_features_padded = torch.stack([
            torch.cat([nf, torch.zeros(max_nodes - nf.size(0), nf.size(1))], dim=0)
            for nf in node_features
        ])  # [batch_size, max_nodes, input_dim]

        edge_features_padded = torch.stack([
            torch.cat([ef, torch.zeros(max_edges - ef.size(0), ef.size(1))], dim=0)
            for ef in edge_features
        ])  # [batch_size, max_edges, edge_dim]

        # Перенос на пристрій (GPU/CPU)
        device = next(model.parameters()).device
        node_features_padded = node_features_padded.to(device)
        edge_features_padded = edge_features_padded.to(device)
        doc_features = doc_features.to(device)
        labels = labels.to(device)

        # Прямий прохід
        optimizer.zero_grad()
        outputs = model(node_features_padded, edge_features_padded, doc_features)  # [batch_size, output_dim]

        # Розрахунок втрат
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Зворотний прохід і оновлення ваг
        loss.backward()
        optimizer.step()

    # Повернення середньої втрати за епоху
    return total_loss / len(train_data)




def calculate_statistics(model, data):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель.
    :param data: Дані для оцінки (список словників).
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for item in data:
            node_features = item["node_features"]  # Вузлові атрибути
            edge_features = item["edge_features"]  # Атрибути зв'язків
            doc_features = item["doc_features"]  # Документні атрибути
            label = item["label"].item()  # Мітка графа

            # Перенесення даних на пристрій
            device = next(model.parameters()).device
            node_features = node_features.to(device)
            edge_features = edge_features.to(device)
            doc_features = doc_features.to(device)

            # Передбачення моделі
            output = model(node_features.unsqueeze(0), edge_features.unsqueeze(0), doc_features.unsqueeze(0)).item()
            predictions.append(output)
            labels.append(label)

    # Розрахунок метрик
    roc_auc = calculate_roc_auc(labels, predictions)
    precision, recall = calculate_precision_recall(labels, predictions)
    f1_score = calculate_f1_score(labels, predictions)

    stats = {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "f1_score": f1_score
    }

    return stats


def create_optimizer(model, learning_rate=0.001):
    return Adam(model.parameters(), lr=learning_rate)
