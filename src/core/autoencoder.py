import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None):
        """
        Ініціалізація Autoencoder.

        :param input_dim: Вхідна розмірність вузлів і зв'язків.
        :param hidden_dim: Розмір прихованого шару.
        :param output_dim: Розмір вихідного шару (класифікація).
        :param doc_dim: Розмір документних ознак.
        :param edge_dim: Розмір ознак зв'язків (може бути None).
        """
        super(Autoencoder, self).__init__()

        # Енкодер для вузлів і зв'язків
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Декодер для вузлів і зв'язків
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Енкодер для документних атрибутів
        self.doc_encoder = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Остаточний шар для об'єднання та класифікації
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, sequence, doc_features):
        """
        Прямий прохід через модель.

        :param sequence: Тензор послідовності вузлів і зв'язків [batch_size, max_seq_length, input_dim].
        :param doc_features: Тензор документа [batch_size, doc_dim].
        :return: Реконструйовані дані та класифікація [batch_size, output_dim].
        """
        # Енкодинг послідовності
        batch_size, seq_length, _ = sequence.size()
        sequence_flat = sequence.view(batch_size * seq_length, -1)
        encoded_seq = self.encoder(sequence_flat)  # [batch_size * seq_length, hidden_dim]
        reconstructed_seq = self.decoder(encoded_seq)  # [batch_size * seq_length, input_dim]
        reconstructed_seq = reconstructed_seq.view(batch_size, seq_length, -1)  # [batch_size, seq_length, input_dim]

        # Агрегація (наприклад, середнє значення) для фіксованого розміру
        aggregated_seq = encoded_seq.view(batch_size, seq_length, -1).mean(dim=1)  # [batch_size, hidden_dim]

        # Енкодинг документних атрибутів
        encoded_doc = self.doc_encoder(doc_features)  # [batch_size, hidden_dim]

        # Об'єднання латентних представлень
        combined_latent = torch.cat([aggregated_seq, encoded_doc], dim=1)  # [batch_size, hidden_dim * 2]

        # Вихідна класифікація
        output = self.final_fc(combined_latent)  # [batch_size, output_dim]

        return reconstructed_seq, output

def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для Autoencoder.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для Autoencoder, розмірність входу (input_dim), розмірність документа (doc_dim).
    """
    data_list = []
    max_sequence_length = 0

    def transform_graph(graph, node_attrs, edge_attrs):
        """
        Перетворення графу на послідовність із вузлів і зв'язків.

        :param graph: Граф із вузлами та зв'язками.
        :param node_attrs: Список атрибутів вузлів.
        :param edge_attrs: Список атрибутів зв'язків.
        :return: Тензор послідовності [sequence_length, input_dim].
        """
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1].get("SEQUENCE_COUNTER_", 0))
        sequence = []
        input_dim = len(node_attrs) + len(edge_attrs)

        for node, node_data in sorted_nodes:
            node_features = [float(node_data.get(attr, 0.0)) for attr in node_attrs]
            node_features += [0.0] * (input_dim - len(node_features))
            sequence.append(node_features)

            for _, target, edge_data in graph.edges(node, data=True):
                edge_features = [float(edge_data.get(attr, 0.0)) for attr in edge_attrs]
                edge_features += [0.0] * (input_dim - len(edge_features))
                sequence.append(edge_features)

                target_data = graph.nodes[target]
                target_features = [float(target_data.get(attr, 0.0)) for attr in node_attrs]
                target_features += [0.0] * (input_dim - len(target_features))
                sequence.append(target_features)

        sequence_tensor = torch.tensor(sequence, dtype=torch.float)
        return sequence_tensor

    def transform_doc(doc_info, selected_doc_attrs):
        """
        Перетворення атрибутів документа на тензор.

        :param doc_info: Інформація про документ.
        :param selected_doc_attrs: Вибрані атрибути документа.
        :return: Тензор атрибутів документа.
        """
        return torch.tensor([doc_info.get(attr, 0.0) for attr in selected_doc_attrs], dtype=torch.float)

    selected_node_attrs = ["type", "DURATION_", "START_TIME_", "END_TIME_", "active_executions", "SEQUENCE_COUNTER_", "overdue_work", "duration_work"]
    selected_edge_attrs = ["DURATION_", "taskaction_code", "overdue_work"]
    selected_doc_attrs = ["PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate", "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"]

    input_dim = len(selected_node_attrs) + len(selected_edge_attrs)
    doc_dim = len(selected_doc_attrs)

    for idx, row in normal_graphs.iterrows():
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        max_sequence_length = max(max_sequence_length, sequence_tensor.size(0))

        data = {
            "sequence": sequence_tensor,
            "doc_features": doc_features,
            "label": torch.tensor([0], dtype=torch.float)  # Нормальні дані
        }
        data_list.append(data)

    for idx, row in anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)].iterrows():
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        max_sequence_length = max(max_sequence_length, sequence_tensor.size(0))

        data = {
            "sequence": sequence_tensor,
            "doc_features": doc_features,
            "label": torch.tensor([1], dtype=torch.float)  # Аномальні дані
        }
        data_list.append(data)

    for data in data_list:
        padding = torch.zeros(max_sequence_length - data["sequence"].size(0), data["sequence"].size(1))
        data["sequence"] = torch.cat([data["sequence"], padding], dim=0)

    return data_list, input_dim, doc_dim

def train_epoch(model, data, optimizer, batch_size=24, loss_fn=None):
    """
    Виконує одну епоху навчання для моделі Autoencoder.

    :param model: Модель Autoencoder.
    :param data: Дані для навчання (список словників).
    :param optimizer: Оптимізатор.
    :param batch_size: Розмір батчу.
    :param loss_fn: Функція втрат.
    :return: Середнє значення втрат за епоху.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.train()
    total_loss = 0
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    for batch_idx in range(num_batches):
        batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        max_seq_length = max(item["sequence"].size(0) for item in batch)
        padded_sequences = torch.stack([
            torch.cat([item["sequence"], torch.zeros(max_seq_length - item["sequence"].size(0), item["sequence"].size(1))])
            for item in batch
        ]).to(next(model.parameters()).device)

        doc_features = torch.stack([item["doc_features"] for item in batch]).to(next(model.parameters()).device)

        optimizer.zero_grad()

        reconstructed_seq, outputs = model(padded_sequences, doc_features)

        loss = loss_fn(reconstructed_seq, padded_sequences)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss

def calculate_statistics(model, data):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель Autoencoder.
    :param data: Дані для оцінки (список словників).
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for item in data:
            sequence = item["sequence"].unsqueeze(0).to(next(model.parameters()).device)
            doc_features = item["doc_features"].unsqueeze(0).to(next(model.parameters()).device)
            label = item["label"].item()

            _, output = model(sequence, doc_features)
            predictions.append(output.item())
            labels.append(label)

    roc_auc = calculate_roc_auc(labels, predictions)
    precision, recall = calculate_precision_recall(labels, predictions)
    f1_score = calculate_f1_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "f1_score": f1_score
    }

def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    return Adam(model.parameters(), lr=learning_rate)
