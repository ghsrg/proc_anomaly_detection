import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score, calculate_auprc, calculate_adr, calculate_far, calculate_fpr, calculate_fnr
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH
from sklearn.metrics import confusion_matrix

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_heads=None, num_layers=2, max_seq_length=1000):
        """
        Ініціалізація Transformer-моделі.

        :param input_dim: Вхідна розмірність для вузлів і зв'язків.
        :param hidden_dim: Розмір прихованого шару.
        :param output_dim: Кількість вихідних ознак.
        :param doc_dim: Розмірність атрибутів документа.
        :param num_heads: Кількість голов для Multi-Head Attention (за замовчуванням дорівнює input_dim).
        :param num_layers: Кількість шарів Transformer Encoder.
        :param max_seq_length: Максимальна довжина послідовності.
        """
        super(Transformer, self).__init__()  # Ініціалізація базового класу nn.Module

        # Вибір кількості голів
        if num_heads is None:
            num_heads = input_dim
        if input_dim % num_heads != 0:
            raise ValueError(f"`input_dim` ({input_dim}) має бути кратним `num_heads` ({num_heads}).")

        # Ініціалізація модулів
        self.positional_encoding = PositionalEncoding(input_dim, max_seq_length)
        self.d_model = input_dim

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)


        # Обробка атрибутів документа
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Об'єднання ознак і класифікація
        combined_dim = self.d_model + hidden_dim
        self.fc = nn.Linear(combined_dim, output_dim)

        # Активації
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, sequence, doc_features, mask=None):
        """
        Прямий прохід через Transformer.

        :param sequence: Послідовність [batch_size, max_seq_length, input_dim].
        :param doc_features: Атрибути документа [batch_size, doc_dim].
        :param mask: Маска для послідовності [batch_size, max_seq_length].
        :return: Вихідний результат [batch_size, output_dim].
        """

        device = next(self.parameters()).device
        sequence = sequence.to(device)
        doc_features = doc_features.to(device)
        if mask is not None:
            mask = mask.to(device)
        # Позиційне кодування
        sequence = self.positional_encoding(sequence)

        # Transformer Encoder
        encoded_sequence = self.transformer_encoder(sequence.transpose(0, 1), src_key_padding_mask=mask)
        encoded_sequence = encoded_sequence.transpose(0, 1)  # Повертаємо до [batch_size, max_seq_length, d_model]

        # Обробка документних атрибутів
        doc_emb = self.activation(self.doc_fc(doc_features))  # [batch_size, hidden_dim]

        # Агрегація послідовності
        sequence_emb = encoded_sequence.mean(dim=1)  # [batch_size, d_model]

        # Об'єднання
        combined = torch.cat([sequence_emb, doc_emb], dim=1)  # [batch_size, d_model + hidden_dim]

        # Класифікація
        output = self.fc(combined)  # [batch_size, output_dim]
        #return self.final_activation(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model  # Збереження d_model для перевірки
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + (d_model % 2)])
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(2) != self.d_model:
            raise ValueError(f"Розмірність вхідного тензора ({x.size(2)}) не збігається з d_model ({self.d_model}).")
        return x + self.pe[:, :x.size(1), :].to(x.device)



def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для Transformer із врахуванням порядку вузлів і зв'язків.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для Transformer, розмірність входу (input_dim), розмірність документа (doc_dim).
    """
    data_list = []
    max_sequence_length = 0

    def transform_graph(graph, node_attrs, edge_attrs, input_dim):
        """
        Перетворення графу на послідовність із вирівнюванням атрибутів до input_dim.

        :param graph: Граф із вузлами та зв'язками.
        :param node_attrs: Список атрибутів вузлів.
        :param edge_attrs: Список атрибутів зв'язків.
        :param input_dim: Загальна розмірність входу (вузли + зв'язки).
        :return: Тензор послідовності [sequence_length, input_dim].
        """
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1].get("SEQUENCE_COUNTER_", 0))
        sequence = []

        for node, node_data in sorted_nodes:
            # Формування атрибутів вузла
            node_features = [float(node_data.get(attr, 0.0)) for attr in node_attrs]
            node_features += [0.0] * (input_dim - len(node_features))  # Доповнення до input_dim
            sequence.append(node_features)

            for _, target, edge_data in graph.edges(node, data=True):
                # Формування атрибутів зв'язку
                edge_features = [float(edge_data.get(attr, 0.0)) for attr in edge_attrs]
                edge_features += [0.0] * (input_dim - len(edge_features))  # Доповнення до input_dim
                sequence.append(edge_features)

        return torch.tensor(sequence, dtype=torch.float)

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

    input_dim = len(selected_node_attrs) + len(selected_edge_attrs)  # Розмірність вузлів і зв'язків разом
    doc_dim = len(selected_doc_attrs)  # Розмірність атрибутів документа

    # Обробка нормальних графів
    #for idx, row in normal_graphs.iterrows():
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка нормальних графів", total=len(normal_graphs)):

        graph_file = row["graph_path"]  # Шлях до файлу графу
        doc_info = row.get("doc_info", {})  # Інформація про документ
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])  # Повний шлях до графа
        graph = load_graph(full_path)  # Завантаження графу

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs, input_dim)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        max_sequence_length = max(max_sequence_length, sequence_tensor.size(0))

        data = {
            "sequence": sequence_tensor,
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

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs, input_dim)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        max_sequence_length = max(max_sequence_length, sequence_tensor.size(0))

        data = {
            "sequence": sequence_tensor,
            "doc_features": doc_features,
            "label": torch.tensor([1], dtype=torch.float)  # Аномальні дані
        }
        data_list.append(data)

    # Доповнення всіх послідовностей до max_sequence_length
    for data in data_list:
        padding = torch.zeros(max_sequence_length - data["sequence"].size(0), data["sequence"].size(1))
        data["sequence"] = torch.cat([data["sequence"], padding], dim=0)

    print(f"Node attributes: {selected_node_attrs}")
    print(f"Edge attributes: {selected_edge_attrs}")
    print(f"Input dim (calculated): {input_dim}")

    return data_list, input_dim, doc_dim

def train_epoch(model, data, optimizer, batch_size=24, loss_fn=None):
    if loss_fn is None:
        #pos_weight має бути перенесений на пристрій моделі
        #num_normal = sum(1 for item in data if item["label"].item() == 0)
        #num_anomalous = sum(1 for item in data if item["label"].item() == 1)
        pos_weight = torch.tensor([15000 / 1600], dtype=torch.float).to(next(model.parameters()).device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    total_loss = 0
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", leave=False, dynamic_ncols=True, mininterval=10):
        # Формування батчу
        batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Доповнення послідовностей у батчі до однакової довжини
        max_seq_length = max(item["sequence"].size(0) for item in batch)
        device = next(model.parameters()).device
        padded_sequences = torch.stack([
            torch.cat([item["sequence"].to(device),
                       torch.zeros(max_seq_length - item["sequence"].size(0),
                                   item["sequence"].size(1), device=device)])
            for item in batch
        ])

        doc_features = torch.stack([item["doc_features"] for item in batch]).to(device)
        labels = torch.stack([item["label"] for item in batch]).to(device)

        # Маска для послідовності (ігнорування паддінгу)
        mask = (padded_sequences.sum(dim=2) == 0).to(device)  # [batch_size, max_seq_length]

        # Скидання градієнтів
        optimizer.zero_grad()

        # Передбачення
        outputs = model(padded_sequences, doc_features, mask=mask)

        # Перевірка, чи всі тензори на одному пристрої
        assert outputs.device == labels.device, "Outputs and labels are on different devices"

        # Обчислення втрат
        loss = loss_fn(outputs, labels)

        # Зворотне поширення
        loss.backward()
        optimizer.step()

        # Оновлення загальних втрат
        total_loss += loss.item()

    # Середнє значення втрат за епоху
    average_loss = total_loss / num_batches
    return average_loss


def calculate_statistics(model, data, batch_size=24, threshold=0.5):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель.
    :param data: Дані для оцінки (список об'єктів Data).
    :param batch_size: Розмір батчу.
    :param threshold: Поріг для визначення класу.
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []
    true_labels, predicted_labels = [], []

    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Формування батчу
            batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # Підготовка даних для моделі
            max_seq_length = max(item["sequence"].size(0) for item in batch)
            padded_sequences = torch.stack([
                torch.cat([item["sequence"], torch.zeros(max_seq_length - item["sequence"].size(0),
                                                         item["sequence"].size(1))])
                for item in batch
            ]).to(device)

            doc_features = torch.stack([item["doc_features"] for item in batch]).to(device)
            batch_labels = [item["label"].item() for item in batch]

            # Маска для послідовності
            mask = (padded_sequences.sum(dim=2) == 0).to(device)

            # Передбачення моделі
            outputs = model(padded_sequences, doc_features, mask=mask).squeeze(1)
            probabilities = torch.sigmoid(outputs).tolist()

            predictions.extend(probabilities)
            labels.extend(batch_labels)

            # Передбачені мітки
            predicted_batch_labels = [1 if p > threshold else 0 for p in probabilities]
            true_labels.extend(batch_labels)
            predicted_labels.extend(predicted_batch_labels)

    # Матриця плутанини
    confusion_matrix_object = {
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }

    # Розрахунок метрик
    auprc = calculate_auprc(labels, predictions)
    adr = calculate_adr(labels, predictions)
    far = calculate_far(labels, predictions)
    fpr = calculate_fpr(labels, predictions)
    fnr = calculate_fnr(labels, predictions)
    roc_auc = calculate_roc_auc(labels, predictions)
    precision, recall = calculate_precision_recall(labels, predictions)
    f1_score = calculate_f1_score(labels, predictions)

    stats = {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "f1_score": f1_score,
        "auprc": auprc,
        "adr": adr,
        "far": far,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": confusion_matrix_object
    }
    return stats



def create_optimizer(model, learning_rate=0.001):
    #return Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
