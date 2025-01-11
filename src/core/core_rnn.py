import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score, calculate_auprc, calculate_adr, calculate_far, calculate_fpr, calculate_fnr
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None):
        """
        Ініціалізація рекурентної нейронної мережі (RNN) з LSTM.

        :param input_dim: Вхідна розмірність вузлів і зв'язків.
        :param hidden_dim: Розмір прихованого шару.
        :param output_dim: Розмір вихідного шару (класифікація).
        :param doc_dim: Розмір документних ознак.
        """
        super(RNN, self).__init__()

        # RNN-шар для вузлів і зв'язків
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # Шар для обробки документних ознак
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Остаточний шар для класифікації
        self.fc = nn.Linear(hidden_dim * 3, output_dim)

        # Активації
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, sequence, doc_features):
        """
        Прямий прохід через модель.

        :param sequence: Тензор послідовності вузлів і зв'язків [batch_size, max_seq_length, input_dim].
        :param doc_features: Тензор документа [batch_size, doc_dim].
        :return: Класифікація [batch_size, output_dim].
        """
        device = next(self.parameters()).device
        sequence = sequence.to(device)
        doc_features = doc_features.to(device)

        # Обробка послідовності через LSTM
        lstm_out, _ = self.lstm(sequence)  # [batch_size, max_seq_length, hidden_dim * 2]
        lstm_out = lstm_out.mean(dim=1)  # Агрегація по довжині послідовності [batch_size, hidden_dim * 2]

        # Обробка документних атрибутів
        doc_emb = self.activation(self.doc_fc(doc_features))  # [batch_size, hidden_dim]

        # Об'єднання всіх ознак
        combined = torch.cat([lstm_out, doc_emb], dim=1)  # [batch_size, hidden_dim * 3]
        output = self.fc(combined)  # [batch_size, output_dim]
        #return self.final_activation(output)
        return output

def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для RNN із врахуванням порядку вузлів (за SEQUENCE_COUNTER_) і чергуванням вузлів і зв'язків.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для RNN, розмірність входу (input_dim), розмірність документа (doc_dim).
    """
    data_list = []
    max_sequence_length = 0

    def transform_graph(graph, node_attrs, edge_attrs):
        """
        Перетворення графу на послідовність із чергуванням вузлів і зв'язків.

        :param graph: Граф із вузлами та зв'язками.
        :param node_attrs: Список атрибутів вузлів.
        :param edge_attrs: Список атрибутів зв'язків.
        :return: Тензор послідовності [sequence_length, input_dim].
        """
        # Сортування вузлів за SEQUENCE_COUNTER_
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1].get("SEQUENCE_COUNTER_", 0))

        sequence = []
        max_node_dim = len(node_attrs)
        max_edge_dim = len(edge_attrs)
        input_dim = max_node_dim + max_edge_dim

        for i, (node, node_data) in enumerate(sorted_nodes):
            # Вузол
            node_features = [float(node_data.get(attr, 0.0)) for attr in node_attrs]
            node_features += [0.0] * (input_dim - len(node_features))  # Доповнення
            sequence.append(node_features)

            # Пошук зв'язків від цього вузла
            for _, target, edge_data in graph.edges(node, data=True):
                edge_features = [float(edge_data.get(attr, 0.0)) for attr in edge_attrs]
                edge_features += [0.0] * (input_dim - len(edge_features))  # Доповнення
                sequence.append(edge_features)

                # Додавання наступного вузла, якщо є
                target_data = graph.nodes[target]
                target_features = [float(target_data.get(attr, 0.0)) for attr in node_attrs]
                target_features += [0.0] * (input_dim - len(target_features))  # Доповнення
                sequence.append(target_features)

        # Перетворення у тензор
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

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs)
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

        sequence_tensor = transform_graph(graph, selected_node_attrs, selected_edge_attrs)
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

    return data_list, input_dim, doc_dim


def train_epoch(model, data, optimizer, batch_size=24, loss_fn=None):
    """
    Виконує одну епоху навчання для моделі RNN.

    :param model: Модель RNN.
    :param data: Дані для навчання (список словників).
    :param optimizer: Оптимізатор.
    :param batch_size: Розмір батчу.
    :param loss_fn: Функція втрат.
    :return: Середнє значення втрат за епоху.
    """
    # Перевірка функції втрат
    if loss_fn is None:
        #pos_weight = torch.tensor([15000 / 1600], dtype=torch.float)
        #num_normal = sum(1 for item in data if item["label"].item() == 0)
        #num_anomalous = sum(1 for item in data if item["label"].item() == 1)
        #pos_weight = torch.tensor([num_normal / num_anomalous], dtype=torch.float)
        #pos_weight = torch.tensor([num_normal / num_anomalous], dtype=torch.float).to(next(model.parameters()).device)
        pos_weight = torch.tensor([15000 / 1600], dtype=torch.float).to(next(model.parameters()).device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #loss_fn = nn.BCELoss() # Використання стандартної функції втрат, не завбути вкючити Sigmoid

    model.train()
    total_loss = 0
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    #for batch_idx in range(num_batches):
    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", leave=False, dynamic_ncols=True, mininterval=10):

        # Вибір даних для батчу
        batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Доповнення послідовностей у батчі до однакової довжини
        max_seq_length = max(item["sequence"].size(0) for item in batch)

        padded_sequences = torch.stack([
            torch.cat([item["sequence"], torch.zeros(max_seq_length - item["sequence"].size(0), item["sequence"].size(1))])
            for item in batch
        ]).to(next(model.parameters()).device)

        doc_features = torch.stack([item["doc_features"] for item in batch]).to(next(model.parameters()).device)
        labels = torch.stack([item["label"] for item in batch]).to(next(model.parameters()).device)

        # Скидання градієнтів
        optimizer.zero_grad()

        # Передбачення
        outputs = model(padded_sequences, doc_features)

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


def calculate_statistics(model, data, threshold=0.5):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель RNN.
    :param data: Дані для оцінки (список словників).
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for item in data:
            # Отримання послідовності та атрибутів документа
            sequence = item["sequence"].unsqueeze(0).to(next(model.parameters()).device)  # [1, max_seq_length, input_dim]
            doc_features = item["doc_features"].unsqueeze(0).to(next(model.parameters()).device)  # [1, doc_dim]
            label = item["label"].item()

            # Передбачення моделі
            output = model(sequence, doc_features).item()
            predictions.append(output)
            labels.append(label)

            # Обчислення передбачених міток
            probability = torch.sigmoid(torch.tensor(output)).item()
            #print("probability", probability)
            predicted_label = 1 if probability > threshold else 0
            true_labels.append(label)
            predicted_labels.append(predicted_label)

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
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    #return Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
