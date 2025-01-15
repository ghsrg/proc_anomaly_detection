import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score, calculate_auprc, calculate_adr, calculate_far, calculate_fpr, calculate_fnr
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH
from tqdm import tqdm
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, edge_dim, doc_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()
        # Енкодер вузлів і зв'язків
        self.node_edge_encoder = nn.Sequential(
            nn.Linear(input_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Енкодер документів
        self.doc_encoder = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Класифікація
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Для прогнозування ймовірності аномалії
        )

    def forward(self, node_features, edge_features, doc_features):
        # Об'єднання вузлів і зв'язків
        combined_sequence = torch.cat([node_features, edge_features], dim=2)
        encoded_sequence = self.node_edge_encoder(combined_sequence)

        # Агрегація (середнє по всіх елементах послідовності)
        aggregated_sequence = torch.mean(encoded_sequence, dim=1)

        # Енкодинг документів
        encoded_docs = self.doc_encoder(doc_features)

        # Об'єднання та класифікація
        combined = torch.cat([aggregated_sequence, encoded_docs], dim=1)
        output = self.fc(combined)

        return output

    def forward(self, node_features, edge_features, doc_features):
        # Об'єднання вузлів і зв'язків
        combined_sequence = torch.cat([node_features, edge_features], dim=2)
        encoded_sequence = self.node_edge_encoder(combined_sequence)

        # Агрегація (середнє по всіх елементах послідовності)
        aggregated_sequence = torch.mean(encoded_sequence, dim=1)

        # Енкодинг документів
        encoded_docs = self.doc_encoder(doc_features)

        # Об'єднання та класифікація
        combined = torch.cat([aggregated_sequence, encoded_docs], dim=1)
        output = self.fc(combined)

        return output



def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для Autoencoder.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для Autoencoder, розмірності вузлів, зв'язків і документів.
    """
    data_list = []
    max_nodes = 0
    max_edges = 0

    def transform_graph(graph, node_attrs, edge_attrs, max_nodes, max_edges):
        node_features = []
        for _, node_data in graph.nodes(data=True):
            features = [float(node_data.get(attr, 0.0)) for attr in node_attrs]
            node_features.append(features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_features = []
        for _, _, edge_data in graph.edges(data=True):
            #print(f"Original edge_data: {edge_data}")
            features = [float(edge_data.get(attr, 0.0)) for attr in edge_attrs]
            edge_features.append(features)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        #print(f"Original edge features tensor: {edge_features}")
        #print(f"Edge features shape before padding: {edge_features.shape}")

        padded_node_features = torch.zeros(max_nodes, node_features.size(1))
        padded_node_features[:node_features.size(0), :] = node_features

        padded_edge_features = torch.zeros(max_edges, edge_features.size(1))
        padded_edge_features[:edge_features.size(0), :] = edge_features
        #print(f"Padded edge features: {padded_edge_features}")
        #print(f"Edge features shape after padding: {padded_edge_features.shape}")

        # Debugging prints
        #print(f"Original node features shape: {node_features.shape}")
        #print(f"Original edge features shape: {edge_features.shape}")
        #print(f"Padded node features shape: {padded_node_features.shape}")
        #print(f"Padded edge features shape: {padded_edge_features.shape}")

        return padded_node_features, padded_edge_features

    def transform_doc(doc_info, selected_doc_attrs):
        doc_features = torch.tensor([doc_info.get(attr, 0.0) for attr in selected_doc_attrs], dtype=torch.float)
        #print(f"Document features shape: {doc_features.shape}")
        return doc_features

    selected_node_attrs = ["type", "DURATION_", "START_TIME_", "END_TIME_", "active_executions", "SEQUENCE_COUNTER_",
                           "overdue_work", "duration_work"]
    selected_edge_attrs = ["DURATION_E", "taskaction_code_E", "overdue_work_E"]
    selected_doc_attrs = ["PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate", "CategoryL1", "CategoryL2",
                          "CategoryL3", "ClassSSD", "Company_SO"]

    input_dim = len(selected_node_attrs)
    doc_dim = len(selected_doc_attrs)
    max_nodes = 0
    max_edges = 0
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Розрахунок нормальних графів", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)
        max_nodes = max(max_nodes, len(graph.nodes))
        max_edges = max(max_edges, len(graph.edges))
    filtered_anomalous_graphs = anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]
    for idx, row in tqdm(filtered_anomalous_graphs.iterrows(), desc="Розрахунок аномальних графів",
                         total=len(filtered_anomalous_graphs)):
        graph_file = row["graph_path"]
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        max_nodes = max(max_nodes, len(graph.nodes))
        max_edges = max(max_edges, len(graph.edges))
    #print (f"max_nodes: {max_nodes}")
    #print (f"max_edges: {max_edges}")
    max_nodes = max(max_nodes, max_edges)
    max_edges = max(max_edges, max_nodes)
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка нормальних графів", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        #print(f"Processing graph {idx + 1}/{len(graph)} - {row['graph_path']}")
        #print(f"Nodes count: {len(graph.nodes)}, Edges count: {len(graph.edges)}")

        #max_nodes = max(max_nodes, len(graph.nodes))
        #max_edges = max(max_edges, len(graph.edges))
        #print(f"Processing normal graph {idx}: {graph_file}, Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

        node_features, edge_features = transform_graph(graph, selected_node_attrs, selected_edge_attrs, max_nodes,
                                                       max_edges)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        data = {
            "node_features": node_features,
            "edge_features": edge_features,
            "doc_features": doc_features,
            "label": torch.tensor([0], dtype=torch.float)
        }
        data_list.append(data)

    filtered_anomalous_graphs = anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]
    for idx, row in tqdm(filtered_anomalous_graphs.iterrows(), desc="Обробка аномальних графів",
                         total=len(filtered_anomalous_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        #print(f"Processing graph {idx + 1}/{len(normal_graphs)} - {row['graph_path']}")
        #print(f"Nodes count: {len(graph.nodes)}, Edges count: {len(graph.edges)}")

        #max_nodes = max(max_nodes, len(graph.nodes))
        #max_edges = max(max_edges, len(graph.edges))
        #print(f"Processing anomalous graph {idx}: {graph_file}, Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

        node_features, edge_features = transform_graph(graph, selected_node_attrs, selected_edge_attrs, max_nodes,
                                                       max_edges)
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        data = {
            "node_features": node_features,
            "edge_features": edge_features,
            "doc_features": doc_features,
            "label": torch.tensor([1], dtype=torch.float)
        }
        data_list.append(data)

    #print(f"Final prepared data: Number of samples: {len(data_list)}")
    #print(f"Input dimension: {input_dim}, Document dimension: {doc_dim}")
    #print(f"Max nodes: {max_nodes}, Max edges: {max_edges}")
    return data_list, input_dim, doc_dim



def train_epoch(model, data, optimizer, batch_size=64, loss_fn=None):
    model.train()
    total_loss = 0
    if loss_fn is None:
        loss_fn = nn.BCELoss()  # Бінарна крос-ентропія

    for i in tqdm(range(0, len(data), batch_size), desc="Поділ на батчі", unit="батч", leave=False, dynamic_ncols=True, mininterval=4):
    #for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # Перевіряємо розміри кожного елемента в батчі
        #for idx, item in enumerate(batch):
            #print(f"Batch index {idx}: node_features shape {item['node_features'].shape}")
        # Підготовка даних
        node_features = torch.stack([item["node_features"] for item in batch]).to(next(model.parameters()).device)
        edge_features = torch.stack([item["edge_features"] for item in batch]).to(next(model.parameters()).device)
        doc_features = torch.stack([item["doc_features"] for item in batch]).to(next(model.parameters()).device)
        labels = torch.stack([item["label"] for item in batch]).to(next(model.parameters()).device)



        optimizer.zero_grad()

        # Прямий прохід
        predictions = model(node_features, edge_features, doc_features)

        # Розрахунок втрат
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data)



def calculate_statistics(model, data, threshold=0.5):
    """
    Розрахунок статистики для Autoencoder після навчання.

    :param model: Навчена модель Autoencoder.
    :param data: Дані для оцінки.
    :param threshold: Поріг для визначення аномалій.
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for item in data:
            node_features = item["node_features"].unsqueeze(0).to(next(model.parameters()).device)
            edge_features = item["edge_features"].unsqueeze(0).to(next(model.parameters()).device)
            doc_features = item["doc_features"].unsqueeze(0).to(next(model.parameters()).device)
            label = item["label"].item()

            # Передбачення Autoencoder
            output = model(node_features, edge_features, doc_features).item()

            predictions.append(output)
            labels.append(label)

            # Визначення передбаченої мітки
            predicted_label = 1 if output > threshold else 0
            true_labels.append(label)
            predicted_labels.append(predicted_label)

    # Розрахунок метрик
    confusion_matrix_object = {
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }

    # Розрахунок метрик для оцінки якості
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


def create_optimizer(model, learning_rate=1e-3, weight_decay=1e-5):
    """
    Створює оптимізатор для Autoencoder.

    :param model: Модель Autoencoder.
    :param learning_rate: Швидкість навчання.
    :param weight_decay: Регуляризація (L2).
    :return: Оптимізатор.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
