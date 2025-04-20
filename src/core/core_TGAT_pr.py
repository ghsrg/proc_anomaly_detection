import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATv2Conv  # as fallback if TGAT not available
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm

logger = get_logger(__name__)

class TGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__()
        self.heads = heads
        self.att = TransformerConv(in_channels, out_channels // heads, heads=heads, dropout=0.1)

    def forward(self, x, edge_index):
        return self.att(x, edge_index)

class TemporalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, t):
        pe = torch.zeros_like(x)
        position = t.unsqueeze(1)
        max_len = x.size(1)
        div_term = torch.exp(torch.arange(0, max_len, 2, dtype=torch.float, device=x.device) *
                             -(torch.log(torch.tensor(10000.0)) / max_len))

        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)

        pe[:, 0::2] = sin_term[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_term[:, :pe[:, 1::2].shape[1]]
        return x + pe
class TGAT_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None):
        super(TGAT_pr, self).__init__()
        self.conv1 = TGATLayer(input_dim, hidden_dim)
        self.conv2 = TGATLayer(hidden_dim, hidden_dim)
        self.temporal_encoding = TemporalEncoding(input_dim)


        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.3)

        self.doc_fc = nn.Linear(doc_dim, hidden_dim)
        self.global_pool = global_mean_pool
        self.activation = nn.ReLU()
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_features', None)
        doc_features = getattr(data, 'doc_features', None)
        timestamps = getattr(data, 'timestamps', None)

        if timestamps is not None:
            x = self.temporal_encoding(x, timestamps)
        else:
            logger.warning("Відсутні часові мітки для фічеризації графа.")

        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.bn1(x))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.bn2(x))
        x = self.dropout(x)
        x = self.global_pool(x, batch)

        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features))
        else:
            doc_emb = torch.zeros(x.shape[0], self.doc_fc.out_features, device=x.device)

        x = torch.cat([x, doc_emb], dim=1)
        x = self.activation(self.fusion_head(x))
        x = self.dropout(x)

        task_output = self.task_head(x)
        time_output = self.time_head(x)
        return task_output, time_output

def simplify_bpmn_id(raw_id: str) -> str:
    match = re.match(r'^([^_]+_[^_]+)', raw_id)
    return match.group(1) if match else raw_id

def prepare_data(normal_graphs):
    """
    Prepares data for GNN prediction (next activity and time) with TGAT timestamps variant 3.
    :param normal_graphs: Registry of normal graphs.
    :return: List of Data objects for GNN.
    """
    data_list = []
    max_features = 0
    max_doc_dim = 0

    def infer_graph_attributes(graph):
        numeric_attrs, global_attrs, edge_attrs = set(), set(), set()
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

    def transform_doc(doc_info, selected_doc_attrs):
        doc_features = []
        for attr in selected_doc_attrs:
            value = doc_info.get(attr, 0)
            try:
                doc_features.append(float(value))
            except (ValueError, TypeError):
                doc_features.append(0.0)
        return torch.tensor(doc_features, dtype=torch.float)

    def transform_graph(graph, current_nodes, next_node, node_attrs, edge_attrs, doc_info, selected_doc_attrs):
        node_map = {node: idx for idx, node in enumerate(graph.nodes())}
        nonlocal max_features

        if graph.number_of_edges() == 0:
            return None

        node_features, active_mask, timestamps = [], [], []
        for node_id, node_data in graph.nodes(data=True):
            features = [
                float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                for attr in node_attrs
            ]
            node_features.append(features)
            active_mask.append(1.0 if node_id in current_nodes else 0.0)
            # TGAT timestamp: only meaningful for active nodes, others marked as large future time
            t = float(node_data.get("START_TIME_", 0.0))
            timestamps.append(t if node_id in current_nodes else 1.1)

        x = torch.tensor(node_features, dtype=torch.float)
        active_mask = torch.tensor(active_mask, dtype=torch.float).view(-1, 1)
        x = torch.cat([x, active_mask], dim=1)

        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])

        edges = [(node_map[edge[0]], node_map[edge[1]]) for edge in graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        edge_attr = torch.tensor([
            [float(edge_data.get(attr, 0)) for attr in edge_attrs]
            for _, _, edge_data in graph.edges(data=True)
        ], dtype=torch.float) if edge_attrs else None

        if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
            edge_attr = torch.zeros((edge_index.shape[1], len(edge_attrs) if edge_attrs else 1), dtype=torch.float)

        doc_features = transform_doc(doc_info, selected_doc_attrs)
        time_target = torch.tensor([graph.nodes[next_node].get("duration_work", 0.0)], dtype=torch.float) if next_node else None
        inverse_node_map = [simplify_bpmn_id(n) for n in graph.nodes()]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([node_map[next_node]], dtype=torch.long) if next_node else None,
            doc_features=doc_features,
            time_target=time_target,
            node_ids=inverse_node_map,
            timestamps=torch.tensor(timestamps, dtype=torch.float)
        )

        return data

    selected_node_attrs = [
        "DURATION_", "START_TIME_", "END_TIME_", "active_executions", "user_compl_login",
        "SEQUENCE_COUNTER_", "taskaction_code", "task_status"
    ]
    selected_edge_attrs = ["DURATION_E"]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_PR_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        executed = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > 0]
        if not executed:
            continue

        for i in range(1, len(executed) + 1):
            current_nodes = executed[:i]
            executed_seq = [graph.nodes[n].get("SEQUENCE_COUNTER_", 0) for n in current_nodes]
            max_seq = max(executed_seq)
            candidates = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > max_seq]
            if not candidates:
                continue
            next_node = min(candidates, key=lambda n: graph.nodes[n]["SEQUENCE_COUNTER_"])

            data = transform_graph(graph, current_nodes, next_node, selected_node_attrs, selected_edge_attrs, doc_info, selected_doc_attrs)
            if data:
                data_list.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

    return data_list, max_features, max_doc_dim


def train_epoch(model, data, optimizer, batch_size=64, alpha=0.20):
    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", position=1,leave=False, dynamic_ncols=True):

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        batch_tensor = torch.cat([torch.full((item.x.size(0),), idx, dtype=torch.long) for idx, item in enumerate(batch)], dim=0)
        x = torch.cat([item.x for item in batch], dim=0)
        edge_index = torch.cat([item.edge_index for item in batch], dim=1)
        edge_attr = torch.cat([item.edge_attr for item in batch], dim=0) if batch[0].edge_attr is not None else None
        doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None
        timestamps = torch.cat([item.timestamps for item in batch], dim=0) if hasattr(batch[0], 'timestamps') else None

        y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
        y_time = torch.stack([item.time_target for item in batch]).view(-1)

        # Переносимо на GPU якщо є
        x, edge_index, edge_attr, batch_tensor, doc_features = [t.to(model.task_head.weight.device) if t is not None else None for t in [x, edge_index, edge_attr, batch_tensor, doc_features]]
        y_task = y_task.to(model.task_head.weight.device)
        y_time = y_time.to(model.task_head.weight.device)

        optimizer.zero_grad()
        outputs_task, outputs_time = model.forward(
            type("Batch", (object,), {
                "x": x,
                "edge_index": edge_index,
                "edge_features": edge_attr,
                "batch": batch_tensor,
                "doc_features": doc_features,
                "timestamps": timestamps
            })
        )

        loss_task = nn.CrossEntropyLoss()(outputs_task, y_task)
        loss_time = nn.MSELoss()(outputs_time.view(-1), y_time)
        loss = loss_task + alpha * loss_time

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def calculate_statistics(model, val_data, top_k=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_ids = []
    all_true_ids = []
    time_preds = []
    time_labels = []

    with torch.no_grad():
        # Формуємо батч вручну із усіх об'єктів у data_loader
        batch_tensor = torch.cat([
            torch.full((item.x.size(0),), idx, dtype=torch.long)
            for idx, item in enumerate(val_data)
        ], dim=0)

        # Об'єднані тензори по всіх графах
        x = torch.cat([item.x for item in val_data], dim=0)
        edge_index = torch.cat([item.edge_index for item in val_data], dim=1)
        edge_attr = torch.cat([item.edge_attr for item in val_data], dim=0) if val_data[0].edge_attr is not None else None
        doc_features = torch.stack([item.doc_features for item in val_data], dim=0) if val_data[0].doc_features is not None else None
        timestamps = torch.cat([item.timestamps for item in val_data], dim=0) if hasattr(val_data[0], 'timestamps') else None

        y_task = torch.tensor([item.y.item() for item in val_data], dtype=torch.long)
        y_time = torch.stack([item.time_target for item in val_data]).view(-1)

        # Усі node_ids у порядку побудови батчу
        #node_ids_full = sum([item.id_map for item in val_data], [])
        node_ids_full = [node_id for item in val_data for node_id in item.node_ids]

        # Створюємо об'єкт батчу
        data = type("Batch", (object,), {
            "x": x,
            "edge_index": edge_index,
            "edge_features": edge_attr,
            "batch": batch_tensor,
            "doc_features": doc_features,
            "timestamps": timestamps
        })

        # Прогнозуємо
        task_logits, time_output  = model(data)
        preds = torch.argmax(task_logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_task.cpu().numpy())

        # Якщо є node_ids — конвертуємо індекси назад у ID вузлів
        if node_ids_full:
            all_pred_ids = [node_ids_full[i] for i in all_preds]
            all_true_ids = [node_ids_full[i] for i in all_labels]

        # Top-k Accuracy
        topk_preds = torch.topk(task_logits, k=top_k, dim=1).indices.cpu().numpy()
        topk_hits = [label in pred_row for label, pred_row in zip(all_labels, topk_preds)]
        top_k_accuracy = sum(topk_hits) / len(topk_hits)

        # Статистика по часу
        time_preds = time_output.view(-1).cpu().numpy()
        time_labels = y_time.cpu().numpy()

    # Класифікаційні метрики
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Регресійні метрики
    mae = mean_absolute_error(time_labels, time_preds)
    mse = mean_squared_error(time_labels, time_preds)
    rmse = mse ** 0.5
    r2 = r2_score(time_labels, time_preds)
    r2 = max(0, r2)  # якщо R² < 0, встановлюємо 0 щоб на графіку не було негативних значень

    return {
        "accuracy": acc,
        "top_k_accuracy": top_k_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "true_node_ids": all_true_ids,
        "pred_node_ids": all_pred_ids,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    #return Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def prepare_data_log_only(normal_graphs):
    """
    Формує дані для GNN, використовуючи лише інформацію з логів виконання (SEQUENCE_COUNTER_).
    Метою є прогнозування наступної задачі з-поміж усіх можливих, без структури BPMN.

    :param normal_graphs: Таблиця з graph_path і doc_info.
    :return: data_list, max_features, max_doc_dim
    """

    data_list = []
    max_features = 0
    max_doc_dim = 0

    selected_node_attrs = [
        "DURATION_", "START_TIME_", "END_TIME_", "active_executions",
        "SEQUENCE_COUNTER_", "overdue_work", "duration_work"
    ]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    def transform_doc(doc_info):
        features = []
        for attr in selected_doc_attrs:
            value = doc_info.get(attr, 0)
            try:
                features.append(float(value))
            except:
                features.append(0.0)
        return torch.tensor(features, dtype=torch.float)

    for _, row in tqdm(normal_graphs.iterrows(), desc="Логи → графи"):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_PR_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        executed_nodes = [(n, d) for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > 0]
        if len(executed_nodes) < 2:
            continue

        executed_nodes.sort(key=lambda x: x[1]["SEQUENCE_COUNTER_"])
        node_ids = [n for n, _ in executed_nodes]
        node_map = {n: i for i, n in enumerate(node_ids)}

        # Підготовка фіч для всіх виконаних вузлів
        node_features = [
            [
                float(graph.nodes[n].get(attr, 0)) if isinstance(graph.nodes[n].get(attr), (int, float)) else 0.0
                for attr in selected_node_attrs
            ]
            for n in node_ids
        ]
        x_all = torch.tensor(node_features, dtype=torch.float)

        for i in range(len(node_ids) - 1):
            current_nodes = node_ids[:i + 1]  # поточна історія
            next_node = node_ids[i + 1]       # справжня наступна дія

            active_mask = [1.0 if n in current_nodes else 0.0 for n in node_ids]
            x = x_all.clone()
            x = torch.cat([x, torch.tensor(active_mask, dtype=torch.float).view(-1, 1)], dim=1)

            edge_index = torch.tensor([[], []], dtype=torch.long)  # без структури
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
            y = torch.tensor([node_map[next_node]], dtype=torch.long)
            doc_features = transform_doc(doc_info)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, doc_features=doc_features)
            data_list.append(data)
            max_features = max(max_features, x.shape[1])
            max_doc_dim = max(max_doc_dim, doc_features.numel())

    return data_list, max_features, max_doc_dim
