import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm
from collections import Counter
from types import SimpleNamespace

logger = get_logger(__name__)



class EGCN_H_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=470):
        """
        :param input_dim: розмірність ознак вузлів (включно з active_mask)
        :param hidden_dim: розмірність прихованих шарів
        :param output_dim: кількість класів (для задачі класифікації)
        :param doc_dim: розмірність ознак документа
        :param num_nodes: фіксована кількість вузлів (потрібна для EvolveGCNH)
        """
        super(EGCN_H_pr, self).__init__()
        print(f"📦 Init EGCN_H_pr with input_dim={input_dim}, output_dim={output_dim}, num_nodes={num_nodes}")

        self.num_nodes = num_nodes
        self.input_dim = input_dim

        # Рекурентна GNN
        self.gnn = EvolveGCNH(num_of_nodes=num_nodes, in_channels=input_dim)

        # Обробка ознак документа
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Обробка виходу GNN
        self.gnn_fc = nn.Linear(input_dim, hidden_dim)

        # Об'єднання
        self.bnf = nn.LayerNorm(hidden_dim * 2)
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)

        # Два виходи: класифікація та регресія часу
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

        self.global_pool = global_mean_pool

    def forward(self, data, doc_feature=None):
        device = self.task_head.weight.device

        # Безпечне від'єднання графа ваг
        if self.gnn.weight is not None:
            self.gnn.weight = self.gnn.weight.detach()

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        x_gnn = self.gnn(x, edge_index)
        x_pooled = self.global_pool(x_gnn, batch)
        x_gnn_transformed = self.gnn_fc(x_pooled)

        if doc_feature is not None:
            doc_emb = self.activation(self.doc_fc(doc_feature.unsqueeze(0).to(device)))
        else:
            doc_emb = torch.zeros((1, self.doc_fc.out_features), device=device)

        x = torch.cat([x_gnn_transformed, doc_emb], dim=1)
        x = self.bnf(x)
        x = self.activation(self.fusion_head(x))
        x = self.dropout(x)

        return self.task_head(x), self.time_head(x)



def train_epoch(model, data_sequences, optimizer, alpha=0, global_node_dict=None, train_activity_counter=None, limit=None):
    """
    Навчання моделі на послідовностях Instance Graphs (List[List[Data]]) без обнулення ваг.
    """
    model.train()
    total_loss = 0
    device = model.task_head.weight.device
    total = len(data_sequences) if limit is None else min(limit, len(data_sequences))

    for sequence in tqdm(data_sequences[:total], desc="Навчання послідовностей", unit="proc", dynamic_ncols=True):
        if not sequence:
            continue

        optimizer.zero_grad()

        # Підготовка послідовності
        doc_features = torch.stack([d.doc_features for d in sequence], dim=0).to(device)
        y_task = torch.tensor([d.y.item() for d in sequence], dtype=torch.long).to(device)
        y_time = torch.stack([d.time_target for d in sequence]).view(-1).to(device)

        # Forward для всієї послідовності
        out_task, out_time = model(sequence, doc_features)

        # Розрахунок втрат
        loss_task = nn.CrossEntropyLoss()(out_task, y_task)
        loss_time = nn.MSELoss()(out_time.view(-1), y_time)
        loss = loss_task + alpha * loss_time

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Оновлення лічильника тренувань (якщо передано)
        if global_node_dict and train_activity_counter:
            for label in y_task.cpu().numpy():
                if label in global_node_dict.values():
                    train_activity_counter[label] += 1

    return total_loss / total if total > 0 else 0.0


def calculate_statistics(model, val_sequences, global_node_dict, global_statistics, top_k=3, train_activity_counter=None):
    model.eval()

    all_preds, all_labels = [], []
    all_pred_ids, all_true_ids = [], []
    topk_hits, time_preds, time_labels = [], [], []

    from collections import Counter
    activity_total_counter = Counter()
    activity_correct_counter = Counter()

    with torch.no_grad():
        for sequence in tqdm(val_sequences, desc="Валідація (послідовності)", unit="seq", dynamic_ncols=True):
            if not sequence:
                continue

            doc_features = torch.stack([d.doc_features for d in sequence], dim=0).to(model.task_head.weight.device)
            y_task = torch.tensor([d.y.item() for d in sequence], dtype=torch.long).to(model.task_head.weight.device)
            y_time = torch.stack([d.time_target for d in sequence]).view(-1).to(model.task_head.weight.device)

            outputs_task, outputs_time = model(sequence, doc_features)
            preds = torch.argmax(outputs_task, dim=1)
            topk = torch.topk(outputs_task, k=top_k, dim=1).indices.cpu().numpy()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_task.cpu().numpy())

            topk_hits.extend([
                true_label in top_row
                for true_label, top_row in zip(y_task.cpu().numpy(), topk)
            ])

            time_preds.extend(outputs_time.view(-1).cpu().numpy())
            time_labels.extend(y_time.cpu().numpy())

            for d, pred, label in zip(sequence, preds, y_task):
                node_ids = d.node_ids
                pred_id = node_ids[pred.item()] if pred.item() < len(node_ids) else "INVALID"
                label_id = node_ids[label.item()] if label.item() < len(node_ids) else "INVALID"

                all_pred_ids.append(pred_id)
                all_true_ids.append(label_id)

                label_idx = global_node_dict.get(label_id)
                if label_idx is not None:
                    activity_total_counter[label_idx] += 1
                    if pred_id == label_id:
                        activity_correct_counter[label_idx] += 1

    # Метрики
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score
    )

    valid_indices = [i for i, pid in enumerate(all_pred_ids) if pid != "INVALID"]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    precision = precision_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    recall = recall_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    f1 = f1_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    acc = accuracy_score(filtered_labels, filtered_preds)
    filtered_topk_hits = [topk_hits[i] for i in valid_indices]
    top_k_accuracy = sum(filtered_topk_hits) / len(filtered_topk_hits) if filtered_topk_hits else 0.0
    out_of_scope_rate = (len(all_preds) - len(filtered_preds)) / len(all_preds) if all_preds else 0.0
    cm = confusion_matrix(filtered_labels, filtered_preds)

    # Регресійні метрики
    mae = mean_absolute_error(time_labels, time_preds)
    mse = mean_squared_error(time_labels, time_preds)
    rmse = mse ** 0.5
    r2 = max(0, r2_score(time_labels, time_preds))

    # Денормалізація
    min_dur = global_statistics["node_numeric"]["duration_work"]["min"]
    max_dur = global_statistics["node_numeric"]["duration_work"]["max"]
    if max_dur > min_dur:
        denorm_time_preds = [(v * (max_dur - min_dur) + min_dur) for v in time_preds]
        denorm_time_labels = [(v * (max_dur - min_dur) + min_dur) for v in time_labels]
        mae_real = mean_absolute_error(denorm_time_labels, denorm_time_preds)
        mse_real = mean_squared_error(denorm_time_labels, denorm_time_preds)
        rmse_real = mse_real ** 0.5
        r2_real = max(0, r2_score(denorm_time_labels, denorm_time_preds))
    else:
        mae_real = mse_real = rmse_real = r2_real = None

    activity_train_vs_val_accuracy = {
        node_idx: {
            "train_count": train_activity_counter.get(node_idx, 0) if train_activity_counter else 0,
            "val_accuracy": (activity_correct_counter[node_idx] / activity_total_counter[node_idx])
            if activity_total_counter[node_idx] > 0 else None
        }
        for node_idx in global_node_dict.values()
    }

    return {
        "accuracy": acc,
        "top_k_accuracy": top_k_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "true_node_ids": all_true_ids,
        "pred_node_ids": all_pred_ids,
        "mae": mae_real,
        "rmse": rmse_real,
        "r2": r2_real,
        "out_of_scope_rate": out_of_scope_rate,
        "activity_train_vs_val_accuracy": activity_train_vs_val_accuracy
    }


def prepare_data(normal_graphs, max_node_count, max_edge_count, limit=None):
    """
    Готує List[List[Data]], де кожна підпослідовність — Instance Graphs одного процесу.
    """
    from collections import defaultdict
    from torch_geometric.data import Data
    import torch

    data_by_process = defaultdict(list)
    max_features = 0
    max_doc_dim = 0
    global_node_set = set()

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
        return torch.tensor([
            float(doc_info.get(attr, 0)) if isinstance(doc_info.get(attr), (int, float)) else 0.0
            for attr in selected_doc_attrs
        ], dtype=torch.float)

    def transform_graph(graph, current_nodes, next_node, node_attrs, edge_attrs, doc_info, selected_doc_attrs):
        nonlocal max_features, max_doc_dim, global_node_set

        real_node_ids = list(graph.nodes())
        simplified_node_ids = [simplify_bpmn_id(n) for n in real_node_ids]
        global_node_set.update(simplified_node_ids)
        node_map = {node: idx for idx, node in enumerate(real_node_ids)}

        node_features, active_mask, timestamps = [], [], []
        for node_id in real_node_ids:
            features = [
                float(graph.nodes[node_id].get(attr, 0)) if isinstance(graph.nodes[node_id].get(attr), (int, float)) else 0.0
                for attr in node_attrs
            ]
            node_features.append(features)
            active_mask.append(1.0 if node_id in current_nodes else 0.0)
            timestamps.append(float(graph.nodes[node_id].get("START_TIME_", 0)) if node_id in current_nodes else 1.1)

        pad_size = max_node_count - len(real_node_ids)
        node_features.extend([[0.0] * len(node_attrs)] * pad_size)
        active_mask.extend([0.0] * pad_size)
        timestamps.extend([1.1] * pad_size)

        x = torch.tensor(node_features, dtype=torch.float)
        x = torch.cat([x, torch.tensor(active_mask, dtype=torch.float).view(-1, 1)], dim=1)
        timestamps = torch.tensor(timestamps, dtype=torch.float)
        max_features = max(max_features, x.shape[1])

        real_edges = [(node_map[u], node_map[v]) for u, v in graph.edges() if u in node_map and v in node_map]
        edge_index = torch.tensor(real_edges, dtype=torch.long).t().contiguous() if real_edges else torch.empty((2, 0), dtype=torch.long)

        if edge_attrs and real_edges:
            edge_attr = torch.tensor([
                [float(graph.get_edge_data(u, v).get(attr, 0)) for attr in edge_attrs]
                for u, v in graph.edges()
            ], dtype=torch.float)
        else:
            edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.float)

        if edge_index.shape[1] < 1:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, len(edge_attrs) if edge_attrs else 1), dtype=torch.float)

        doc_features = transform_doc(doc_info, selected_doc_attrs)
        max_doc_dim = max(max_doc_dim, doc_features.numel())
        time_target = torch.tensor([graph.nodes[next_node].get("duration_work", 0.0)], dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([node_map[next_node]], dtype=torch.long),
            doc_features=doc_features,
            time_target=time_target,
            node_ids=simplified_node_ids,
            timestamps=timestamps
        )

    selected_node_attrs = [
        "DURATION_", "START_TIME_", "PROC_KEY_", "active_executions", "user_compl_login",
        "SEQUENCE_COUNTER_", "taskaction_code", "task_status"
    ]
    selected_edge_attrs = ["DURATION_E"]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    count = 0
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів", total=len(normal_graphs)):
        if limit and count >= limit:
            break

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

        executed.sort(key=lambda n: graph.nodes[n].get("SEQUENCE_COUNTER_", 0))
        sequence = []

        for i in range(1, len(executed)):
            current_nodes = executed[:i]
            max_seq = max(graph.nodes[n].get("SEQUENCE_COUNTER_", 0) for n in current_nodes)
            candidates = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > max_seq]
            if not candidates:
                continue
            next_node = min(candidates, key=lambda n: graph.nodes[n]["SEQUENCE_COUNTER_"])

            data = transform_graph(graph, current_nodes, next_node, selected_node_attrs, selected_edge_attrs,
                                   doc_info, selected_doc_attrs)
            if data:
                sequence.append(data)

        if sequence:
            key = graph_file  # або інший ключ, якщо обробляється кілька версій одного процесу
            data_by_process[key].extend(sequence)
            count += 1

    # Формування остаточного List[List[Data]]
    data_sequences = []
    for seq in data_by_process.values():
        seq.sort(key=lambda d: d.timestamps.min().item() if hasattr(d, 'timestamps') else 0)
        data_sequences.append(seq)

    global_node_dict = {name: idx for idx, name in enumerate(sorted(global_node_set))}
    return data_sequences, max_features, max_doc_dim, global_node_dict



def prepare_data_log_only(normal_graphs,max_node_count):
    data_list = []
    max_features = 0
    max_doc_dim = 0
    global_node_dict = {}

    def transform_doc(doc_info, selected_doc_attrs):
        return torch.tensor([
            float(doc_info.get(attr, 0) or 0.0) for attr in selected_doc_attrs
        ], dtype=torch.float)

    def transform_graph(node_ids, graph, current_nodes, next_node, node_attrs, doc_info, selected_doc_attrs):
        nonlocal max_features
        simplified_node_ids = [simplify_bpmn_id(n) for n in node_ids]

        for node_id in simplified_node_ids:
            if node_id not in global_node_dict:
                global_node_dict[node_id] = len(global_node_dict)

        node_map = {node: idx for idx, node in enumerate(node_ids)}
        x = torch.tensor([
            [float(graph.nodes[n].get(attr, 0)) if isinstance(graph.nodes[n].get(attr), (int, float)) else 0.0
             for attr in node_attrs]
            for n in node_ids
        ], dtype=torch.float)

        active_mask = torch.tensor([
            1.0 if n in current_nodes else 0.0 for n in node_ids
        ], dtype=torch.float).view(-1, 1)
        x = torch.cat([x, active_mask], dim=1)

        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])

        timestamps = torch.tensor([
            float(graph.nodes[n].get("START_TIME_", 0.0)) if n in current_nodes else 1.1
            for n in node_ids
        ], dtype=torch.float)

        edge_list = [(node_map[node_ids[i]], node_map[node_ids[i+1]]) for i in range(len(current_nodes)-1)]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)

        doc_features = transform_doc(doc_info, selected_doc_attrs)
        time_target = torch.tensor([graph.nodes[next_node].get("duration_work", 0.0)], dtype=torch.float)
        inverse_node_map = [simplify_bpmn_id(n) for n in node_ids]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([node_map[next_node]], dtype=torch.long),
            doc_features=doc_features,
            time_target=time_target,
            node_ids=inverse_node_map,
            timestamps=timestamps
        )

    selected_node_attrs = [
        "DURATION_", "START_TIME_", "PROC_KEY_", "active_executions", "user_compl_login",
        "SEQUENCE_COUNTER_", "taskaction_code", "task_status"
    ]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    for _, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів як логів", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_PR_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        executed_nodes = [(n, d) for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > 0]
        if len(executed_nodes) < 2:
            continue

        executed_nodes.sort(key=lambda x: x[1].get("SEQUENCE_COUNTER_", 0))
        node_ids = [n for n, _ in executed_nodes]

        for i in range(1, len(node_ids)):
            current_nodes = node_ids[:i]
            next_node = node_ids[i]

            data = transform_graph(node_ids, graph, current_nodes, next_node, selected_node_attrs, doc_info, selected_doc_attrs)
            if data:
                data_list.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

    return data_list, max_features, max_doc_dim, global_node_dict


def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    #return Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

def simplify_bpmn_id(raw_id: str) -> str:
    match = re.match(r'^([^_]+_[^_]+)', raw_id)
    return match.group(1) if match else raw_id