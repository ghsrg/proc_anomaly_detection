import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric.data import Data
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm
from types import SimpleNamespace

logger = get_logger(__name__)


from torch_geometric_temporal.nn.recurrent import EvolveGCNH

class EGCN_H_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=100):
        super(EGCN_H_pr, self).__init__()

        # GNN шар: in_channels = кількість ознак вузла
        self.gnn = EvolveGCNH(num_of_nodes=num_nodes, in_channels=input_dim)

        self.dropout = nn.Dropout(p=0.3)
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)
        self.global_pool = global_mean_pool
        self.activation = nn.ReLU()
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bnf = nn.BatchNorm1d(hidden_dim * 2)
        self.task_head = nn.Linear(hidden_dim, output_dim)  # Класифікація вузлів
        self.time_head = nn.Linear(hidden_dim, 1)  # Прогноз часу

    def forward(self, sequence, doc_features=None):
        x_seq = [data.x.to(self.task_head.weight.device) for data in sequence]
        edge_index_seq = [data.edge_index.to(self.task_head.weight.device) for data in sequence]
        batch_seq = [data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long,
                                                                           device=data.x.device) for data in sequence]

        nodes_mask_list = [torch.ones(x.size(0), dtype=torch.bool, device=x.device) for x in x_seq]

        # Проходимо покроково
        for x_t, edge_index_t, mask_t in zip(x_seq, edge_index_seq, nodes_mask_list):
            x_t = self.gnn(x_t, edge_index_t, mask_t)

        x = self.global_pool(x_t, batch_seq[-1])
        x = self.dropout(x)

        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features))
        else:
            doc_emb = torch.zeros(x.shape[0], self.doc_fc.out_features, device=x.device)

        x = torch.cat([x, doc_emb], dim=1)
        x = self.bnf(x)
        x = self.activation(self.fusion_head(x))
        x = self.dropout(x)

        task_output = self.task_head(x)
        time_output = self.time_head(x)
        return task_output, time_output


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

def train_epoch(model, data, optimizer, batch_size=1, alpha=0):
    """
    Тренування однієї епохи для EvolveGCN-моделі на послідовностях графів.
    :param model: GNN-модель (EvolveGCN)
    :param data: Список послідовностей графів (List[List[Data]])
    :param optimizer: Оптимізатор
    :param batch_size: Розмір батчу (для EvolveGCN краще 1)
    :param alpha: Вага регресійної втрати (час)
    :return: Середнє значення втрати
    """
    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", position=1, leave=False, dynamic_ncols=True):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]  # batch = List[List[Data]]

        # Працюємо з одним прикладом — першим у батчі
        sequence = batch[0]
        x_seq = [graph.x.to(model.task_head.weight.device) for graph in sequence]
        edge_index_seq = [graph.edge_index.to(model.task_head.weight.device) for graph in sequence]

        # Беремо дані останнього графа в послідовності
        last_graph = sequence[-1]
        doc_features = last_graph.doc_features.to(model.task_head.weight.device) if last_graph.doc_features is not None else None
        y_task = last_graph.y.to(model.task_head.weight.device)
        y_time = last_graph.time_target.to(model.task_head.weight.device)

        optimizer.zero_grad()
        #output_task, output_time = model(x_seq, edge_index_seq, doc_features)
        output_task, output_time = model(sequence, doc_features)

        loss_task = nn.CrossEntropyLoss()(output_task, y_task)
        loss_time = nn.MSELoss()(output_time.view(-1), y_time.view(-1))
        loss = loss_task + alpha * loss_time

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_batches

def calculate_statistics2(model, val_data, global_node_dict, global_statistics, batch_size=64, top_k=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_ids = []
    all_true_ids = []
    topk_hits = []
    time_preds = []
    time_labels = []



    num_batches = (len(val_data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Валідація (батчами)", unit="батч"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(val_data))
            batch = val_data[start_idx:end_idx]


            batch_tensor = torch.cat([
                torch.full((item[-1].x.size(0),), idx, dtype=torch.long) for idx, item in enumerate(batch)
            ], dim=0)

            x = torch.cat([item.x for item in batch], dim=0)
            edge_index = torch.cat([item.edge_index for item in batch], dim=1)
            edge_attr = torch.cat([item.edge_attr for item in batch], dim=0) if batch[0].edge_attr is not None else None
            doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None
            timestamps = torch.cat([item.timestamps for item in batch], dim=0) if hasattr(batch[0], 'timestamps') else None

            y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
            y_time = torch.stack([item.time_target for item in batch]).view(-1)

            x, edge_index, edge_attr, batch_tensor, doc_features = [
                t.to(model.task_head.weight.device) if t is not None else None
                for t in [x, edge_index, edge_attr, batch_tensor, doc_features]
            ]
            y_task = y_task.to(model.task_head.weight.device)
            y_time = y_time.to(model.task_head.weight.device)
            if timestamps is not None:
                timestamps = timestamps.to(model.task_head.weight.device)

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

            for item, pred, label in zip(batch, preds, y_task):
                # отримаємо всі допустимі глобальні індекси вузлів для цього графа
                valid_global_indices = {global_node_dict.get(node_id) for node_id in item.node_ids}
                pred_id = global_node_dict.get(item.node_ids[pred.item()]) if pred.item() < len(item.node_ids) else None
                label_id = global_node_dict.get(item.node_ids[label.item()]) if label.item() < len(item.node_ids) else None

                all_pred_ids.append(item.node_ids[pred.item()] if pred_id in valid_global_indices else "INVALID")
                all_true_ids.append(item.node_ids[label.item()] if label_id in valid_global_indices else "INVALID")

    # Фільтрація валідних індексів (де передбачення не INVALID)
    valid_indices = [i for i, pred_id in enumerate(all_pred_ids) if pred_id != "INVALID"]

    # Відфільтровані списки
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    # Метрики класифікації
    precision = precision_score(filtered_labels, filtered_preds, average='macro')
    recall = recall_score(filtered_labels, filtered_preds, average='macro')
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    acc = accuracy_score(filtered_labels, filtered_preds)
    filtered_topk_hits = [topk_hits[i] for i in valid_indices]
    top_k_accuracy = sum(filtered_topk_hits) / len(filtered_topk_hits) if filtered_topk_hits else 0.0

    out_of_scope_count = len(all_preds) - len(filtered_preds)
    out_of_scope_rate = out_of_scope_count / len(all_preds)

    # Класифікаційні метрики
    #precision = precision_score(all_labels, all_preds, average='macro')
    #recall = recall_score(all_labels, all_preds, average='macro')
    #f1 = f1_score(all_labels, all_preds, average='macro')
    #acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    #top_k_accuracy = sum(topk_hits) / len(topk_hits)

    # Регресійні метрики
    mae = mean_absolute_error(time_labels, time_preds)
    mse = mean_squared_error(time_labels, time_preds)
    rmse = mse ** 0.5
    r2 = max(0, r2_score(time_labels, time_preds))

    # Регресійні метрики (денормалізовані значення)
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
        "out_of_scope_rate": out_of_scope_rate
    }

def calculate_statistics(model, val_data, global_node_dict, global_statistics, top_k=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_ids = []
    all_true_ids = []
    topk_hits = []
    time_preds = []
    time_labels = []

    with torch.no_grad():
        for sequence in val_data:
            last_graph = sequence[-1]
            doc_features = last_graph.doc_features.to(model.task_head.weight.device) if last_graph.doc_features is not None else None
            y_task = last_graph.y.to(model.task_head.weight.device)
            y_time = last_graph.time_target.to(model.task_head.weight.device)

            outputs_task, outputs_time = model(sequence, doc_features)

            pred = torch.argmax(outputs_task, dim=1)
            topk = torch.topk(outputs_task, k=top_k, dim=1).indices.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_task.cpu().numpy())

            topk_hits.extend([
                true_label in top_row
                for true_label, top_row in zip(y_task.cpu().numpy(), topk)
            ])

            time_preds.extend(outputs_time.view(-1).cpu().numpy())
            time_labels.extend(y_time.cpu().numpy())

            valid_global_indices = {global_node_dict.get(nid) for nid in last_graph.node_ids}
            pred_id = global_node_dict.get(last_graph.node_ids[pred.item()]) if pred.item() < len(last_graph.node_ids) else None
            label_id = global_node_dict.get(last_graph.node_ids[y_task.item()]) if y_task.item() < len(last_graph.node_ids) else None

            all_pred_ids.append(last_graph.node_ids[pred.item()] if pred_id in valid_global_indices else "INVALID")
            all_true_ids.append(last_graph.node_ids[y_task.item()] if label_id in valid_global_indices else "INVALID")

    from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

    valid_indices = [i for i, pred_id in enumerate(all_pred_ids) if pred_id != "INVALID"]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    precision = precision_score(filtered_labels, filtered_preds, average='macro')
    recall = recall_score(filtered_labels, filtered_preds, average='macro')
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    acc = accuracy_score(filtered_labels, filtered_preds)
    filtered_topk_hits = [topk_hits[i] for i in valid_indices]
    top_k_accuracy = sum(filtered_topk_hits) / len(filtered_topk_hits) if filtered_topk_hits else 0.0

    out_of_scope_count = len(all_preds) - len(filtered_preds)
    out_of_scope_rate = out_of_scope_count / len(all_preds)

    mae = mean_absolute_error(time_labels, time_preds)
    mse = mean_squared_error(time_labels, time_preds)
    rmse = mse ** 0.5
    r2 = max(0, r2_score(time_labels, time_preds))

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

    return {
        "accuracy": acc,
        "top_k_accuracy": top_k_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(filtered_labels, filtered_preds),
        "true_node_ids": all_true_ids,
        "pred_node_ids": all_pred_ids,
        "mae": mae_real,
        "rmse": rmse_real,
        "r2": r2_real,
        "out_of_scope_rate": out_of_scope_rate
    }

def prepare_data(normal_graphs):
    all_sequences = []
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

        node_features, active_mask, timestamps = [], [], []
        for node_id, node_data in graph.nodes(data=True):
            simplified_name = simplify_bpmn_id(node_id)
            global_node_set.add(simplified_name)
            features = [float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0 for attr in node_attrs]
            node_features.append(features)
            active_mask.append(1.0 if node_id in current_nodes else 0.0)
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
        "DURATION_", "START_TIME_", "PROC_KEY_", "active_executions", "user_compl_login",
        "SEQUENCE_COUNTER_", "taskaction_code", "task_status"
    ]
    selected_edge_attrs = ["DURATION_E"]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів (динамічні)", total=len(normal_graphs)):
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

        sequence = []
        for i in range(1, len(executed)):
            current_nodes = executed[:i]
            executed_seq = [graph.nodes[n].get("SEQUENCE_COUNTER_", 0) for n in current_nodes]
            max_seq = max(executed_seq)
            candidates = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > max_seq]
            if not candidates:
                continue
            next_node = min(candidates, key=lambda n: graph.nodes[n]["SEQUENCE_COUNTER_"])

            data = transform_graph(graph, current_nodes, next_node, selected_node_attrs, selected_edge_attrs, doc_info, selected_doc_attrs)
            if data:
                sequence.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

        if sequence:
            all_sequences.append(sequence)

    global_node_dict = {name: idx for idx, name in enumerate(sorted(global_node_set))}

    return all_sequences, max_features, max_doc_dim, global_node_dict

def prepare_data_log_only(normal_graphs):
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


