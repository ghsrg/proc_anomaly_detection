import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool  # Для сумісності стилю

logger = get_logger(__name__)


class MLP_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=None):
        super(MLP_pr, self).__init__()

        # Проекція вузлів
        self.node_fc1 = nn.Linear(input_dim, hidden_dim)
        self.node_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Проекція документа
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Об'єднання вузлів і документа
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bnf = nn.BatchNorm1d(hidden_dim * 2)

        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

        # Виходи
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

        # Пулінг для сумісності стилю (хоча для MLP x вже буде [batch_size, hidden_dim])
        self.global_pool = global_mean_pool

    def forward(self, data):
        x = data.x  # [batch_size, input_dim]
        doc_features = getattr(data, 'doc_features', None)
        batch = getattr(data, 'batch', None)

        # Обробка вузлових ознак
        x = self.activation(self.node_fc1(x))
        x = self.bn1(x)
        x = self.activation(self.node_fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        # Пулінг: для MLP зробимо вигляд, що batch — це усі різні приклади
        if batch is None:
            batch = torch.arange(x.size(0), device=x.device)
        x = self.global_pool(x, batch)

        # Обробка документу
        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features))
        else:
            doc_emb = torch.zeros(x.shape[0], self.doc_fc.out_features, device=x.device)

        # Об'єднання вузлів і документа
        fusion = torch.cat([x, doc_emb], dim=1)
        fusion = self.bnf(fusion)
        fusion = self.activation(self.fusion_head(fusion))
        fusion = self.dropout(fusion)

        # Виходи
        task_output = self.task_head(fusion)
        time_output = self.time_head(fusion)

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

def train_epoch(model, data, optimizer, batch_size=64, alpha=0, global_node_dict=None, train_activity_counter=None):
    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", position=1, leave=False, dynamic_ncols=True):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        batch_tensor = torch.cat([
            torch.full((item.x.size(0),), idx, dtype=torch.long)
            for idx, item in enumerate(batch)
        ], dim=0)

        x = torch.cat([item.x for item in batch], dim=0)
        doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None

        y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
        y_time = torch.stack([item.time_target for item in batch]).view(-1)

        # Переносимо на пристрій
        x, batch_tensor, doc_features = [
            t.to(model.task_head.weight.device) if t is not None else None
            for t in [x, batch_tensor, doc_features]
        ]
        y_task = y_task.to(model.task_head.weight.device)
        y_time = y_time.to(model.task_head.weight.device)

        optimizer.zero_grad()

        outputs_task, outputs_time = model.forward(
            type("Batch", (object,), {
                "x": x,
                "doc_features": doc_features
            })
        )

        loss_task = nn.CrossEntropyLoss()(outputs_task, y_task)
        loss_time = nn.MSELoss()(outputs_time.view(-1), y_time)
        loss = loss_task + alpha * loss_time

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Заповнення train_activity_counter (як ми раніше робили)
        for item in batch:
            y_label = item.y.item()
            if hasattr(item, 'node_ids'):
                if y_label < len(item.node_ids):
                    node_id = item.node_ids[y_label]
                    node_idx = global_node_dict.get(node_id, None)
                    if node_idx is not None:
                        train_activity_counter[node_idx] += 1
            else:
                if y_label in global_node_dict.values():
                    train_activity_counter[y_label] += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def calculate_statistics(model, val_data, global_node_dict, global_statistics, batch_size=64, top_k=3, train_activity_counter=None):
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

            x = torch.cat([item.x for item in batch], dim=0)
            doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None

            y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
            y_time = torch.stack([item.time_target for item in batch]).view(-1)

            x, doc_features = [
                t.to(model.task_head.weight.device) if t is not None else None
                for t in [x, doc_features]
            ]
            y_task = y_task.to(model.task_head.weight.device)
            y_time = y_time.to(model.task_head.weight.device)

            outputs_task, outputs_time = model.forward(
                type("Batch", (object,), {
                    "x": x,
                    "doc_features": doc_features
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
                if hasattr(item, 'node_ids') and item.node_ids is not None:
                    valid_global_indices = {global_node_dict.get(node_id) for node_id in item.node_ids}
                    pred_id = global_node_dict.get(item.node_ids[pred.item()]) if pred.item() < len(
                        item.node_ids) else None
                    label_id = global_node_dict.get(item.node_ids[label.item()]) if label.item() < len(
                        item.node_ids) else None

                    all_pred_ids.append(item.node_ids[pred.item()] if pred_id in valid_global_indices else "INVALID")
                    all_true_ids.append(item.node_ids[label.item()] if label_id in valid_global_indices else "INVALID")
                else:
                    # Якщо немає node_ids (MLP) — просто записуємо індекси
                    all_pred_ids.append(str(pred.item()))
                    all_true_ids.append(str(label.item()))

    # Фільтрація валідних індексів
    valid_indices = [i for i, pred_id in enumerate(all_pred_ids) if pred_id != "INVALID"]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    # Класифікаційні метрики
    precision = precision_score(filtered_labels, filtered_preds, average='macro')
    recall = recall_score(filtered_labels, filtered_preds, average='macro')
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    acc = accuracy_score(filtered_labels, filtered_preds)
    filtered_topk_hits = [topk_hits[i] for i in valid_indices]
    top_k_accuracy = sum(filtered_topk_hits) / len(filtered_topk_hits) if filtered_topk_hits else 0.0

    out_of_scope_count = len(all_preds) - len(filtered_preds)
    out_of_scope_rate = out_of_scope_count / len(all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    # Регресійні метрики
    mae = mean_absolute_error(time_labels, time_preds)
    mse = mean_squared_error(time_labels, time_preds)
    rmse = mse ** 0.5
    r2 = max(0, r2_score(time_labels, time_preds))

    # Регресійні метрики (денормалізовані)
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

    # Додатковий підрахунок активностей (якщо треба)
    from collections import Counter
    activity_total_counter = Counter()
    activity_correct_counter = Counter()

    for true_label, pred_label in zip(all_labels, all_preds):
        activity_total_counter[true_label] += 1
        if true_label == pred_label:
            activity_correct_counter[true_label] += 1

    activity_train_vs_val_accuracy = {}

    if train_activity_counter is not None:
        for node_idx in global_node_dict.values():
            train_count = train_activity_counter.get(node_idx, 0)
            val_total = activity_total_counter.get(node_idx, 0)
            val_correct = activity_correct_counter.get(node_idx, 0)
            acc_node = (val_correct / val_total) if val_total > 0 else None
            activity_train_vs_val_accuracy[node_idx] = {
                "train_count": train_count,
                "val_accuracy": acc_node
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

def prepare_data_log_only(normal_graphs, max_node_count=None, max_edge_count=None):
    """
    Підготовка даних для навчання MLP (без використання структури графу).
    :param normal_graphs: Реєстр графів.
    :return: Список об'єктів Data для MLP, розміри ознак вузлів і документа, глобальний словник вузлів.
    """
    data_list = []
    max_features = 0
    max_doc_dim = 0
    global_node_set = set()

    def simplify_bpmn_id(raw_id: str) -> str:
        match = re.match(r'^([^_]+_[^_]+)', raw_id)
        return match.group(1) if match else raw_id

    def transform_doc(doc_info, selected_doc_attrs):
        doc_features = []
        for attr in selected_doc_attrs:
            value = doc_info.get(attr, 0)
            try:
                doc_features.append(float(value))
            except (ValueError, TypeError):
                doc_features.append(0.0)
        return torch.tensor(doc_features, dtype=torch.float)

    selected_node_attrs = [
        "DURATION_", "START_TIME_", "PROC_KEY_", "active_executions", "user_compl_login",
        "SEQUENCE_COUNTER_", "taskaction_code", "task_status"
    ]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів для MLP", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_PR_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        # Збираємо глобальні імена вузлів
        for node_id in graph.nodes():
            simplified_name = simplify_bpmn_id(node_id)
            global_node_set.add(simplified_name)

        executed = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > 0]
        if not executed:
            continue

        for i in range(1, len(executed)):
            current_nodes = executed[:i]
            executed_seq = [graph.nodes[n].get("SEQUENCE_COUNTER_", 0) for n in current_nodes]
            max_seq = max(executed_seq)
            candidates = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > max_seq]
            if not candidates:
                continue

            next_node = min(candidates, key=lambda n: graph.nodes[n]["SEQUENCE_COUNTER_"])

            # Ознаки активних вузлів (середнє по вузлах)
            node_features_list = []
            for node_id in current_nodes:
                node_data = graph.nodes[node_id]
                features = [
                    float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                    for attr in selected_node_attrs
                ]
                node_features_list.append(features)

            if not node_features_list:
                continue

            x = torch.tensor(node_features_list, dtype=torch.float)
            x_mean = x.mean(dim=0)  # середнє по активних вузлах

            max_features = max(max_features, x_mean.numel())

            doc_features = transform_doc(doc_info, selected_doc_attrs)
            max_doc_dim = max(max_doc_dim, doc_features.numel())

            # Цільові значення
            inverse_node_map = [simplify_bpmn_id(n) for n in graph.nodes()]
            target_idx = inverse_node_map.index(simplify_bpmn_id(next_node)) if simplify_bpmn_id(next_node) in inverse_node_map else 0

            time_target = torch.tensor([graph.nodes[next_node].get("duration_work", 0.0)], dtype=torch.float)

            data = Data(
                x=x_mean.unsqueeze(0),  # потрібно [batch_size, feature_dim]
                doc_features=doc_features,
                y=torch.tensor([target_idx], dtype=torch.long),
                time_target=time_target,
                node_ids=inverse_node_map  # ⬅️ додано!

            )

            data_list.append(data)

    global_node_dict = {name: idx for idx, name in enumerate(sorted(global_node_set))}

    return data_list, max_features, max_doc_dim, global_node_dict




