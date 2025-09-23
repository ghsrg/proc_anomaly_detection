from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv  # as fallback if TGAT not available
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm

logger = get_logger(__name__)

class GCN2_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=None):
        """
        :param input_dim: розмірність node features
        :param hidden_dim: приховані шари
        :param output_dim: кількість класів
        :param doc_dim: розмірність ознак документа
        """
        super().__init__()
        print(f"📦 Init GCN_pr with input_dim={input_dim}, output_dim={output_dim}")

        # GCN частина
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.global_pool = global_mean_pool

        # Документна частина
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Злиття GCN та документа
        self.bnf = nn.LayerNorm(hidden_dim * 2)
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)

        # Виходи: класифікація дії + регресія часу
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

    def forward(self, data_batch, doc_features=None):
        """
        :param data_batch: Data об'єкт з полями x, edge_index, batch
        :param doc_features: (опц) батч тензор ознак документа [batch_size x doc_dim]
        """
        device = self.task_head.weight.device
        x = data_batch.x.to(device)
        edge_index = data_batch.edge_index.to(device)
        batch = data_batch.batch.to(device)

        # GCN → агрегування
        x = self.activation(self.gcn(x, edge_index))
        x_pooled = self.global_pool(x, batch)

        # Документ
        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features.to(device)))
        else:
            doc_emb = torch.zeros((x_pooled.size(0), self.doc_fc.out_features), device=device)

        # Об'єднання
        x = torch.cat([x_pooled, doc_emb], dim=1)
        x = self.bnf(x)
        x = self.activation(self.fusion_head(x))
        x = self.dropout(x)

        return self.task_head(x), self.time_head(x)
class GCN_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=None):
        super().__init__()
        print(f"📦 Init GCN_pr (deep) with input_dim={input_dim}, output_dim={output_dim}")

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)  # для резідуального зв'язку

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.global_pool = global_mean_pool

        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        self.bnf = nn.LayerNorm(hidden_dim * 2)
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)

        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

    def forward(self, data_batch, doc_features=None):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        device = self.task_head.weight.device
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)

        # Збереження початкового x для резідуального зв'язку
        residual = x
        x = self.activation(self.conv1(x, edge_index))
        x = x + self.input_proj(residual)  # перший скіп

        x = self.activation(self.conv2(x, edge_index))
        x = x + x  # простий self-loop

        x = self.activation(self.conv3(x, edge_index))
        x = x + x  # ще один self-loop

        x_pooled = self.global_pool(x, batch)

        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features.to(device)))
        else:
            doc_emb = torch.zeros((x_pooled.size(0), self.doc_fc.out_features), device=device)

        x = torch.cat([x_pooled, doc_emb], dim=1)
        x = self.bnf(x)
        x = self.activation(self.fusion_head(x))
        x = self.dropout(x)

        return self.task_head(x), self.time_head(x)

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

        if global_node_dict is not None and train_activity_counter is not None:
            for label in y_task.cpu().numpy():
                if label in global_node_dict.values():
                    train_activity_counter[label] += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def calculate_statistics(
    model,
    val_data,
    global_node_dict,
    global_statistics,
    batch_size=64,
    top_k=3,
    train_activity_counter=None,
    min_count=1,
    window_size=10
):
    model.eval()

    # === Глобальні списки ===
    all_preds, all_labels, all_pred_ids, all_true_ids = [], [], [], []
    topk_hits, time_preds, time_labels = [], [], []

    # === prefix-based агрегація ===
    per_prefix = defaultdict(lambda: {
        "accuracy": [], "f1": [], "precision": [], "recall": [],
        "conf": [], "out_of_scope": [],
        **{f"top{k}": [] for k in [1, 3, 5]}
    })

    num_batches = (len(val_data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Валідація (батчами)", unit="батч"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(val_data))
            batch = val_data[start_idx:end_idx]

            # Формування батчу
            batch_tensor = torch.cat([
                torch.full((item.x.size(0),), i, dtype=torch.long)
                for i, item in enumerate(batch)
            ], dim=0)

            x = torch.cat([item.x for item in batch], dim=0)
            edge_index = torch.cat([item.edge_index for item in batch], dim=1)
            edge_attr = torch.cat([item.edge_attr for item in batch], dim=0) if batch[0].edge_attr is not None else None
            doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None
            timestamps = torch.cat([item.timestamps for item in batch], dim=0) if hasattr(batch[0], "timestamps") else None

            y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
            y_time = torch.stack([item.time_target for item in batch]).view(-1)

            device = model.task_head.weight.device
            x, edge_index, edge_attr, batch_tensor, doc_features = [
                t.to(device) if t is not None else None
                for t in [x, edge_index, edge_attr, batch_tensor, doc_features]
            ]
            y_task = y_task.to(device)
            y_time = y_time.to(device)
            if timestamps is not None:
                timestamps = timestamps.to(device)

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
            probs = torch.softmax(outputs_task, dim=1)
            confs = torch.max(probs, dim=1).values.cpu().numpy()
            topk_all = {k: torch.topk(outputs_task, k=k, dim=1).indices.cpu().numpy() for k in [1, 3, 5]}

            # === глобальний збір ===
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_task.cpu().numpy())
            time_preds.extend(outputs_time.view(-1).cpu().numpy())
            time_labels.extend(y_time.cpu().numpy())

            # === по кожному IG ===
            for i, item in enumerate(batch):
                prefix_len = int(item.prefix_mask.sum().item() if hasattr(item, "prefix_mask") else item.x[:, -1].sum().item())
                acc_i = int(preds[i].cpu().item() == y_task[i].cpu().item())
                f1_i = f1_score([y_task[i].cpu().item()], [preds[i].cpu().item()], average="macro", zero_division=0)
                prec_i = precision_score([y_task[i].cpu().item()], [preds[i].cpu().item()], average="macro", zero_division=0)
                rec_i = recall_score([y_task[i].cpu().item()], [preds[i].cpu().item()], average="macro", zero_division=0)
                conf_i = confs[i]

                # top-k
                hits = {k: int(y_task[i].cpu().item() in topk_all[k][i]) for k in [1, 3, 5]}

                per_prefix[prefix_len]["accuracy"].append(acc_i)
                per_prefix[prefix_len]["f1"].append(f1_i)
                per_prefix[prefix_len]["precision"].append(prec_i)
                per_prefix[prefix_len]["recall"].append(rec_i)
                per_prefix[prefix_len]["conf"].append(conf_i)
                for k, v in hits.items():
                    per_prefix[prefix_len][f"top{k}"].append(v)

                # out-of-scope логіка
                valid_global_indices = {global_node_dict.get(node_id) for node_id in item.node_ids}
                pred_id = global_node_dict.get(item.node_ids[preds[i].item()]) if preds[i].item() < len(item.node_ids) else None
                label_id = global_node_dict.get(item.node_ids[y_task[i].item()]) if y_task[i].item() < len(item.node_ids) else None

                pred_invalid = pred_id not in valid_global_indices
                label_invalid = label_id not in valid_global_indices
                out_scope = int(pred_invalid or label_invalid)

                per_prefix[prefix_len]["out_of_scope"].append(out_scope)
                all_pred_ids.append(item.node_ids[preds[i].item()] if not pred_invalid else "INVALID")
                all_true_ids.append(item.node_ids[y_task[i].item()] if not label_invalid else "INVALID")

    # === глобальні метрики ===
    valid_indices = [i for i, pred_id in enumerate(all_pred_ids) if pred_id != "INVALID"]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    precision = precision_score(filtered_labels, filtered_preds, average="macro")
    recall = recall_score(filtered_labels, filtered_preds, average="macro")
    f1 = f1_score(filtered_labels, filtered_preds, average="macro")
    acc = accuracy_score(filtered_labels, filtered_preds)

    filtered_topk_hits = [topk_hits[i] for i in valid_indices] if topk_hits else []
    top_k_accuracy = sum(filtered_topk_hits) / len(filtered_topk_hits) if filtered_topk_hits else 0.0

    out_of_scope_count = len(all_preds) - len(filtered_preds)
    out_of_scope_rate = out_of_scope_count / len(all_preds) if all_preds else 0

    cm = confusion_matrix(all_labels, all_preds) if all_labels else None

    # регресійні метрики
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

    # статистика по активностям
    from collections import Counter
    activity_total_counter = Counter()
    activity_correct_counter = Counter()
    for true_label, pred_label in zip(all_labels, all_preds):
        activity_total_counter[true_label] += 1
        if true_label == pred_label:
            activity_correct_counter[true_label] += 1

    activity_train_vs_val_accuracy = {
        node_idx: {
            "train_count": train_activity_counter.get(node_idx, 0) if train_activity_counter else 0,
            "val_accuracy": (activity_correct_counter[node_idx] / activity_total_counter[node_idx])
            if activity_total_counter[node_idx] > 0 else None
        }
        for node_idx in global_node_dict.values()
    }

    # === Префіксна статистика ===
    prefix_stats = []
    for L, bucket in per_prefix.items():
        row = {"prefix_len": L, "count": len(bucket["accuracy"])}
        for key, values in bucket.items():
            if not values:
                continue
            row[f"{key}_mean"] = np.mean(values)
            row[f"{key}_std"] = np.std(values)
            row[f"{key}_min"] = np.min(values)
            row[f"{key}_max"] = np.max(values)
        prefix_stats.append(row)

    df = pd.DataFrame(prefix_stats).sort_values("prefix_len").reset_index(drop=True)

    if min_count > 1:
        df["count_filt"] = df["count"].copy()
        mask = df["count"] < min_count
        df.loc[mask, df.columns.difference(["prefix_len", "count", "count_filt"])] = np.nan

    if window_size > 1:
        cols = [c for c in df.columns if c not in ["prefix_len", "count", "count_filt"]]
        df[cols] = df[cols].rolling(window=window_size, min_periods=1, center=True).mean()

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
        "activity_train_vs_val_accuracy": activity_train_vs_val_accuracy,
        "prefix_stats": df.to_dict(orient="records")
    }

def calculate_statistics2(model, val_data, global_node_dict, global_statistics, batch_size=64, top_k=3, train_activity_counter=None):
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
                torch.full((item.x.size(0),), i, dtype=torch.long)
                for i, item in enumerate(batch)
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

    from collections import Counter

    # Підрахунок скільки разів зустрілась кожна активність на валідації
    activity_total_counter = Counter()
    activity_correct_counter = Counter()

    for true_label, pred_label in zip(all_labels, all_preds):
        activity_total_counter[true_label] += 1
        if true_label == pred_label:
            activity_correct_counter[true_label] += 1

    activity_train_vs_val_accuracy = {
        node_idx: {
            "train_count": train_activity_counter.get(node_idx, 0) if train_activity_counter is not None else 0,
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

def prepare_data(normal_graphs):
    """
    Prepares data for GNN prediction (next activity and time) with TGAT timestamps variant 3.
    :param normal_graphs: Registry of normal graphs.
    :return: List of Data objects for GNN, max_features, max_doc_dim, global_node_dict
    """
    data_list = []
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

        if graph.number_of_edges() == 0:
            return None

        node_features, active_mask, timestamps = [], [], []
        for node_id, node_data in graph.nodes(data=True):
            simplified_name = simplify_bpmn_id(node_id)
            global_node_set.add(simplified_name)

            features = [
                float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                for attr in node_attrs
            ]
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
                data_list.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

    # Побудова глобального словника
    global_node_dict = {name: idx for idx, name in enumerate(sorted(global_node_set))}

    return data_list, max_features, max_doc_dim, global_node_dict

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

def prepare_data_log_only2(normal_graphs):
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

