import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
import re
from tqdm import tqdm

logger = get_logger(__name__)


class Graphormer_pr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim, edge_dim=None, num_nodes=None):
        """
        :param input_dim: Розмірність вхідних ознак вузла.
        :param hidden_dim: Розмірність прихованого простору.
        :param output_dim: Кількість класів для прогнозу (задача класифікації).
        :param doc_dim: Розмірність ознак документа.
        :param edge_dim: (Опціонально) розмірність ознак ребер.
        :param num_nodes: (Опціонально) максимальна кількість вузлів у графі (для позиційного кодування).
        """
        super(Graphormer_pr, self).__init__()
        print(f"📦 Init Graphormer_pr with input_dim={input_dim}, output_dim={output_dim}")

        # Проєкція входу (node features) у прихований простір
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Позиційне кодування: якщо num_nodes задано, використовуємо його, інакше беремо стандартно 500
        num_positions = num_nodes if num_nodes is not None else 500
        self.pos_embedding = nn.Embedding(num_positions, hidden_dim)

        # TransformerEncoder для обробки вузлової послідовності
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        # Використовуємо 4 шари Transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Проєкція документних ознак
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Шари для об'єднання графового і документного представлень
        self.fusion_bn = nn.LayerNorm(hidden_dim * 2)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

        # Вихідні шари: для класифікації та регресії (час)
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.time_head = nn.Linear(hidden_dim, 1)

        # Глобальний пулінг (будемо використовувати mean pooling після трансформера)
        self.global_pool = global_mean_pool

    def forward(self, data):
        """
        :param data: об'єкт torch_geometric.data.Data
               Має поля:
                 - x: [num_nodes, input_dim]
                 - edge_index (не використовується безпосередньо у Graphormer, оскільки структура подається через позиційне кодування)
                 - batch: [num_nodes], індекси графів (якщо є)
                 - doc_features: [batch_size, doc_dim] (опціонально)
                 - timestamps: (опціонально, але тут не використовується)
        """
        x = data.x  # [num_nodes, input_dim]
        batch = getattr(data, 'batch', None)
        doc_features = getattr(data, 'doc_features', None)

        device = x.device
        # Проєкція в прихований простір
        x = self.activation(self.input_proj(x))  # [num_nodes, hidden_dim]

        # Створюємо позиційні індекси на основі порядку вузлів
        pos_ids = torch.arange(x.size(0), device=device) % self.pos_embedding.num_embeddings
        pos_emb = self.pos_embedding(pos_ids)  # [num_nodes, hidden_dim]
        x = x + pos_emb

        # Тепер групуємо вузли по графах (якщо batch не заданий, вважаємо, що це один граф)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        unique_batches = batch.unique()
        x_seq_list = []
        for b in unique_batches:
            x_b = x[batch == b]  # [n_b, hidden_dim]
            x_seq_list.append(x_b)
        max_len = max([t.size(0) for t in x_seq_list])
        # Паддимо послідовності до однакової довжини
        padded_sequences = []
        for t in x_seq_list:
            pad_size = max_len - t.size(0)
            if pad_size > 0:
                pad = torch.zeros(pad_size, t.size(1), device=device)
                t = torch.cat([t, pad], dim=0)
            padded_sequences.append(t)
        x_seq = torch.stack(padded_sequences, dim=0)  # [batch_size, max_len, hidden_dim]

        # Обробка через TransformerEncoder
        x_transformed = self.transformer_encoder(x_seq)  # [batch_size, max_len, hidden_dim]
        # Агрегуємо за допомогою mean pooling (альтернативно можна використовувати CLS-токен)
        x_graph = x_transformed.mean(dim=1)  # [batch_size, hidden_dim]

        # Обробка документних ознак
        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features.to(device)))
        else:
            doc_emb = torch.zeros(x_graph.size(0), self.doc_fc.out_features, device=device)

        # Об'єднання графового представлення з документним
        fusion = torch.cat([x_graph, doc_emb], dim=1)
        fusion = self.fusion_bn(fusion)
        fusion = self.activation(self.fusion_fc(fusion))
        fusion = self.dropout(fusion)

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

def simplify_bpmn_id(raw_id: str) -> str:
    match = re.match(r'^([^_]+_[^_]+)', raw_id)
    return match.group(1) if match else raw_id

def prepare_data(normal_graphs, max_node_count, max_edge_count, limit=100):
    """
    Prepares data for GNN prediction (next activity and time) for dynamic graphs.
    :param normal_graphs: Registry of normal graphs.
    :param max_node_count: Maximum number of nodes across all graphs.
    :param max_edge_count: Maximum number of edges across all graphs.
    :param limit: Limit the number of graphs to process (for testing).
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

    def transform_graph(graph, current_nodes, next_node, node_attrs, edge_attrs, doc_info, selected_doc_attrs,
                        max_node_count, max_edge_count):
        nonlocal max_features, global_node_set

        node_map = {node: idx for idx, node in enumerate(graph.nodes())}
        real_node_ids = list(graph.nodes())
        global_node_set.update(real_node_ids)

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
        real_node_ids.extend([f"padding_{i}" for i in range(pad_size)])

        x = torch.tensor(node_features, dtype=torch.float)
        x = torch.cat([x, torch.tensor(active_mask, dtype=torch.float).view(-1, 1)], dim=1)
        timestamps = torch.tensor(timestamps, dtype=torch.float)

        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])

        real_edges = [(node_map[u], node_map[v]) for u, v in graph.edges()
                      if u in node_map and v in node_map]
        edge_index = torch.tensor(real_edges, dtype=torch.long).t().contiguous() if real_edges else torch.empty((2, 0), dtype=torch.long)

        if edge_attrs and real_edges:
            edge_attr = torch.tensor([
                [float(graph.get_edge_data(u, v).get(attr, 0)) for attr in edge_attrs]
                for u, v in graph.edges()
            ], dtype=torch.float)
        else:
            edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.float)

        if edge_index.shape[1] < max_edge_count:
            padding_size = max_edge_count - edge_index.shape[1]
            edge_index_padding = torch.zeros((2, padding_size), dtype=torch.long)
            edge_attr_padding = torch.zeros((padding_size, edge_attr.shape[1]), dtype=torch.float)

            edge_index = torch.cat([edge_index, edge_index_padding], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr_padding], dim=0)

        doc_features = transform_doc(doc_info, selected_doc_attrs)
        time_target = torch.tensor([graph.nodes[next_node].get("duration_work", 0.0)], dtype=torch.float) if next_node else None

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([node_map[next_node]], dtype=torch.long) if next_node else None,
            doc_features=doc_features,
            time_target=time_target,
            node_ids=real_node_ids,
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
        if count >= limit:
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

        for i in range(1, len(executed)):
            current_nodes = executed[:i]
            executed_seq = [graph.nodes[n].get("SEQUENCE_COUNTER_", 0) for n in current_nodes]
            max_seq = max(executed_seq)
            candidates = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > max_seq]
            if not candidates:
                continue
            next_node = min(candidates, key=lambda n: graph.nodes[n]["SEQUENCE_COUNTER_"])

            data = transform_graph(graph, current_nodes, next_node, selected_node_attrs, selected_edge_attrs,
                                   doc_info, selected_doc_attrs, max_node_count, max_edge_count)
            if data:
                data_list.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

        count += 1

    global_node_dict = {name: idx for idx, name in enumerate(sorted(global_node_set))}

    return data_list, max_features, max_doc_dim, global_node_dict

def prepare_data_log_only(normal_graphs,max_node_count=None):
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


