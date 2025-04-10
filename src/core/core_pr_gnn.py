import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from src.utils.file_utils import join_path, load_graph
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.config.config import NORMALIZED_PR_NORMAL_GRAPH_PATH, LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

class GNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, task_output_dim, time_output_dim, doc_dim, edge_dim=None):
        super(GNNPredictor, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)

        self.doc_fc = nn.Linear(doc_dim, hidden_dim)
        self.global_pool = global_mean_pool
        self.activation = nn.ReLU()

        self.task_head = nn.Linear(hidden_dim * 2, task_output_dim)
        self.time_head = nn.Linear(hidden_dim * 2, time_output_dim)

    def forward(self, data):
        x, edge_index, batch = data.node_features, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_features', None)
        doc_features = getattr(data, 'doc_features', None)

        x = self.activation(self.conv1(x, edge_index, edge_attr))
        x = self.activation(self.conv2(x, edge_index, edge_attr))
        x = self.activation(self.conv3(x, edge_index, edge_attr))
        x = self.global_pool(x, batch)

        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features))
        else:
            doc_emb = torch.zeros(x.shape[0], self.doc_fc.out_features, device=x.device)

        x = torch.cat([x, doc_emb], dim=1)
        task_output = self.task_head(x)
        time_output = self.time_head(x)
        return task_output, time_output
def prepare_data(normal_graphs):
    """
    Prepares data for GNN prediction (next activity and time).

    :param normal_graphs: Registry of normal graphs.
    :return: List of Data objects for GNN.
    """
    from torch_geometric.data import Data
    import torch
    from tqdm import tqdm

    data_list = []
    max_features = 0
    max_doc_dim = 0

    # Визначає числові атрибути вузлів та ребер для подальшої фічеризації
    def infer_graph_attributes(graph):
        numeric_attrs = set()
        global_attrs = set()
        edge_attrs = set()

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

    # Перетворює словник з атрибутами документа на числовий вектор
    def transform_doc(doc_info, selected_doc_attrs):
        doc_features = []
        for attr in selected_doc_attrs:
            value = doc_info.get(attr, 0)
            try:
                doc_features.append(float(value))
            except (ValueError, TypeError):
                doc_features.append(0.0)
        return torch.tensor(doc_features, dtype=torch.float)

    # Основна функція перетворення графа на об'єкт Data для GNN
    def transform_graph(graph, current_nodes, next_node_ids, node_attrs, edge_attrs, doc_info, selected_doc_attrs):
        import torch
        from torch_geometric.data import Data

        node_map = {node: idx for idx, node in enumerate(graph.nodes())}
        nonlocal max_features

        if graph.number_of_edges() == 0:
            logger.warning("Граф без зв'язків. Пропущено.")
            return None

        node_features = []
        active_mask = []   # маркує поточні активні вузли (вже виконані)
        next_mask = []     # мітки — вузли, які мають бути наступними

        for node_id, node_data in graph.nodes(data=True):
            features = [
                float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                for attr in node_attrs
            ]
            node_features.append(features)
            active_mask.append(1.0 if node_id in current_nodes else 0.0)
            next_mask.append(1.0 if node_id in next_node_ids else 0.0)

        x = torch.tensor(node_features, dtype=torch.float)
        active_mask = torch.tensor(active_mask, dtype=torch.float).view(-1, 1)
        next_mask = torch.tensor(next_mask, dtype=torch.float).view(-1, 1)
        x = torch.cat([x, active_mask], dim=1)  # додаємо маркер активності

        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])
        if torch.isnan(x).any():
            logger.warning(f"Node features contain nan: {x}")

        # Побудова зв'язків (ребер)
        edges = [(node_map[edge[0]], node_map[edge[1]]) for edge in graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Обробка атрибутів ребер
        edge_attr = None
        if edge_attrs:
            edge_attr = torch.tensor(
                [
                    [
                        float(edge_data.get(attr, 0)) if isinstance(edge_data.get(attr), (int, float)) else 0.0
                        for attr in edge_attrs
                    ]
                    for _, _, edge_data in graph.edges(data=True)
                ],
                dtype=torch.float
            )

        if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
            edge_attr = torch.zeros((edge_index.shape[1], len(edge_attrs) if edge_attrs else 1), dtype=torch.float)

        if torch.isnan(edge_attr).any():
            logger.warning(f"Edge attributes contain nan: {edge_attr}")

        if edge_index.ndim < 2 or edge_index.shape[1] == 0:
            raise ValueError("edge_index має некоректну форму.")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_attr size {edge_attr.shape[0]} не відповідає кількості зв'язків {edge_index.shape[1]}.")

        doc_features = transform_doc(doc_info, selected_doc_attrs)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=next_mask,  # мітка — вектор, які вузли мають бути виконані наступними
            doc_features=doc_features
        )

        return data

    # Обрані атрибути вузлів, ребер та документа
    selected_node_attrs = [
        "DURATION_", "START_TIME_", "END_TIME_", "active_executions",
        "SEQUENCE_COUNTER_", "overdue_work", "duration_work"
    ]
    selected_edge_attrs = ["DURATION_E"]
    selected_doc_attrs = [
        "PurchasingBudget", "InitialPrice", "FinalPrice", "ExpectedDate",
        "CategoryL1", "CategoryL2", "CategoryL3", "ClassSSD", "Company_SO"
    ]

    # Проходимо по всіх графах і будуємо окремі приклади для кожного кроку процесу
    for idx, row in tqdm(normal_graphs.iterrows(), desc="Обробка графів", total=len(normal_graphs)):
        graph_file = row["graph_path"]
        doc_info = row.get("doc_info", {})
        full_path = join_path([NORMALIZED_PR_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        # Визначаємо вузли, які вже були виконані (SEQUENCE_COUNTER_ > 0)
        executed = [n for n, d in graph.nodes(data=True) if d.get("SEQUENCE_COUNTER_", 0) > 0]

        if not executed:
            continue

        # Створюємо кроки навчання: які вузли виконані, які будуть наступними
        for i in range(1, len(executed) + 1):
            current_nodes = executed[:i]
            next_nodes = []
            for n in current_nodes:
                next_nodes.extend([t for _, t in graph.out_edges(n)])  # потенційні наступні вузли
            next_nodes = list(set(next_nodes) - set(current_nodes))  # виключаємо вже виконані

            data = transform_graph(graph, current_nodes, next_nodes, selected_node_attrs, selected_edge_attrs, doc_info, selected_doc_attrs)
            if data:
                data_list.append(data)
                max_doc_dim = max(max_doc_dim, data.doc_features.numel())

    return data_list, max_features, max_doc_dim



def calculate_task_statistics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            task_logits, _ = model(data)
            preds = torch.argmax(task_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def calculate_time_statistics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _, time_output = model(data)
            all_preds.extend(time_output.view(-1).cpu().numpy())
            all_labels.extend(data.time_target.view(-1).cpu().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    r2 = r2_score(all_labels, all_preds)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def train_epoch(model, data, optimizer, batch_size=24, alpha=1.0):
    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Батчі", unit="батч", leave=False, dynamic_ncols=True):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        batch_tensor = torch.cat([torch.full((item.x.size(0),), idx, dtype=torch.long) for idx, item in enumerate(batch)], dim=0)
        x = torch.cat([item.x for item in batch], dim=0)
        edge_index = torch.cat([item.edge_index for item in batch], dim=1)
        edge_attr = torch.cat([item.edge_attr for item in batch], dim=0) if batch[0].edge_attr is not None else None
        doc_features = torch.stack([item.doc_features for item in batch], dim=0) if batch[0].doc_features is not None else None

        y_task = torch.tensor([item.y.item() for item in batch], dtype=torch.long)
        y_time = torch.stack([item.time_target for item in batch]).view(-1)

        # Переносимо на GPU якщо є
        x, edge_index, edge_attr, batch_tensor, doc_features = [t.to(model.task_head.weight.device) if t is not None else None for t in [x, edge_index, edge_attr, batch_tensor, doc_features]]
        y_task = y_task.to(model.task_head.weight.device)
        y_time = y_time.to(model.task_head.weight.device)

        optimizer.zero_grad()
        outputs_task, outputs_time = model.forward(
            type("Batch", (object,), {
                "node_features": x,
                "edge_index": edge_index,
                "edge_features": edge_attr,
                "batch": batch_tensor,
                "doc_features": doc_features
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
def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    #return Adam(model.parameters(), lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)