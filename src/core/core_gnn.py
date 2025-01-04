import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from src.utils.logger import get_logger
from src.core.metrics import calculate_precision_recall, calculate_roc_auc, calculate_f1_score
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.utils.file_utils import join_path, load_graph
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH

logger = get_logger(__name__)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Ініціалізація графової нейронної мережі (GNN).

        :param input_dim: Кількість вхідних ознак для вузлів.
        :param hidden_dim: Розмір прихованого шару.
        :param output_dim: Кількість вихідних ознак для вузлів.
        """
        super(GNN, self).__init__()
        #self.conv1 = GCNConv(input_dim, hidden_dim)  # Перший графовий шар
        #self.conv2 = GCNConv(hidden_dim, output_dim)  # Другий графовий шар
        self.conv1 = GATConv(input_dim, hidden_dim)  # Перший графовий шар
        self.conv2 = GATConv(hidden_dim, output_dim)  # Другий графовий шар
        self.activation = nn.ReLU()  # Функція активації для внутрішніх шарів
        self.global_pool = global_mean_pool  # Глобальне пулінгування
        self.final_activation = nn.Sigmoid()  # Активація для класифікації

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Прямий прохід через графову нейронну мережу.

        :param x: Тензор вузлів (node features).
        :param edge_index: Тензор зв’язків (edges).
        :param batch: Інформація про пакети (необов’язково).
        :return: Вихідний тензор після проходження через GNN.
        """
        x = self.activation(self.conv1(x, edge_index, edge_attr))  # Перший графовий шар з активацією
        x = self.conv2(x, edge_index, edge_attr)  # Другий графовий шар
        x = self.global_pool(x, batch)  # Глобальне пулінгування для отримання графових ознак
        x = self.final_activation(x)  # Фінальна активація (Sigmoid)
        return x


def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Prepares data for GNN training.

    :param normal_graphs: Registry of normal graphs.
    :param anomalous_graphs: Registry of anomalous graphs.
    :param anomaly_type: Type of anomaly for training.
    :return: List of Data objects for GNN.
    """
    data_list = []
    max_features = 0

    def infer_graph_attributes(graph):
        """
        Infers numeric, global, and edge attributes for a graph.

        :param graph: NetworkX graph.
        :return: Lists of numeric_attrs, global_attrs, edge_attrs.
        """
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

    def transform_graph(graph, label, node_attrs, edge_attrs):
        """
        Transforms a NetworkX graph into a PyTorch Geometric Data object, використовуючи лише зазначені атрибути.

        :param graph: NetworkX graph.
        :param label: Label of the graph (0 - normal, 1 - anomalous).
        :param node_attrs: Масив назв атрибутів вузлів, які слід додати до навчання.
        :param edge_attrs: Масив назв атрибутів ребер, які слід додати до навчання.
        :return: PyTorch Geometric Data object.
        """
        node_map = {node: idx for idx, node in enumerate(graph.nodes())}
        nonlocal max_features

        # Перевірка наявності зв'язків у графі
        if graph.number_of_edges() == 0:
            logger.warning("Граф без зв'язків. Пропущено.")
            return None


        # **Підготовка вузлових атрибутів**
        node_features = []
        for node_id, node_data in graph.nodes(data=True):
            features = [
                float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                for attr in node_attrs
            ]
            node_features.append(features)

            # Логування вузлових атрибутів
            #selected_node_data = {attr: node_data.get(attr, None) for attr in node_attrs}
            #print(f"Вузол {node_id} (вузлові атрибути): {selected_node_data}")
            #print(f"Атрибути для моделі (вузли): {features}")

        x = torch.tensor(node_features, dtype=torch.float)
        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])
        if torch.isnan(x).any():
            logger.warning(f"Node features contain nan: {x}")
        if x.shape[1] > 0:
            max_features = max(max_features, x.shape[1])
        else:
            logger.warning(f"Graph without numeric node attributes detected.")


        # Підготовка зв'язків (ребер)
        edges = [(node_map[edge[0]], node_map[edge[1]]) for edge in graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


        # Підготовка атрибутів ребер
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
        #print(f"edge_attr.shape: {edge_attr.shape}, edge_index.shape: {edge_index.shape}")
        # Якщо edge_attr відсутній або розмір не відповідає кількості зв'язків
        if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
             edge_attr = torch.zeros((edge_index.shape[1], len(edge_attrs) if edge_attrs else 1), dtype=torch.float)

        # Логування атрибутів ребер
        #print(f"edge_attr: {edge_attr}, shape: {edge_attr.shape if edge_attr is not None else 'None'}")
        #for node1, node2, edge_data in graph.edges(data=True):
        #    edge = (node1, node2)
        #    selected_edge_data = {attr: edge_data.get(attr, None) for attr in edge_attrs}
        #    print(f"Ребро {edge} (атрибути ребер): {selected_edge_data}")

        if torch.isnan(edge_attr).any():
            logger.warning(f"Edge attributes contain nan: {edge_attr}")

        # Перевірка edge_attr
        if edge_attr is None or edge_attr.ndim < 2:
            edge_attr = torch.zeros((graph.number_of_edges(), len(edge_attrs)), dtype=torch.float)

        # Перевірка відповідності
        if edge_index.ndim < 2 or edge_index.shape[1] == 0:
            raise ValueError("edge_index має некоректну форму.")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_attr size {edge_attr.shape[0]} не відповідає кількості зв'язків {edge_index.shape[1]}.")


        # **Глобальні атрибути**
        global_features = []
        for node, node_data in graph.nodes(data=True):
            if node_data.get("type") == "startEvent":
                global_features = [
                    float(node_data.get(attr, 0)) if isinstance(node_data.get(attr), (int, float)) else 0.0
                    for attr in graph.graph.get("global_attrs", [])
                ]
                break

        global_features = torch.tensor(global_features, dtype=torch.float)

        if torch.isnan(global_features).any():
            logger.warning(f"Global features contain nan: {global_features}")

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float),
            global_features=global_features
        )

        return data

    selected_node_attrs = ["type", "DURATION_", "START_TIME_", "END_TIME_", "active_executions", "SEQUENCE_COUNTER_", "overdue_work", "duration_work"]
    selected_edge_attrs = ["DURATION_", "taskaction_code", "overdue_work"]
    #selected_attrs = ["DURATION_"]

    prev_progress_percent = -1  # Ініціалізуємо попередній прогрес на значення поза діапазоном
    for idx, graph_file in enumerate(normal_graphs["graph_path"], start=1):
        full_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        # Логування прогресу завантаження
        total_graphs = len(normal_graphs["graph_path"])
        progress_percent = (idx / total_graphs) * 100
        if progress_percent - prev_progress_percent >= 1:
            print(f"Підготовка нормального графа {idx}/{total_graphs} ({progress_percent:.2f}%)")
            prev_progress_percent = progress_percent  # Оновлюємо попередній прогрес

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        data = transform_graph(graph, 0, selected_node_attrs, selected_edge_attrs)
        if data is None:
            logger.info(f"Пропускаємо нормальний граф {graph_file} відсутні зв'язки")
            continue
        if data.edge_index.size(1) > 5:  # Беремо графи які мають більше 5 зв'язків
            data_list.append(data)
        else:
            logger.info(f"Пропускаємо нормальний граф {graph_file} кількість зв'язків {data.edge_index.size(1)}")

    prev_progress_percent = -1  # Ініціалізуємо попередній прогрес на значення поза діапазоном
    for idx, graph_file in enumerate(anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]["graph_path"], start=1):
        full_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, graph_file])
        graph = load_graph(full_path)

        # Логування прогресу завантаження
        total_graphs = len(anomalous_graphs[anomalous_graphs["params"].str.contains(anomaly_type)]["graph_path"])

        progress_percent = (idx / total_graphs) * 100
        if progress_percent - prev_progress_percent >= 1:
            print(f"Підготовка аномального графа {idx}/{total_graphs} ({progress_percent:.2f}%)")
            prev_progress_percent = progress_percent  # Оновлюємо попередній прогрес

        numeric_attrs, global_attrs, edge_attrs = infer_graph_attributes(graph)
        graph.graph["numeric_attrs"] = numeric_attrs
        graph.graph["global_attrs"] = global_attrs
        graph.graph["edge_attrs"] = edge_attrs

        data = transform_graph(graph, 1, selected_node_attrs, selected_edge_attrs)
        if data is None:
            logger.info(f"Пропускаємо аномальний граф {graph_file} відсутні зв'язки")
            continue
        if data.edge_index.dim() < 2 or data.edge_index.size(1) > 5:  #  Беремо графи які мають більше 5 зв'язків

            # Перевірка на коректність edge_index
            if data.edge_index.numel() == 0 or data.edge_index.size(0) != 2:
                logger.warning(f"Graph {graph_file} має некоректний edge_index. Пропущено.")
                logger.info(f"Graph {graph_file} має некоректний edge_index. Пропущено.")
                continue

            data_list.append(data)
        else:
            logger.info(f"Пропускаємо аномальний граф {graph_file} кількість зв'язків {data.edge_index.size(1)}")

    return data_list, max_features



def train_epoch(model, data, optimizer, batch_size=24, loss_fn=None):
    """
    Виконує одну епоху навчання з розбиттям на пакети (batch).

    :param model: GNN модель.
    :param data: Дані для навчання (список об'єктів Data).
    :param optimizer: Оптимізатор для оновлення ваг моделі.
    :param batch_size: Розмір пакету для навчання.
    :param loss_fn: Функція втрат (опціонально, якщо None, використовується nn.BCELoss).
    :return: Середнє значення втрат за епоху.
    """
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    model.train()
    total_loss = 0
    num_batches = (len(data) + batch_size - 1) // batch_size  # Обчислення кількості пакетів
    progress_percent=0
    for batch_idx in range(num_batches):
        # Обчислення поточного прогресу
        current_iteration = batch_idx + 1
        total_iterations = num_batches
        prev_progress_percent = progress_percent or 0
        progress_percent = (current_iteration / total_iterations) * 100

        # Вивід поточного прогресу
        if progress_percent - prev_progress_percent > 1:  # яко прогрез більше ніж 1 %
            print(f"Ітерація {current_iteration}/{total_iterations} ({progress_percent:.2f}%)")

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        # Підготовка даних для батчу
        x = torch.cat([item.x for item in batch], dim=0)
        edge_index = torch.cat([item.edge_index for item in batch], dim=1)
        edge_attr = torch.cat([item.edge_attr for item in batch], dim=0) if batch[0].edge_attr is not None else None

        logger.debug(f"edge_index.shape: {edge_index.shape}")
        logger.debug(f"edge_attr.shape: {edge_attr.shape}")
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError(f"edge_index.size(1) ({edge_index.shape[1]}) != edge_attr.size(0) ({edge_attr.shape[0]})")

        batch_tensor = torch.cat(
            [torch.full((item.x.size(0),), idx, dtype=torch.long) for idx, item in enumerate(batch)], dim=0
        )
        y = torch.tensor([item.y.item() for item in batch], dtype=torch.float)  # Одна мітка на граф

        # **Додано для перевірки**
        #print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
        # Скидання градієнтів
        optimizer.zero_grad()

        # Прогноз і обчислення втрат
        outputs = model(x, edge_index, edge_attr, batch_tensor)  # Передати batch
        # print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")
        # print(f"outputs.shape: {outputs.shape}, y.shape: {y.shape}")
        logger.debug(outputs, variable_name=f"outputs {outputs} ", max_lines=10, depth=10)
        logger.debug(y, variable_name=f"y {y} ", max_lines=10, depth=10)
        loss = loss_fn(outputs, y.unsqueeze(1))

        # Зворотнє поширення і оновлення ваг
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.info(f"Середнє значення втрат за епоху: {avg_loss}")
    return avg_loss


def calculate_statistics(model, data):
    """
    Розраховує статистику моделі після навчання.

    :param model: Навчена модель.
    :param data: Дані для оцінки (список об'єктів Data).
    :return: Словник зі статистикою.
    """
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for graph in data:
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            batch = graph.batch
            label = graph.y.item()  # Мітка графа

            # Передбачення моделі
            output = model(x, edge_index, edge_attr, batch).item()
            predictions.append(output)
            labels.append(label)

    # Розрахунок метрик
    #if hasattr(model, "requires_roc_auc") and model.requires_roc_auc:
    roc_auc = calculate_roc_auc(labels, predictions)
    #else:
    #    roc_auc = None

    precision, recall = calculate_precision_recall(labels, predictions)

    f1_score = calculate_f1_score(labels, predictions)
    stats = {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "f1_score": f1_score
    }

    return stats


def create_optimizer(model, learning_rate=0.001):
    """
    Створює оптимізатор для навчання моделі.

    :param model: Модель, параметри якої потрібно оптимізувати.
    :param learning_rate: Рівень навчання (learning rate).
    :return: Ініціалізований оптимізатор.
    """
    return Adam(model.parameters(), lr=learning_rate)
