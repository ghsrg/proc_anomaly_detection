import pandas as pd
from src.utils.logger import get_logger
from src.utils.file_utils import load_register, save_register, load_graph, save_graph
from src.utils.file_utils_l import join_path

from src.config.config import  NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH, NORMAL_GRAPH_PATH, ANOMALOUS_GRAPH_PATH
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

logger = get_logger(__name__)


def normalize_all_graphs():
    """
    Нормалізує всі графи (нормальні та аномальні) і створює нові реєстри.
    """
    logger.info("Запуск нормалізації графів...")

    normal_graphs = load_register("normal_graphs")
    anomalous_graphs = load_register("anomalous_graphs")

    if normal_graphs.empty or anomalous_graphs.empty:
        raise ValueError("Реєстри графів порожні. Перевірте дані!")

    # Отримання імен файлів без шляхів
    normal_file_names = normal_graphs["graph_path"]
    anomalous_file_names = anomalous_graphs["graph_path"]

    # Розрахунок статистики для нормалізації
    all_paths = [join_path([NORMAL_GRAPH_PATH, file_name]) for file_name in normal_file_names] + \
                [join_path([ANOMALOUS_GRAPH_PATH, file_name]) for file_name in anomalous_file_names]
    stats = calculate_global_statistics(all_paths)

    normalized_normal_graphs = []
    normalized_anomalous_graphs = []

    # Нормалізація та збереження нормальних графів
    for _, row in normal_graphs.iterrows():
        graph = load_graph(join_path([NORMAL_GRAPH_PATH, row["graph_path"]]))
        normalized = normalize_graph(graph, stats)
        save_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, row["graph_path"]])
        save_graph(normalized, save_path)

        # Додавання оновленого запису до реєстру
        normalized_normal_graphs.append({
            "id": row["id"],
            "doc_id": row["doc_id"],
            "root_proc_id": row["root_proc_id"],
            "graph_path": row["graph_path"],
            "date": row["date"],
            "params": row["params"]
        })

    # Нормалізація та збереження аномальних графів
    for _, row in anomalous_graphs.iterrows():
        graph = load_graph(join_path([ANOMALOUS_GRAPH_PATH, row["graph_path"]]))
        normalized = normalize_graph(graph, stats)
        save_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, row["graph_path"]])
        save_graph(normalized, save_path)

        # Додавання оновленого запису до реєстру
        normalized_anomalous_graphs.append({
            "id": row["id"],
            "doc_id": row["doc_id"],
            "root_proc_id": row["root_proc_id"],
            "graph_path": row["graph_path"],
            "date": row["date"],
            "params": row["params"]
        })

        # Збереження оновлених реєстрів
    save_register(pd.DataFrame(normalized_normal_graphs), "normalized_normal_graphs")
    save_register(pd.DataFrame(normalized_anomalous_graphs), "normalized_anomalous_graphs")

    logger.info("Нормалізація графів завершена.")


def calculate_global_statistics(graph_paths):
    """
    Розраховує глобальні статистики для нормалізації атрибутів графів.

    :param graph_paths: Список шляхів до графів.
    :return: Словник статистик для нормалізації.
    """
    stats = {
        "numeric": defaultdict(list),
        "text": defaultdict(set),
        "date": defaultdict(list)
    }

    for graph_path in graph_paths:
        graph = load_graph(graph_path)

        for node, data in graph.nodes(data=True):
            for attr, value in data.items():
                if isinstance(value, (int, float)):
                    stats["numeric"][attr].append(float(value))
                elif isinstance(value, str):
                    stats["text"][attr].add(value)
                elif isinstance(value, (np.datetime64, str)):
                    try:
                        stats["date"][attr].append(np.datetime64(value))
                    except ValueError:
                        continue

        for _, _, data in graph.edges(data=True):
            for attr, value in data.items():
                if isinstance(value, (int, float)):
                    stats["numeric"][attr].append(float(value))
                elif isinstance(value, str):
                    stats["text"][attr].add(value)
                elif isinstance(value, (np.datetime64, str)):
                    try:
                        stats["date"][attr].append(np.datetime64(value))
                    except ValueError:
                        continue

    # Перетворення в єдиний тип
    numeric_stats = {
        attr: {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "std": np.std(values)
        }
        for attr, values in stats["numeric"].items() if values
    }

    text_stats = {
        attr: {
            "unique_values": sorted(values)
        }
        for attr, values in stats["text"].items()
    }

    date_stats = {
        attr: {
            "min": np.min(values),
            "max": np.max(values)
        }
        for attr, values in stats["date"].items() if values
    }

    return {
        "numeric": numeric_stats,
        "text": text_stats,
        "date": date_stats
    }


def normalize_graph(graph: nx.Graph, global_stats: dict) -> nx.Graph:
    """
    Нормалізує дані графа на основі глобальної статистики та додає всі можливі атрибути.

    :param graph: Граф NetworkX, який потрібно нормалізувати.
    :param global_stats: Словник зі статистикою для нормалізації.
    :return: Нормалізований граф.
    """
    normalized_graph = graph.copy()

    # Додати всі можливі атрибути до вузлів
    for node, data in normalized_graph.nodes(data=True):
        for attr in global_stats["numeric"]:
            if attr not in data:
                data[attr] = 0.0  # За замовчуванням 0 для числових атрибутів
        for attr in global_stats["text"]:
            if attr not in data:
                data[attr] = -1  # За замовчуванням -1 для текстових атрибутів

    # Додати всі можливі атрибути до ребер
    for u, v, data in normalized_graph.edges(data=True):
        for attr in global_stats["numeric"]:
            if attr not in data:
                data[attr] = 0.0  # За замовчуванням 0 для числових атрибутів
        for attr in global_stats["text"]:
            if attr not in data:
                data[attr] = -1  # За замовчуванням -1 для текстових атрибутів

    # Нормалізація вузлів
    for node, data in normalized_graph.nodes(data=True):
        for attr, value in data.items():
            if attr in global_stats["numeric"]:
                stats = global_stats["numeric"][attr]
                try:
                    numeric_value = float(value)
                    if stats["max"] > stats["min"]:
                        data[attr] = (numeric_value - stats["min"]) / (stats["max"] - stats["min"])
                    else:
                        data[attr] = 0.0
                except (ValueError, TypeError):
                    data[attr] = 0.0
            elif attr in global_stats["text"]:
                mapping = {v: i for i, v in enumerate(global_stats["text"][attr]["unique_values"])}
                data[attr] = mapping.get(value, -1)

    # Нормалізація ребер
    for u, v, data in normalized_graph.edges(data=True):
        for attr, value in data.items():
            if attr in global_stats["numeric"]:
                stats = global_stats["numeric"][attr]
                try:
                    numeric_value = float(value)
                    if stats["max"] > stats["min"]:
                        data[attr] = (numeric_value - stats["min"]) / (stats["max"] - stats["min"])
                    else:
                        data[attr] = 0.0
                except (ValueError, TypeError):
                    data[attr] = 0.0
            elif attr in global_stats["text"]:
                mapping = {v: i for i, v in enumerate(global_stats["text"][attr]["unique_values"])}
                data[attr] = mapping.get(value, -1)

    # Логування некоректних значень
    for node, data in normalized_graph.nodes(data=True):
        for attr, value in data.items():
            if isinstance(value, (float, np.number)) and np.isnan(value):
                logger.error(f"Node {node} has NaN in attribute {attr}")

    for u, v, data in normalized_graph.edges(data=True):
        for attr, value in data.items():
            if isinstance(value, (float, np.number)) and np.isnan(value):
                logger.error(f"Edge ({u}, {v}) has NaN in attribute {attr}")

    return normalized_graph



