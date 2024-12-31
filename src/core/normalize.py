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
                    stats["numeric"][attr].append(value)
                elif isinstance(value, str):
                    stats["text"][attr].add(value)
                elif isinstance(value, (np.datetime64, str)):
                    try:
                        stats["date"][attr].append(np.datetime64(value))
                    except ValueError:
                        continue

        for u, v, data in graph.edges(data=True):  # Замість `edge, data`
            for attr, value in data.items():
                if isinstance(value, (int, float)):
                    stats["numeric"][attr].append(value)
                elif isinstance(value, str):
                    stats["text"][attr].add(value)
                elif isinstance(value, (np.datetime64, str)):
                    try:
                        stats["date"][attr].append(np.datetime64(value))
                    except ValueError:
                        continue

    # Обчислення статистик
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
            "unique_count": len(values),
            "most_common": Counter(values).most_common(1)[0][0] if values else None
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
    Нормалізує дані графа на основі глобальної статистики.

    :param graph: Граф NetworkX, який потрібно нормалізувати.
    :param global_stats: Словник зі статистикою для нормалізації.
    :return: Нормалізований граф.
    """
    normalized_graph = graph.copy()

    # Нормалізація вузлів
    if "nodes" in global_stats:
        for node, data in normalized_graph.nodes(data=True):
            for attr, value in data.items():
                if attr in global_stats["nodes"].get("numeric", {}):
                    stats = global_stats["nodes"]["numeric"][attr]
                    if stats["max"] > stats["min"]:
                        data[attr] = (value - stats["min"]) / (stats["max"] - stats["min"])
                elif attr in global_stats["nodes"].get("text", {}):
                    mapping = {v: i for i, v in enumerate(global_stats["nodes"]["text"][attr]["unique_values"])}
                    data[attr] = mapping.get(value, -1)
                elif attr in global_stats["nodes"].get("date", {}):
                    min_date, max_date = global_stats["nodes"]["date"][attr]["min"], global_stats["nodes"]["date"][attr]["max"]
                    if max_date > min_date:
                        data[attr] = (np.datetime64(value) - min_date).astype(float) / (max_date - min_date).astype(float)

    # Нормалізація ребер
    if "edges" in global_stats:
        for u, v, data in normalized_graph.edges(data=True):
            for attr, value in data.items():
                if attr in global_stats["edges"].get("numeric", {}):
                    stats = global_stats["edges"]["numeric"][attr]
                    if stats["max"] > stats["min"]:
                        data[attr] = (value - stats["min"]) / (stats["max"] - stats["min"])
                elif attr in global_stats["edges"].get("text", {}):
                    mapping = {v: i for i, v in enumerate(global_stats["edges"]["text"][attr]["unique_values"])}
                    data[attr] = mapping.get(value, -1)
                elif attr in global_stats["edges"].get("date", {}):
                    min_date, max_date = global_stats["edges"]["date"][attr]["min"], global_stats["edges"]["date"][attr]["max"]
                    if max_date > min_date:
                        data[attr] = (np.datetime64(value) - min_date).astype(float) / (max_date - min_date).astype(float)

    return normalized_graph

