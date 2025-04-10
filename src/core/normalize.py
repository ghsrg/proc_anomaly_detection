import pandas as pd
from src.utils.logger import get_logger
from src.utils.file_utils import load_register, save_register, load_graph, save_graph, save_statistics_to_json
from src.utils.file_utils_l import join_path, is_file_exist
from src.utils.visualizer import visualize_distribution
from src.config.config import NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH, NORMAL_GRAPH_PATH, \
    ANOMALOUS_GRAPH_PATH, PROCESSED_DATA_PATH
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import gc
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from tqdm import tqdm

logger = get_logger(__name__)

def normalize_all_graphs(min_nodes, min_edges):
    """
    Нормалізує всі графи (нормальні та аномальні) і створює нові реєстри.
    """
    logger.info("Запуск нормалізації графів...")


    normal_graphs = load_register("normal_graphs")
    anomalous_graphs = load_register("anomalous_graphs")

    if normal_graphs.empty:## or anomalous_graphs.empty:
        raise ValueError("Реєстри графів порожні. Перевірте дані!")

    # Отримання імен файлів без шляхів
    normal_file_names = normal_graphs["graph_path"]
    anomalous_file_names = anomalous_graphs["graph_path"]
    # Отримання інфи по документам
    normal_docs_info = list(normal_graphs["doc_info"])
    anomalous_docs_info = list(anomalous_graphs["doc_info"])

    #print(normal_docs_info)
    # Розрахунок статистики для нормалізації
    all_paths = [join_path([NORMAL_GRAPH_PATH, file_name]) for file_name in normal_file_names] + \
                [join_path([ANOMALOUS_GRAPH_PATH, file_name]) for file_name in
                 anomalous_file_names]  # Розрахунок статистики для нормалізації
    all_docs_info = normal_docs_info + anomalous_docs_info
    stats = calculate_global_statistics(all_paths, all_docs_info, min_nodes, min_edges)
    print(f"Статистика по вузлах: {stats['node_count']}")
    print(f"Статистика по звя'зках: {stats['edge_count']}")

    stat_path = join_path([PROCESSED_DATA_PATH, "normalized_statistics"])
    save_statistics_to_json(stats, stat_path)

    normalized_normal_graphs = []
    normalized_anomalous_graphs = []

    total_graphs = len(normal_graphs)

    prev_progress_percent = 0

    # Нормалізація та збереження нормальних графів
    #for idx, (_, row) in enumerate(normal_graphs.iterrows(), start=1):
    for idx, (_, row) in enumerate(tqdm(normal_graphs.iterrows(), desc="Обробка нормальних графів", total=len(normal_graphs)),
                                       start=1):
        save_path = join_path([NORMALIZED_NORMAL_GRAPH_PATH, row["graph_path"]])
        if is_file_exist(save_path):
            print(f"Normalized normal graph {save_path} exists")
            continue
        load_path = join_path([NORMAL_GRAPH_PATH, row["graph_path"]])
        graph = load_graph(load_path)

        # Оновлення прогресу
        progress_percent = (idx / total_graphs) * 100
        if progress_percent - prev_progress_percent >= 1:
            #print(f"Нормалізовано нормальних графів: {idx}/{total_graphs} ({progress_percent:.2f}%)")
            prev_progress_percent = progress_percent

        # Перевірка на мінімальну кількість вузлів та зв'язків
        if len(graph.nodes) < min_nodes or len(graph.edges) < min_edges:
            logger.warning(f"Граф {row['graph_path']} пропущено через недостатню кількість вузлів або зв'язків.")
            continue

        normalized = normalize_graph(graph, stats)

        save_graph(normalized, save_path)
        del graph
        del normalized
        gc.collect()
        #  Нормалізуємо атрибути документа
        doc_info = normalize_doc_info(row["doc_info"], stats)

        # Додавання оновленого запису до реєстру
        normalized_normal_graphs.append({
            "id": row["id"],
            "doc_id": row["doc_id"],
            "root_proc_id": row["root_proc_id"],
            "graph_path": row["graph_path"],
            "date": row["date"],
            "params": row["params"],
            'doc_info': doc_info  # Збереження параметрів документа
        })

    # Нормалізація та збереження аномальних графів

    total_anormalgraphs = len(anomalous_graphs)
    prev_progress_percent = 0

    # Нормалізація та збереження аномальних графів
    #for idx, (_, row) in enumerate(anomalous_graphs.iterrows(), start=1):
    for idx, (_, row) in enumerate(tqdm(anomalous_graphs.iterrows(), desc="Обробка аномальних графів", total=len(anomalous_graphs)),
            start=1):
        save_path = join_path([NORMALIZED_ANOMALOUS_GRAPH_PATH, row["graph_path"]])
        if is_file_exist(save_path):
            print(f"Normalized anomalous graph {save_path} exists")
            continue
        load_path = join_path([ANOMALOUS_GRAPH_PATH, row["graph_path"]])
        graph = load_graph(load_path)

        # Оновлення прогресу
        progress_percent = (idx / total_anormalgraphs) * 100
        if progress_percent - prev_progress_percent >= 1:
            #print(f"Нормалізовано аномальних графів: {idx}/{total_anormalgraphs} ({progress_percent:.2f}%)")
            prev_progress_percent = progress_percent

        # Перевірка на мінімальну кількість вузлів та зв'язків
        if len(graph.nodes) < min_nodes or len(graph.edges) < min_edges:
            logger.warning(f"Граф {row['graph_path']} пропущено через недостатню кількість вузлів або зв'язків.")
            continue

        #  Нормалізуємо граф
        normalized = normalize_graph(graph, stats)
        save_graph(normalized, save_path)
        del graph
        del normalized
        gc.collect()
        #  Нормалізуємо атрибути документа
        doc_info = normalize_doc_info(row["doc_info"], stats)

        # Додавання оновленого запису до реєстру
        normalized_anomalous_graphs.append({
            "id": row["id"],
            "doc_id": row["doc_id"],
            "root_proc_id": row["root_proc_id"],
            "graph_path": row["graph_path"],
            "date": row["date"],
            "params": row["params"],
            'doc_info': doc_info  # Збереження параметрів документа
        })
        #print(normalized_anomalous_graphs)
        # Збереження оновлених реєстрів
    save_register(pd.DataFrame(normalized_normal_graphs), "normalized_normal_graphs")
    save_register(pd.DataFrame(normalized_anomalous_graphs), "normalized_anomalous_graphs")

    logger.info("Нормалізація графів завершена.")

def process_single_graph(graph_path, min_nodes, min_edges):
    graph = load_graph(graph_path)
    result = {
        "node_numeric": defaultdict(list),
        "node_text": defaultdict(set),
        "edge_numeric": defaultdict(list),
        "edge_text": defaultdict(set),
        "node_count": [],
        "edge_count": []
    }

    if len(graph.nodes) >= min_nodes and len(graph.edges) >= min_edges:
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        result["node_count"].append(node_count)
        result["edge_count"].append(edge_count)

        for node, data in graph.nodes(data=True):
            for attr, value in data.items():
                if attr == 'TASK_ID_' or attr == 'taskaction_code':
                    continue
                if isinstance(value, (int, float)) and not np.isnan(value):
                    result["node_numeric"][attr].append(value)
                elif isinstance(value, str):
                    result["node_text"][attr].add(value)

        for _, _, data in graph.edges(data=True):
            for attr, value in data.items():
                if attr == 'id' or attr == 'taskaction_code':
                    continue
                if isinstance(value, (int, float)) and not np.isnan(value):
                    result["edge_numeric"][attr].append(value)
                elif isinstance(value, str):
                    if attr == 'DURATION_E' or attr == 'duration_work_E':
                        continue
                    result["edge_text"][attr].add(value)

    del graph
    #gc.collect()
    return result

def calculate_global_statistics(graph_paths, all_docs_info, min_nodes=6, min_edges=6):
    """
    Розраховує глобальні статистики для нормалізації атрибутів графів, включаючи кількість вузлів та зв'язків.

    :param graph_paths: Список шляхів до графів.
    :param all_docs_info: Список атрибутів документів у форматі JSON.
    :param min_edges:  Мінімальна кількість вузлів для графа.
    :param min_nodes: Мінімальна кількість зв'язків для графа.
    :return: Словник статистик для нормалізації.
    """
    start_time = datetime.now()
    print(f"Час початку статистики: {start_time}")

    # Злиття результатів обробки
    def merge_results(all_results):
        stats = {
            "node_numeric": defaultdict(list),
            "node_text": defaultdict(set),
            "edge_numeric": defaultdict(list),
            "edge_text": defaultdict(set),
            "doc_numeric": defaultdict(list),
            "doc_text": defaultdict(set),
            "node_count": [],
            "edge_count": []
        }

        for result in all_results:
            for key, value in result.items():
                if key in ["node_count", "edge_count"]:
                    stats[key].extend(value)
                elif key in ["node_numeric", "edge_numeric"]:
                    for attr, vals in value.items():
                        stats[key][attr].extend(vals)
                elif key in ["node_text", "edge_text"]:
                    for attr, vals in value.items():
                        stats[key][attr].update(vals)

        final_stats = {
            "node_numeric": {
                attr: {
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for attr, values in stats["node_numeric"].items() if values
            },
            "node_text": {
                attr: {
                    "unique_values": sorted(values)
                }
                for attr, values in stats["node_text"].items()
            },
            "edge_numeric": {
                attr: {
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for attr, values in stats["edge_numeric"].items() if values
            },
            "edge_text": {
                attr: {
                    "unique_values": sorted(values)
                }
                for attr, values in stats["edge_text"].items()
            },
            "doc_numeric": {
                attr: {
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for attr, values in stats["doc_numeric"].items() if values
            },
            "doc_text": {
                attr: {
                    "unique_values": sorted(values)
                }
                for attr, values in stats["doc_text"].items()
            },
            "node_count": {
                "min": np.min(stats["node_count"]),
                "max": np.max(stats["node_count"])
            },
            "edge_count": {
                "min": np.min(stats["edge_count"]),
                "max": np.max(stats["edge_count"])
            }
        }

        return final_stats

    # Паралельна обробка
    process_func = partial(process_single_graph, min_nodes=min_nodes, min_edges=min_edges)
    num_cores = multiprocessing.cpu_count() // 2  # Використовувати половину ядер
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = []
        for i, result in enumerate(
                tqdm(executor.map(process_func, graph_paths), total=len(graph_paths), desc="Обробка графів")):
            results.append(result)

    final_stats = merge_results(results)

    # Обробка документів
    for doc_info in all_docs_info:
        for attr, value in doc_info.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                final_stats["doc_numeric"].setdefault(attr, {"min": float("inf"), "max": float("-inf")})
                final_stats["doc_numeric"][attr]["min"] = min(final_stats["doc_numeric"][attr]["min"], value)
                final_stats["doc_numeric"][attr]["max"] = max(final_stats["doc_numeric"][attr]["max"], value)
            elif isinstance(value, str):
                final_stats["doc_text"].setdefault(attr, {"unique_values": set()})
                final_stats["doc_text"][attr]["unique_values"].add(value)

    # Перетворення множин у списки
    for key in ["doc_text"]:
        for attr, values in final_stats[key].items():
            final_stats[key][attr]["unique_values"] = sorted(values["unique_values"])

    stat_path = join_path([PROCESSED_DATA_PATH, f'global_statistics'])
    save_statistics_to_json(final_stats, stat_path)

    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Час завершення статистики: {end_time}")
    print(f"Тривалість статистики: {training_duration}")

    return final_stats

def calculate_global_statistics_old(graph_paths, all_docs_info, min_nodes=6, min_edges=6):
    """
    Розраховує глобальні статистики для нормалізації атрибутів графів, включаючи кількість вузлів та зв'язків.

    :param graph_paths: Список шляхів до графів.
    :param all_docs_info: Список атрибутів документів у форматі JSON.
    :param min_edges:  Мінімальна кількість вузлів для графа.
    :param min_nodes: Мінімальна кількість зв'язків для графа.
    :return: Словник статистик для нормалізації.
    """
    # Фіксація часу початку нормалізації
    start_time = datetime.now()
    print(f"Час початку статистики: {start_time}")
    logger.info(f"Час початку статистики: {start_time}")


    stats = {
        "node_numeric": defaultdict(list),
        "node_text": defaultdict(set),
        "edge_numeric": defaultdict(list),
        "edge_text": defaultdict(set),
        "doc_numeric": defaultdict(list),
        "doc_text": defaultdict(set),
        "node_count": [],
        "edge_count": []
    }
    node_distribution = defaultdict(int)
    edge_distribution = defaultdict(int)

    total_graphs = len(graph_paths)
    prev_progress_percent = 0

    for idx, graph_path in enumerate(graph_paths, start=1):
        graph = load_graph(graph_path)
        if idx > 500:
            break
        # Оновлення прогресу
        progress_percent = (idx / total_graphs) * 100
        if progress_percent - prev_progress_percent >= 1:
            print(f"Зібрано статистику з {idx}/{total_graphs} ({progress_percent:.2f}%)")
            prev_progress_percent = progress_percent

        # Перевірка на мінімальну кількість вузлів та зв'язків
        if len(graph.nodes) < min_nodes or len(graph.edges) < min_edges:
            logger.warning(f"Граф {graph_path} пропущена статистика через недостатню кількість вузлів або зв'язків.")
            continue

        # Додаємо статистику по кількості вузлів та зв'язків
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        stats["node_count"].append(node_count)
        stats["edge_count"].append(edge_count)

        # Оновлення розподілу вузлів і зв'язків
        node_distribution[node_count] += 1
        edge_distribution[edge_count] += 1

        # Збір статистики по атрибутах вузлів
        for node, data in graph.nodes(data=True):
            for attr, value in data.items():
                if attr == "TASK_ID_":
                    continue
                if isinstance(value, (int, float)):
                    stats["node_numeric"][attr].append(0.0 if np.isnan(value) else float(value))
                elif isinstance(value, str):
                    stats["node_text"][attr].add(value)


        # Збір статистики по атрибутах зв'язків
        for _, _, data in graph.edges(data=True):
            for attr, value in data.items():
                if attr == "id":
                    continue
                if isinstance(value, (int, float)):
                    stats["edge_numeric"][attr].append(0.0 if np.isnan(value) else float(value))
                elif isinstance(value, str):
                    stats["edge_text"][attr].add(value)

        # Збір статистики по атрибутах документів
        for doc_info in all_docs_info:
            for attr, value in doc_info.items():
                if isinstance(value, (int, float)):
                    stats["doc_numeric"][attr].append(0.0 if np.isnan(value) else float(value))
                elif isinstance(value, str):
                    stats["doc_text"][attr].add(value)

        del graph
    gc.collect()
    ###***
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Тривалість збору даних: {duration}")
    ###***

    final_stats = {
        "node_numeric": {
            attr: {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
            for attr, values in stats["node_numeric"].items() if values
        },
        "node_text": {
            attr: {
                "unique_values": sorted(values)
            }
            for attr, values in stats["node_text"].items()
        },
        "edge_numeric": {
            attr: {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
            for attr, values in stats["edge_numeric"].items() if values
        },
        "edge_text": {
            attr: {
                "unique_values": sorted(values)
            }
            for attr, values in stats["edge_text"].items()
        },
        "doc_numeric": {
            attr: {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
            for attr, values in stats["doc_numeric"].items() if values
        },
        "doc_text": {
            attr: {
                "unique_values": sorted(values)
            }
            for attr, values in stats["doc_text"].items()
        },
        "node_count": {
            "min": np.min(stats["node_count"]),
            "max": np.max(stats["node_count"]),
            "mean": np.mean(stats["node_count"]),
            "std": np.std(stats["node_count"])
        },
        "edge_count": {
            "min": np.min(stats["edge_count"]),
            "max": np.max(stats["edge_count"]),
            "mean": np.mean(stats["edge_count"]),
            "std": np.std(stats["edge_count"])
        }
    }
    ###***
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Час оновлення списків: {end_time}")
    print(f"Тривалість оновлення списків: {duration}")
    ###***
    stat_path = join_path([PROCESSED_DATA_PATH, f'global_statistics'])
    save_statistics_to_json(final_stats, stat_path)
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Час завершення статистики: {end_time}")
    print(f"Тривалість статистики: {training_duration}")
    return final_stats


def normalize_graph(graph: nx.Graph, global_stats: dict) -> nx.Graph:
    """
    Нормалізує дані графа на основі глобальної статистики та додає всі можливі атрибути.

    :param graph: Граф NetworkX, який потрібно нормалізувати.
    :param global_stats: Словник зі статистикою для нормалізації.
    :return: Нормалізований граф.
    """
    normalized_graph = graph.copy()

    # Нормалізація вузлів
    for node, data in normalized_graph.nodes(data=True):
        for attr in global_stats["node_numeric"]:
            value = data.get(attr, 0.0)
            try:
                value = float(value) if not np.isnan(value) else 0.0
            except (ValueError, TypeError):
                value = 0.0
            stats = global_stats["node_numeric"][attr]
            if stats["max"] > stats["min"]:
                data[attr] = (value - stats["min"]) / (stats["max"] - stats["min"])
            else:
                data[attr] = 0.0

        for attr in global_stats["node_text"]:
            mapping = {v: i for i, v in enumerate(global_stats["node_text"][attr]["unique_values"])}
            data[attr] = mapping.get(data.get(attr, -1), -1)

    # Нормалізація ребер
    for u, v, data in normalized_graph.edges(data=True):
        for attr in global_stats["edge_numeric"]:
            value = data.get(attr, 0.0)
            try:
                value = float(value) if not np.isnan(value) else 0.0
            except (ValueError, TypeError):
                value = 0.0
            stats = global_stats["edge_numeric"][attr]
            if stats["max"] > stats["min"]:
                data[attr] = (value - stats["min"]) / (stats["max"] - stats["min"])
            else:
                data[attr] = 0.0

        for attr in global_stats["edge_text"]:
            mapping = {v: i for i, v in enumerate(global_stats["edge_text"][attr]["unique_values"])}
            data[attr] = mapping.get(data.get(attr, -1), -1)

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


def normalize_doc_info(doc_info: dict, global_stats: dict) -> dict:
    """
    Нормалізує дані документа на основі глобальної статистики.

    :param doc_info: Словник з атрибутами документа.
    :param global_stats: Словник зі статистикою для нормалізації (doc_numeric і doc_text).
    :return: Нормалізований словник документа.
    """
    normalized_doc = {}

    # Нормалізація числових атрибутів документа
    for attr in global_stats["doc_numeric"]:
        value = doc_info.get(attr, 0.0)
        #print('doc_numeric', attr, value)
        try:
            value = float(value) if not np.isnan(value) else 0.0
        except (ValueError, TypeError):
            value = 0.0
        stats = global_stats["doc_numeric"][attr]
        if stats["max"] > stats["min"]:
            normalized_doc[attr] = (value - stats["min"]) / (stats["max"] - stats["min"])
        else:
            normalized_doc[attr] = 0.0

    # Нормалізація текстових атрибутів документа
    for attr in global_stats["doc_text"]:
        mapping = {v: i for i, v in enumerate(global_stats["doc_text"][attr]["unique_values"])}
        normalized_doc[attr] = mapping.get(doc_info.get(attr, -1), -1)
        #print('doc_text', attr, normalized_doc[attr])

    # Логування некоректних значень
    for attr, value in normalized_doc.items():
        if isinstance(value, (float, np.number)) and np.isnan(value):
            logger.error(f"Документ має NaN у атрибуті {attr}")

    return normalized_doc
