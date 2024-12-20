import os
import pandas as pd
import h5py
import networkx as nx
from src.utils.file_utils_l import make_dir, join_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_to_parquet(df: pd.DataFrame, file_name: str):
    """
    Зберігає сирі дані у форматі Parquet.
    :param df: DataFrame для збереження.
    :param file_name: Назва файлу для збереження.
    """
    raw_data_path = os.path.join("data", "raw", f"{file_name}.parquet")
    df.to_parquet(raw_data_path, engine="pyarrow", index=False)
    logger.info(f"Дані збережено у {raw_data_path}")

def load_raw_data(file_name: str) -> pd.DataFrame:
    """
    Завантажує сирі дані з файлу Parquet.
    :param file_name: Назва файлу для завантаження.
    :return: DataFrame із завантаженими даними.
    """
    raw_data_path = os.path.join("data", "raw", f"{file_name}.parquet")
    df = pd.read_parquet(raw_data_path, engine="pyarrow")
    logger.info(f"Дані завантажено з {raw_data_path}")
    return df

def save_to_hdf5(data: dict, file_path: str):
    """
    Зберігає дані у формат HDF5.
    :param data: Словник із даними (назва набору -> масив/список/дані).
    :param file_path: Шлях до файлу.
    """
    with h5py.File(file_path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)
    logger.info(f"Дані збережено у HDF5: {file_path}")

def read_from_hdf5(file_path: str) -> dict:
    """
    Зчитує дані з HDF5 у словник.
    :param file_path: Шлях до файлу.
    :return: Словник із даними (назва набору -> дані).
    """
    data = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
    print(f"Дані зчитано з HDF5: {file_path}")
    return data

def save_graph(graph, filename, format="graphml"):
    """
    Зберігає граф у файл у зазначеному форматі.
    :param graph: NetworkX граф.
    :param filename: Назва файлу для збереження.
    :param format: Формат збереження (graphml, pickle, json).
    """
    supported_formats = ["graphml", "pickle", "json"]
    if format not in supported_formats:
        raise ValueError(f"Формат {format} не підтримується. Доступні формати: {supported_formats}")

    # Збереження у вибраному форматі
    if format == "graphml":
        nx.write_graphml(graph, filename)
    elif format == "pickle":
        nx.write_gpickle(graph, filename)
    elif format == "json":
        data = nx.node_link_data(graph)
        with open(filename, "w") as f:
            import json
            json.dump(data, f)


def save_graphs(process_graph, path):
    """
    Зберігає графи у вигляді файлів для кожного процесу.
    """

    for node, attrs in process_graph.nodes(data=True):

        if attrs.get("type") == "process":
            graph_file = join_path([path, f"{node}.graphml"])
            subgraph = nx.ego_graph(process_graph, node)
            nx.write_graphml(subgraph, graph_file)
            logger.info(f"Граф процесу {node} збережено у {graph_file}.")

