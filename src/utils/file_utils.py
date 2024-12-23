import pandas as pd
import h5py
import networkx as nx
from src.utils.file_utils_l import make_dir, join_path
from src.utils.logger import get_logger
from src.config.config import RAW_PATH
import gzip
from pathlib import Path
logger = get_logger(__name__)


def save_to_parquet(df: pd.DataFrame, file_name: str):
    """
    Зберігає сирі дані у форматі Parquet.
    :param df: DataFrame для збереження.
    :param file_name: Назва файлу для збереження.
    """
    raw_data_path = join_path([RAW_PATH, f"{file_name}.parquet"])
    df.to_parquet(raw_data_path, engine="pyarrow", index=False)
    logger.info(f"Дані збережено у {raw_data_path}")

def read_from_parquet(file_name: str, columns=None) -> pd.DataFrame:
    """
    Завантажує сирі дані з файлу Parquet.
    :param file_name: Назва файлу для завантаження.
    :return: DataFrame із завантаженими даними.
    """
   # raw_data_path = os.path.join("data", "raw", f"{file_name}.parquet")
    raw_data_path = join_path([RAW_PATH, f"{file_name}.parquet"])
    df = pd.read_parquet(raw_data_path, engine="pyarrow", columns=columns)
    logger.info(f"Дані завантажено з {raw_data_path}")
    return df

def save_to_hdf5(data: dict, file_name: str):
    """
    Зберігає дані у формат HDF5.
    :param data: Словник із даними (назва набору -> масив/список/дані).
    :param file_path: Шлях до файлу.
    """
    file_path = join_path([RAW_PATH, f"{file_name}.hdf5"])

    with h5py.File(file_path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)
    logger.info(f"Дані збережено у HDF5: {file_path}")

def read_from_hdf5(file_name: str) -> dict:
    """
    Зчитує дані з HDF5 у словник.
    :param file_path: Шлях до файлу.
    :return: Словник із даними (назва набору -> дані).
    """
    data = {}
    file_path = join_path([RAW_PATH, f"{file_name}.hdf5"])
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
    print(f"Дані зчитано з HDF5: {file_path}")
    return data


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

def save_graph(graph: nx.DiGraph, file_name: str, path: str):
    """
    Зберігає граф у форматі GraphML.
    :param graph: Граф NetworkX.
    :param file_name: Назва файлу для збереження.
    :param path: Шлях до папки для збереження.
    """
    make_dir(path)  # Створює папку, якщо її не існує
    file_path = join_path([path, f"{file_name}.graphml"])
    nx.write_graphml(graph, file_path)
    logger.info(f"Граф збережено у {file_path}")


def save_graph_to_zip(graph: nx.DiGraph, file_name: str, path: str):
    """
    Зберігає граф у форматі GraphML в zip-архів.

    :param graph: Граф NetworkX.
    :param file_name: Назва файлу для ідентифікації графа.
    :param zip_file_path: Шлях до zip-архіву для збереження.
    """
    try:
        # Переконайтесь, що директорія існує
        make_dir(path)

        # Шлях до zip-архіву
        zip_file_path = join_path([path, f"{file_name}.graphml.gz"])

        # Запис графа у стиснутий формат
        with gzip.open(zip_file_path, 'wt', encoding='utf-8') as zip_file:
            # Генерація GraphML як рядка
            graphml_data = nx.generate_graphml(graph)
            for line in graphml_data:
                zip_file.write(line)

        logger.info(f"Граф збережено у {zip_file_path}")
    except PermissionError as e:
        logger.error(f"Помилка доступу до {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Невідома помилка при збереженні графа {file_name}: {e}")
        raise


def save_graph_pic(graph: nx.DiGraph, file_name: str, path: str, visualize_func):
    """
    Зберігає граф у вигляді зображення за допомогою переданої функції візуалізації.

    :param graph: Граф NetworkX.
    :param file_name: Назва файлу для збереження.
    :param path: Шлях до папки для збереження.
    :param visualize_func: Функція для візуалізації графа.
    """
    try:
        make_dir(path)  # Створює папку, якщо її не існує
        file_path = join_path([path, f"{file_name}.png"])

        # Використання функції візуалізації
        visualize_func(graph, file_path)

        print(f"Граф збережено у {file_path}")
    except PermissionError as e:
        print(f"Помилка доступу до {path}: {e}")
    except Exception as e:
        print(f"Невідома помилка при збереженні графа {file_name}: {e}")