import pandas as pd
import numpy as np
import h5py
import networkx as nx
from src.utils.file_utils_l import make_dir, join_path, is_file_exist
from src.utils.logger import get_logger
from src.config.config import RAW_PATH, REGISTER_PATH
import gzip
import json
import torch
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


def save_graph(graph: nx.DiGraph, file_name: str, path: str = None):
    """
    Зберігає граф у форматі GraphML.
    :param graph: Граф NetworkX.
    :param file_name: Назва файлу для збереження.
    :param path: Шлях до папки для збереження.
    """
    try:
        if path:
            make_dir(path)  # Створює папку, якщо її не існує
            file_path = join_path([path, f"{file_name}.graphml"])
        else:
            file_path = f"{file_name}.graphml"  # Вважаємо, що file_name містить повний шлях

        #file_path = join_path([path, f"{file_name}.graphml"])
        nx.write_graphml(graph, file_path)
        logger.debug(f"Граф збережено у {file_path}")

    except Exception as e:
        logger.error(f"Помилка під час зберігання графа {file_name}: {e}")
        raise


def load_graph(file_name: str, path: str = None) -> nx.DiGraph:
    """
    Завантажує граф у форматі GraphML.

    :param file_name: Назва файлу графа без розширення або повний шлях до файлу.
    :param path: (Необов'язково) Шлях до папки, де зберігається файл.
    :return: Граф NetworkX у вигляді nx.DiGraph.
    """
    try:
        if path:
            file_path = join_path([path, f"{file_name}.graphml"])
        else:
            file_path = f"{file_name}.graphml"  # Вважаємо, що file_name містить повний шлях

        graph = nx.read_graphml(file_path)
        logger.debug(f"Граф завантажено з {file_path}")
        return graph
    except FileNotFoundError:
        logger.error(f"Файл графа {file_name}.graphml не знайдено в {path or 'вказаному шляху'}.")
        raise
    except Exception as e:
        logger.error(f"Помилка під час завантаження графа {file_name}: {e}")
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

def initialize_register(file_name: str, columns:['id']):
    """Ініціалізує реєстр, якщо його ще не існує."""
    file_path = join_path([REGISTER_PATH, f"{file_name}.parquet"])
    if not is_file_exist(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_parquet(file_path, index=False)

def save_register(df: pd.DataFrame, file_name: str):
    """Зберігає реєстр."""
    reg_data_path = join_path([REGISTER_PATH, f"{file_name}.parquet"])
    df.to_parquet(reg_data_path, engine="pyarrow", index=False)
    logger.info(f"Дані збережено у {reg_data_path}")


def load_register(file_name: str, columns=None) -> pd.DataFrame:
    """Завантажує реєстр."""
    reg_data_path = join_path([REGISTER_PATH, f"{file_name}.parquet"])
    df = pd.read_parquet(reg_data_path, engine="pyarrow", columns=columns)
    logger.info(f"Дані завантажено з {reg_data_path}")
    return df

def save_checkpoint(model, optimizer, epoch, loss, file_path, stats=None):
    """
    Зберігає стан моделі, оптимізатора та параметри навчання.

    :param model: PyTorch модель.
    :param optimizer: Оптимізатор моделі.
    :param epoch: Поточна епоха навчання.
    :param loss: Значення поточного loss.
    :param file_path: Шлях для збереження файлу.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "loss": loss,
        "stats": stats if stats else {},  # Додаємо статистику, якщо вона передана
    }
    torch.save(checkpoint, file_path)
    logger.info(f"Чекпоінт збережено у {file_path}.")


def load_checkpoint(file_path, model, optimizer=None, stats=None):
    """
    Завантажує стан моделі та (опціонально) оптимізатора.

    :param file_path: Шлях до файлу збереження.
    :param model: PyTorch модель для завантаження стану.
    :param optimizer: (Необов'язково) Оптимізатор для завантаження стану.
    :return: epoch, loss (якщо вони збережені).
    """
    if not is_file_exist(file_path):
        raise FileNotFoundError(f"Файл чекпоінта не існує за шляхом: {file_path}")

    try:
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if stats:
            stats = checkpoint.get('stats')

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)

        print(f"Чекпоінт завантажено з {file_path}")
        return epoch, loss, stats
    except Exception as e:
        raise RuntimeError(f"Помилка під час завантаження чекпоінта: {file_path}. Деталі: {str(e)}")

def save_training_progress(register_name, progress_data):
    """
    Зберігає прогрес навчання у реєстр.
    :param register_name: Назва реєстру.
    :param progress_data: Дані прогресу.
    """
    register = load_register(register_name)
    register = register.append(progress_data, ignore_index=True)
    save_register(register, register_name)
    logger.info(f"Прогрес навчання збережено у реєстр {register_name}.")

def save_prepared_data(data, input_dim, doc_dim, file_path):
    """
    Зберігає підготовлені дані у файл.
    :param data_list: Список підготовлених об'єктів Data.
    :param input_dim: Вхідний розмір для моделі.
    :param file_path: Шлях для збереження файлу.
    """
    torch.save({"data": data, "input_dim": input_dim, "doc_dim": doc_dim}, file_path)
    print(f"Підготовлені дані збережено у {file_path}")

def load_prepared_data(file_path):
    """
    Завантажує підготовлені дані з файлу.
    :param file_path: Шлях до файлу.
    :return: data_list, input_dim.
    """
    try:
        checkpoint = torch.load(f"{file_path}")
        print(f"Підготовлені дані завантажено з {file_path}")
        return checkpoint["data"], checkpoint["input_dim"], checkpoint["doc_dim"]
    except FileNotFoundError:
        print(f"Файл з підготовленими даними не знайдено: {file_path}")
        return None, None, None

def save_statistics_to_json(stats, file_path):
    """
    Зберігає статистику в JSON файл.

    :param stats: Статистика для збереження.
    :param file_path: Шлях до файлу.
    """

    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(f"{file_path}.json", "w") as f:
        json.dump(stats, f, indent=4, default=convert_to_serializable)

def save2csv(df: pd.DataFrame, file_name: str):
    """Зберігає реєстр."""
    if type(df) != 'pd.DataFrame':
        df = pd.DataFrame(df)
    reg_data_path =  f"{file_name}.xlsx"
    df.to_excel(reg_data_path, index=False)
    logger.info(f"Дані збережено у {reg_data_path}")
