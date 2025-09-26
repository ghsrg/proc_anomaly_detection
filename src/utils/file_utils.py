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
import os
import re
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



def aggregate_statistics(directory_path):
    all_results = []

    for filename in os.listdir(directory_path):
        if filename.endswith("statistics.xlsx"):
            full_path = os.path.join(directory_path, filename)

            # Виправлена регулярка
            match = re.match(r"(.+?)(?:_pr)?_(logs|bpmn)_seed(\d+)_statistics\.xlsx", filename)
            if match:
                architecture, data_type, seed = match.groups()

                try:
                    df = pd.read_excel(full_path)

                    if not df.empty:
                        print(f' Aggregate statistics: {filename}')
                        last_row = df.iloc[-1]  # беремо останній рядок
                        result = {
                            "architecture": architecture,
                            "data_type": data_type,
                            "seed": int(seed),
                            "epoch": last_row['epochs'],
                            "train_loss": last_row['train_loss'],
                            "spend_time": last_row['spend_time'],
                            "val_accuracy": last_row['val_accuracy'],
                            "val_top_k_accuracy": last_row['val_top_k_accuracy'],
                            "val_out_of_scope_rate": last_row['val_out_of_scope_rate']
                        }
                        all_results.append(result)
                except Exception as e:
                    print(f"Помилка при обробці {filename}: {e}")

    # Об'єднати все в один DataFrame
    final_df = pd.DataFrame(all_results)
    return final_df

import pandas as pd

def summarize_architecture_metrics(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Формує зведену таблицю по архітектурам, усереднюючи метрики по всіх префіксах.
    Вивід: mean ± std (по prefix_len) для кожної архітектури / data_type / seed.
    """

    metrics = ["accuracy", "f1", "precision", "recall",
               "out_of_scope", "top1", "top3", "top5"]

    results = []
    for (arch, dtype, seed), group in full_df.groupby(["architecture", "data_type", "seed"]):
        row = {"Архітектура": arch, "Тип": dtype, "Seed": seed}
        for m in metrics:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in group.columns and std_col in group.columns:
                mean_val = group[mean_col].mean()
                std_val = group[std_col].mean()
                row[m] = f"{mean_val:.3f} ± {std_val:.3f}"
        results.append(row)

    df_summary = pd.DataFrame(results)

    # впорядкувати колонки
    cols = ["Архітектура", "Тип", "Seed"] + metrics
    return df_summary[cols]

import pandas as pd
import numpy as np

def summarize_by_prefix_bins_mean(full_df: pd.DataFrame, metric: str = "accuracy_mean") -> pd.DataFrame:
    """
    Агрегує статистику по діапазонах довжин префіксів для Logs-only та BPMN-based.

    Parameters
    ----------
    full_df : pd.DataFrame
        Результати після aggregate_prefix_statistics з колонками:
        ['prefix_len', 'accuracy_mean', ..., 'architecture','data_type','seed']
    metric : str
        Метрика для агрегації (наприклад 'accuracy_mean', 'f1_mean').

    Returns
    -------
    summary_df : pd.DataFrame
        Таблиця формату:
        | Префікс (довжина) | Logs-only | BPMN-based |
    """

    # визначаємо бінування довжин
    bins = [0, 10, 50, 100, 250, np.inf]
    labels = ["0–9", "10–49", "50–99", "100–250", ">250"]

    # додаємо категорію
    full_df["prefix_bin"] = pd.cut(full_df["prefix_len"], bins=bins, labels=labels, right=False)

    # групування
    grouped = (
        full_df.groupby(["prefix_bin", "data_type"])[metric]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )

    # перейменування колонок
    grouped = grouped.rename(columns={"prefix_bin": "Префікс (довжина)",
                                      "logs": "Logs-only",
                                      "bpmn": "BPMN-based"})

    return grouped

from scipy.stats import ttest_rel

def paired_ttest_bpmn_vs_logs(full_df: pd.DataFrame):
    """
    Порівнює BPMN-based та Logs-only режими через парний t-тест
    по accuracy для архітектур GNN.
    """
    # Переконаємось, що числові колонки обробляються правильно
    df = full_df.copy()
    for col in ["accuracy_mean"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # усереднити по архітектурі та data_type
    grouped = df.groupby(["architecture", "data_type"])["accuracy_mean"].mean().reset_index()

    # зробимо таблицю архітектура × тип
    pivot = grouped.pivot(index="architecture", columns="data_type", values="accuracy_mean")

    # візьмемо тільки архітектури, де є обидва режими
    pivot = pivot.dropna()

    # підготовка масивів
    bpmn_acc = pivot["bpmn"].values
    logs_acc = pivot["logs"].values

    # t-test
    t_stat, p_val = ttest_rel(bpmn_acc, logs_acc)
    dfree = len(bpmn_acc) - 1

    print("=== Paired t-test BPMN vs Logs ===")
    print(f"t({dfree}) = {t_stat:.2f}, p = {p_val:.4e}")
    return t_stat, p_val, pivot

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def summarize_by_prefix_bins(full_df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """
    Агрегує статистику по діапазонах довжин префіксів для Logs-only та BPMN-based
    та рахує парний t-тест для кожного діапазону.
    Повертає значення у форматі 'mean ± std' і додає t(12) та p-value.

    Parameters
    ----------
    full_df : pd.DataFrame
        Результати після aggregate_prefix_statistics з колонками:
        ['prefix_len', '{metric}_mean', '{metric}_std', 'data_type', 'architecture', ...]
    metric : str
        Назва метрики (без '_mean' чи '_std').

    Returns
    -------
    summary_df : pd.DataFrame
        Таблиця формату:
        | Префікс (довжина) | Logs-only | BPMN-based | t(12) | p-value |
    """

    # визначаємо бінування довжин
    bins = [0, 10, 50, 100, 250, np.inf]
    labels = ["0–9", "10–49", "50–99", "100–250", ">250"]

    # додаємо категорію
    full_df = full_df.copy()
    full_df["prefix_bin"] = pd.cut(full_df["prefix_len"], bins=bins, labels=labels, right=False)

    # групування для mean і std
    grouped = (
        full_df.groupby(["prefix_bin", "data_type"])
        .agg({f"{metric}_mean": "mean", f"{metric}_std": "mean"})
        .reset_index()
    )

    # об'єднуємо mean і std у формат рядка
    grouped["value"] = grouped.apply(
        lambda x: f"{x[f'{metric}_mean']:.3f} ± {x[f'{metric}_std']:.3f}", axis=1
    )

    # робимо широку таблицю (Logs vs BPMN)
    summary = grouped.pivot(index="prefix_bin", columns="data_type", values="value").reset_index()
    summary = summary.rename(columns={
        "prefix_bin": "Префікс (довжина)",
        "logs": "Logs-only",
        "bpmn": "BPMN-based"
    })

    # обчислюємо t-тести по кожному біну
    results = []
    for label in labels:
        bin_df = full_df[full_df["prefix_bin"] == label]
        logs_vals = (
            bin_df[bin_df["data_type"] == "logs"]
            .groupby("architecture")[f"{metric}_mean"].mean()
        )
        bpmn_vals = (
            bin_df[bin_df["data_type"] == "bpmn"]
            .groupby("architecture")[f"{metric}_mean"].mean()
        )

        common_archs = logs_vals.index.intersection(bpmn_vals.index)
        if len(common_archs) > 1:  # щоби було що порівнювати
            t_stat, p_val = ttest_rel(bpmn_vals.loc[common_archs], logs_vals.loc[common_archs])
            results.append((f"t({len(common_archs)-1}) = {t_stat:.2f}", f"{p_val:.2e}"))
        else:
            results.append(("N/A", "N/A"))

    summary["t-test"] = [r[0] for r in results]
    summary["p-value"] = [r[1] for r in results]

    return summary


def summarize_by_prefix_bins2(full_df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """
    Агрегує статистику по діапазонах довжин префіксів для Logs-only та BPMN-based.
    Повертає значення у форматі 'mean ± std'.

    Parameters
    ----------
    full_df : pd.DataFrame
        Результати після aggregate_prefix_statistics з колонками:
        ['prefix_len', '{metric}_mean', '{metric}_std', 'data_type', ...]
    metric : str
        Назва метрики (без '_mean' чи '_std').

    Returns
    -------
    summary_df : pd.DataFrame
        Таблиця формату:
        | Префікс (довжина) | Logs-only | BPMN-based |
    """

    # визначаємо бінування довжин
    bins = [0, 10, 50, 100, 250, np.inf]
    labels = ["0–9", "10–49", "50–99", "100–250", ">250"]

    # додаємо категорію
    full_df["prefix_bin"] = pd.cut(full_df["prefix_len"], bins=bins, labels=labels, right=False)

    # групування для mean і std
    grouped = (
        full_df.groupby(["prefix_bin", "data_type"])
        .agg({f"{metric}_mean": "mean", f"{metric}_std": "mean"})
        .reset_index()
    )

    # об'єднуємо mean і std у формат рядка
    grouped["value"] = grouped.apply(
        lambda x: f"{x[f'{metric}_mean']:.3f} ± {x[f'{metric}_std']:.3f}", axis=1
    )

    # розкладаємо по колонках Logs-only / BPMN-based
    summary = grouped.pivot(index="prefix_bin", columns="data_type", values="value").reset_index()

    # перейменування
    summary = summary.rename(columns={
        "prefix_bin": "Префікс (довжина)",
        "logs": "Logs-only",
        "bpmn": "BPMN-based"
    })

    return summary

def aggregate_prefix_statistics(directory_path: str) -> pd.DataFrame:
    """
    Агрегація статистики по всім архітектурам / data_type / seed.
    Кожен файл statistics.xlsx містить розподіл по prefix_len.
    Результат — один DataFrame зі спільною структурою.

    :param directory_path: шлях до каталогу зі статистикою
    :return: агрегований DataFrame
    """
    all_results = []

    for filename in os.listdir(directory_path):
        if filename.endswith("statistics.xlsx"):
            full_path = os.path.join(directory_path, filename)

            match = re.match(r"(.+?)(?:_pr)?_(logs|bpmn)_seed(\d+)_statistics\.xlsx", filename)
            if not match:
                print(f"[WARN] Пропущено (не підпав під шаблон): {filename}")
                continue

            architecture, data_type, seed = match.groups()

            try:
                df = pd.read_excel(full_path)

                if df.empty:
                    continue

                print(f"[OK] Обробка: {filename} — {len(df)} рядків")

                # додамо архітектуру, тип даних і сид для кожного рядка
                df["architecture"] = architecture
                df["data_type"] = data_type
                df["seed"] = int(seed)

                all_results.append(df)

            except Exception as e:
                print(f"[ERROR] при читанні {filename}: {e}")

    if not all_results:
        print("[WARN] Не знайдено жодних даних")
        return pd.DataFrame()

    # Об’єднати все в один DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df



def summarize_prefix_statistics(final_df: pd.DataFrame, filters: dict = None) -> pd.DataFrame:
    """
    Узагальнює статистику по довжині префіксів.
    - Застосовує фільтри (architecture, data_type, seed), якщо передані.
    - Групує ТІЛЬКИ за prefix_len, об’єднуючи всі архітектури/сид-и/типи даних.
    - Для кожної метрики *_mean бере:
        mean = середнє по seed/архітектурах
        std  = std від *_mean по seed/архітектурах
        min  = min від *_mean
        max  = max від *_mean
    - count → сумарний
    """

    if final_df is None or final_df.empty:
        print("[WARN] summarize_prefix_statistics: порожній DataFrame")
        return final_df

    df = final_df.copy()

    # === Фільтрація ===
    if filters:
        for key, values in filters.items():
            if key in df.columns and values:
                df = df[df[key].isin(values)]
    if df.empty:
        print("[WARN] summarize_prefix_statistics: порожньо після фільтрів")
        return df

    if "prefix_len" not in df.columns:
        raise KeyError("Очікувана колонка 'prefix_len' відсутня")

    # базові метрики як префікси колонок *_mean
    base_metrics = sorted({c[:-5] for c in df.columns if c.endswith("_mean") and c not in ("seed_mean",)})
    if not base_metrics:
        print("[WARN] summarize_prefix_statistics: немає колонок '*_mean'")
        return df

    group_cols = ["prefix_len"]

    def _agg_one_group(g: pd.DataFrame) -> pd.Series:
        out = {"prefix_len": g["prefix_len"].iloc[0], "count": int(np.nansum(g["count"].values))}
        for m in base_metrics:
            vals = g[f"{m}_mean"].values.astype(float)
            out[f"{m}_mean"] = float(np.nanmean(vals))
            out[f"{m}_std"]  = float(np.nanstd(vals))
            out[f"{m}_min"]  = float(np.nanmin(vals))
            out[f"{m}_max"]  = float(np.nanmax(vals))
        return pd.Series(out)

    agg_df = (
        df.groupby(group_cols, as_index=False)
          .apply(_agg_one_group)
          .reset_index(drop=True)
          .sort_values("prefix_len")
    )

    return agg_df
def load_and_aggregate_confusion_matrices(
    folder_path,
    data_type_filter="bpmn",
    reduction="avg",  # або 'sum'
    normalize=True
):
    """
    Зчитує всі .xlsx confusion матриці з вказаної папки, фільтрує за типом (logs/bpmn),
    і повертає одну зведену матрицю.

    Параметри:
    - folder_path: шлях до папки з .xlsx файлами матриць
    - data_type_filter: 'logs' або 'bpmn'
    - reduction: 'avg' або 'sum'
    - normalize: чи нормалізувати по рядках (True/False)

    Повертає:
    - aggregated_df: фінальна матриця pandas.DataFrame
    """
    matrices = []

    for filename in os.listdir(folder_path):
        if filename.endswith("CM.png_full.xlsx") and data_type_filter in filename:
            full_path = os.path.join(folder_path, filename)
            try:
                print(f' Aggregate CMs: {filename}')
                df = pd.read_excel(full_path, index_col=0)
                matrices.append(df)
            except Exception as e:
                print(f"❌ Помилка при зчитуванні {filename}: {e}")

    if not matrices:
        print("⚠️ Немає матриць для агрегації.")
        return None

    # Вирівняти порядок індексів/стовпців
    all_labels = sorted(set().union(*[m.index for m in matrices]))
    matrices = [m.reindex(index=all_labels, columns=all_labels, fill_value=0) for m in matrices]

    # Агрегація
    if reduction == "sum":
        aggregated_df = sum(matrices)
    else:
        aggregated_df = sum(matrices) / len(matrices)

    # Нормалізація по рядках
    if normalize:
        aggregated_df = aggregated_df.div(aggregated_df.sum(axis=1), axis=0).fillna(0)

    return aggregated_df

def combine_activity_stat_files_clear(directory):
    all_dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("_accuracy_stat.xlsx"):
            print(f' Aggregate distributions: {filename}')
            df = pd.read_excel(os.path.join(directory, filename))
            all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def combine_activity_stat_files(directory):
    all_dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("_accuracy_stat.xlsx"):
            print(f' Aggregate distributions: {filename}')
            df = pd.read_excel(os.path.join(directory, filename))
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 🔧 Множимо train_count на 2 тільки для logs
    if "mode" in combined_df.columns and "train_count" in combined_df.columns:
        combined_df.loc[combined_df["mode"] == "logs", "train_count"] *= 2

    return combined_df


def save_activity_stats_to_excel(activity_stats, architecture, mode, seed, file_path):
    """
    Зберігає словник activity_train_vs_val_accuracy у Excel із додатковими колонками.

    Parameters:
    - activity_stats: dict (node_id -> {"train_count": ..., "val_accuracy": ...})
    - architecture: назва архітектури (наприклад: "GAT")
    - mode: режим (наприклад: "bpmn" або "logs")
    - file_path: шлях до Excel файлу (наприклад: "activity_distributions.xlsx")
    """
    df = pd.DataFrame.from_dict(activity_stats, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "node_id"}, inplace=True)
    df["architecture"] = architecture
    df["mode"] = mode
    df["seed"] = seed
    df.to_excel(file_path, index=False)
def aggregate_metric_over_epochs(directory_path, metric_name):
    """
    Збирає задану метрику по епохах для кожної архітектури, seed і типу даних (logs/bpmn).

    Повертає DataFrame у форматі:
    | architecture | data_type | seed | epoch | metric_value |
    """
    all_records = []

    for filename in os.listdir(directory_path):
        if filename.endswith("statistics.xlsx"):
            full_path = os.path.join(directory_path, filename)

            match = re.match(r"(.+?)(?:_pr)?_(logs|bpmn)_seed(\d+)_statistics\.xlsx", filename)
            if match:
                architecture, data_type, seed = match.groups()

                try:
                    df = pd.read_excel(full_path)

                    if not df.empty and metric_name in df.columns:
                        for idx, row in df.iterrows():
                            record = {
                                "architecture": architecture,
                                "data_type": data_type,
                                "seed": int(seed),
                                "epoch": int(row['epochs']),
                                "metric_value": row[metric_name]
                            }
                            all_records.append(record)
                except Exception as e:
                    print(f"Помилка при обробці {filename}: {e}")

    return pd.DataFrame(all_records)

def save_aggregated_statistics(df, output_path):
    """
    Зберігає агреговану таблицю у форматі Excel або CSV.

    :param df: DataFrame з агрегованими результатами
    :param output_path: Шлях для збереження файлу (з розширенням .xlsx або .csv)
    """
    if output_path.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
        print(f"Агрегована статистика збережена у {output_path}")
    elif output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
        print(f"Агрегована статистика збережена у {output_path}")
    else:
        raise ValueError("Файл має закінчуватись на '.xlsx' або '.csv'")


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
            print("Optimizer завантажено")
        if stats:
            stats = checkpoint.get('stats')
            print(f"Статиситку завантажено")

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

def save_prepared_data(data, input_dim, doc_dim, global_node_dict, file_path):
    """
    Зберігає підготовлені дані у файл.
    :param data_list: Список підготовлених об'єктів Data.
    :param input_dim: Вхідний розмір для моделі.
    :param file_path: Шлях для збереження файлу.
    """
    torch.save({"data": data, "input_dim": input_dim, "doc_dim": doc_dim, "global_node_dict": global_node_dict}, file_path)
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
        return checkpoint["data"], checkpoint["input_dim"], checkpoint["doc_dim"], checkpoint["global_node_dict"]
    except FileNotFoundError:
        print(f"Файл з підготовленими даними не знайдено: {file_path}")
        return None, None, None, None

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


def load_global_statistics_from_json(file_path):
    """
    Завантажує глобальну статистику з JSON-файлу.

    :param file_path: Шлях до файлу (без .json)
    :return: Словник зі статистикою
    """
    with open(f"{file_path}.json", "r") as f:
        stats = json.load(f)
    return stats

def save2csv(df: pd.DataFrame, file_name: str):
    """Зберігає реєстр."""
    #print(df)
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, dict) and all(isinstance(v, list) for v in df.values()):
            # Вирівнювання списків за довжиною
            max_length = max(len(values) for values in df.values())
            for key, values in df.items():
                if len(values) < max_length:
                    df[key] = values + [None] * (max_length - len(values))
            df = pd.DataFrame(df)
        else:
            logger.error("Непідтримуваний формат даних для збереження.")
            return

    try:
        reg_data_path = f"{file_name}.xlsx"
        df.to_excel(reg_data_path, index=False)
        logger.info(f"Дані збережено у {reg_data_path}")
    except Exception as e:
        logger.error(f"Помилка при збереженні даних: {e}")


def save_confusion_matrix_to_csv(cm, class_labels, file_path):
    """
    Зберігає confusion matrix у CSV/XLSX з підписами класів по осях.

    :param cm: numpy.ndarray, матриця плутанини.
    :param class_labels: список ідентифікаторів класів (вузлів).
    :param file_path: шлях до файлу без розширення (буде .xlsx).
    """
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    df_cm.index.name = 'True Label'
    df_cm.columns.name = 'Predicted Label'
    df_cm.to_excel(f"{file_path}.xlsx")