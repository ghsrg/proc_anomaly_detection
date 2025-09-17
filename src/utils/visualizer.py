import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import linregress
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

from src.utils.logger import get_logger
from src.utils.graph_utils import clean_graph
from src.utils.file_utils import save2csv, save_confusion_matrix_to_csv
import pandas as pd
from graphviz import Digraph
import inspect
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm, TwoSlopeNorm

logger = get_logger(__name__)


def save_training_diagram(stats, file_path, test_stats=None, title='Training and Validation Metrics'):
    """
    Зберігає графік навчання та валідації, додаючи підписи значень для обраних епох.

    :param stats: Статистика навчання (словник зі списками по епохах).
    :param file_path: Шлях для збереження графіка.
    :param test_stats: Метрики тестових даних (словник).
    """
    fig, ax1 = plt.subplots(figsize=(19.2, 10.8))

    # Перша шкала для метрик (Precision, Recall, F1, ROC AUC)
    metrics = {
        'Precision': ('val_precision', 'blue', 1),
        'Recall': ('val_recall', 'green', 1),
        'ROC AUC': ('val_roc_auc', 'red', 2),
        'F1-score': ('val_f1_score', 'grey', 2),
        'AUPRC': ('val_auprc', 'peru', 2.5),
        'ADR': ('val_adr', 'lime', 0.5),
        'FAR': ('val_far', 'plum', 0.5),
        'FPR': ('val_fpr', 'pink', 0.5),
        'FNR': ('val_fnr', 'cyan', 0.5),
        'ACC': ('val_accuracy', 'orange',  2.5),
        'TOP-k-ACC': ('val_top_k_accuracy', 'purple',  1.5),
        'MAE': ('val_mae', 'purple', 0.5),
        'RMSE': ('val_rmse', 'red', 2.5),
        'R2': ('val_r2', 'brown', 1),
        'Out_of_Scope': ('val_out_of_scope_rate', 'peru', 1)

    }

    num_epochs = len(stats['epochs'])
    step = max(1, num_epochs // 6)  # Крок для підписів

    for label, (key, color, lw) in metrics.items():
        if key not in stats:
            #print(f'passed {key}')
            continue  # Пропускаємо метрику, якої немає в статистиці
        ax1.plot(stats['epochs'], stats[key], label=label, linestyle='-', color=color, linewidth=lw)

        # Додавання підписів для обраних епох
        for i, (epoch, value) in enumerate(zip(stats['epochs'], stats[key])):
            if i % step == 0 or i == num_epochs - 1:
                #print(value, color, key)
                ax1.text(epoch, value, f"{value:.4f}", color=color, fontsize=8, ha='right', va='bottom')

    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Metrics')
    ax1.legend(loc='upper left')

    # Друга шкала для втрат (Loss)
    ax2 = ax1.twinx()
    ax2.plot(stats['epochs'], stats['train_loss'], label='Train Loss', linestyle='-', color='black', linewidth=2)

    # Додавання підписів для втрат
    for i, (epoch, value) in enumerate(zip(stats['epochs'], stats['train_loss'])):
        if i % step == 0 or i == num_epochs - 1:
            ax2.text(epoch, value, f"{value:.4f}", color='orange', fontsize=8, ha='right', va='bottom')

    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')

    # Тестові метрики (якщо є)
    if test_stats:
        test_metrics = {
            'Test Precision': ('precision', 'blue'),
            'Test Recall': ('recall', 'green'),
            'Test ROC AUC': ('roc_auc', 'red'),
            'Test F1-score': ('f1_score', 'grey')
        }
        for label, (key, color) in test_metrics.items():
            if test_stats.get(key) is not None:
                value = test_stats[key]
                ax1.axhline(y=value, color=color, linestyle='--', label=label)
                # Додавання підпису для тестових метрик
                ax1.text(num_epochs, value, f"{value:.4f}", color=color, fontsize=8, ha='right', va='bottom')

    # Оформлення графіка
    plt.title(title, fontsize=16, fontweight='bold', loc='center')

    fig.tight_layout()
    plt.savefig(file_path, dpi=100)
    plt.close()


def plot_prefix_metric_lines(
    df: pd.DataFrame,
    model_type: str,
    pr_mode: str,
    base_path: str
):
    """
    Побудова графіків метрик залежно від довжини префікса:
    - Line plot with error bands (mean ± std) з обмеженням до [0,1]
    - Додаткові лінії min/max (пунктиром)
    - Bar plot: розподіл кількості прикладів по prefix_len з нормальним масштабом
    - Top-1/3/5 та confidence — аналогічно
    """

    def plot_with_std_and_minmax(metric: str, ylabel: str = None):
        if f"{metric}_mean" not in df.columns:
            return

        plt.figure(figsize=(10, 4))
        x = df["prefix_len"]
        mean = df[f"{metric}_mean"]
        std = df[f"{metric}_std"]
        lower = np.clip(mean - std, 0, 1)
        upper = np.clip(mean + std, 0, 1)
        min_vals = df.get(f"{metric}_min", None)
        max_vals = df.get(f"{metric}_max", None)

        plt.plot(x, mean, label="mean")
        plt.fill_between(x, lower, upper, alpha=0.2, label="± std")

        if min_vals is not None and max_vals is not None:
            plt.plot(x, min_vals, linestyle="dashed", linewidth=1, color="gray", label="min")
            plt.plot(x, max_vals, linestyle="dashed", linewidth=1, color="gray", label="max")

        plt.title(f"{model_type} ({pr_mode}) — {metric} mean ± std")
        plt.xlabel("prefix length")
        plt.ylabel(ylabel if ylabel else metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_type}_{pr_mode}_{metric}_lineband.png")
        plt.close()

    # === 1. Line plot with ±std + min/max ===
    for metric in ["accuracy", "f1_macro", "conf", "top1", "top3", "top5"]:
        plot_with_std_and_minmax(metric, ylabel="score")

    # === 2. Bar plot — кількість прикладів для кожного prefix_len ===
    if "count" in df.columns:
        plt.figure(figsize=(10, 4))
        ax = sns.barplot(data=df, x="prefix_len", y="count", color="skyblue")
        plt.title(f"{model_type} ({pr_mode}) — Кількість прикладів по довжині префікса")
        plt.xlabel("prefix length")
        plt.ylabel("count")
        plt.grid(True, axis='y', alpha=0.3)
        # нормалізувати підписи осі Х (тільки частину)
        if len(df["prefix_len"]) > 30:
            step = max(1, len(df["prefix_len"]) // 30)
            for idx, label in enumerate(ax.get_xticklabels()):
                if idx % step != 0:
                    label.set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_type}_{pr_mode}_prefix_count_bar.png")
        plt.close()


def plot_prefix_count_bar(df_counts, model_type: str, pr_mode: str, base_path: str):
    """
    Стовпчиковий графік кількості кейсів по довжині префікса.
    Очікує DataFrame з колонками: prefix_len, count
    """
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_counts, x="prefix_len", y="count", color="steelblue")
    plt.title(f"{model_type} ({pr_mode}) — samples per prefix length")
    plt.xlabel("prefix length L")
    plt.ylabel("#samples")
    plt.grid(True, axis="y", alpha=0.3)
    out_path = f"{base_path}/{model_type}_{pr_mode}_prefix_counts.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def plot_prefix_metric(
    df: pd.DataFrame,
    metric: str,
    model_type: str,
    pr_mode: str,
    path: str,
    kind: str = "line",
    ylabel: str = None
):
    """
    Побудова графіка однієї метрики залежно від довжини префікса.

    kind:
      - "line" → Line plot with error bands (mean ± std) + min/max
      - "bar" → Bar plot (наприклад, для 'count')
    """

    if kind == "line":
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"

        if mean_col not in df.columns:
            print(f"[WARN] {mean_col} не знайдено у DataFrame")
            return

        # Сортуємо дані за prefix_len
        df = df.sort_values("prefix_len")

        plt.figure(figsize=(12, 8))
        x = df["prefix_len"]
        mean = df[mean_col]
        std = df[std_col] if std_col in df.columns else 0
        lower = np.clip(mean - std, 0, 1)
        upper = np.clip(mean + std, 0, 1)

        # основна лінія mean
        plt.plot(x, mean, label="mean", color="blue")
        plt.fill_between(x, lower, upper, alpha=0.2, color="skyblue", label="± std")

        # min / max (пунктиром)
        if min_col in df.columns and max_col in df.columns:
            plt.plot(x, df[min_col], linestyle="dashed", linewidth=0.5, color="gray", label="min", alpha=0.6)
            plt.plot(x, df[max_col], linestyle="dashed", linewidth=0.5, color="black", label="max", alpha=0.6)

        plt.title(f"{model_type} ({pr_mode}) — {metric} mean ± std")
        plt.xlabel("prefix length")
        plt.ylabel(ylabel if ylabel else metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    elif kind == "box":
        value_cols = [c for c in df.columns if c.startswith(metric)]
        if not value_cols:
            print(f"[WARN] {metric} не знайдено у DataFrame для boxplot")
            return

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x="prefix_len", y=f"{metric}_mean", color="skyblue")

        plt.title(f"{model_type} ({pr_mode}) — Boxplot {metric} по довжині префікса")
        plt.xlabel("prefix length")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


    else:
        raise ValueError("kind повинен бути 'line' або 'box'")


def save_training_diagram_old(stats, file_path, test_stats=None):
    """
    Зберігає графік навчання та валідації, додаючи підписи значень для обраних епох.

    :param stats: Статистика навчання (словник зі списками по епохах).
    :param file_path: Шлях для збереження графіка.
    :param test_stats: Метрики тестових даних (словник).
    """
    plt.figure(figsize=(19.2, 10.8))

    # Графіки навчання та валідації
    metrics = {
        'Train Loss': ('train_loss', 'orange'),
        'Precision': ('val_precision', 'blue'),
        'Recall': ('val_recall', 'green'),
        'ROC AUC': ('val_roc_auc', 'red'),
        'F1-score': ('val_f1_score', 'grey')
    }

    num_epochs = len(stats['epochs'])
    step = max(1, num_epochs // 6)  # Крок для підписів

    for label, (key, color) in metrics.items():
        plt.plot(stats['epochs'], stats[key], label=label, linestyle='-', color=color)

        # Додавання підписів лише для певних епох
        for i, (epoch, value) in enumerate(zip(stats['epochs'], stats[key])):
            if i % step == 0 or i == num_epochs - 1:  # Додаємо підписи кожен step або для останньої епохи
                plt.text(epoch, value, f"{value:.4f}", color=color, fontsize=8, ha='right', va='bottom')

    # Тестові метрики (додаються, якщо значення не None)
    if test_stats:
        test_metrics = {
            'Test Precision': ('precision', 'blue'),
            'Test Recall': ('recall', 'green'),
            'Test ROC AUC': ('roc_auc', 'red'),
            'Test F1-score': ('f1_score', 'grey')
        }
        for label, (key, color) in test_metrics.items():
            if test_stats.get(key) is not None:
                value = test_stats[key]
                plt.axhline(y=value, color=color, linestyle='--', label=label)
                # Додавання підпису для тестових метрик
                plt.text(num_epochs, value, f"{value:.4f}", color=color, fontsize=8, ha='right', va='bottom')

    # Обмеження шкали Y
    plt.ylim(-0.05, 1.05)

    # Оформлення графіка
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(file_path, dpi=100)
    plt.close()

def visualize_distribution(distribution_data, file_path):
    """
    Візуалізує розподіл мінімальних, максимальних і середніх значень для тренувальних втрат.

    :param distribution_data: Словник із даними розподілу (мінімум, максимум, середнє).
    :param file_path: Шлях для збереження графіка.
    """
    plt.figure(figsize=(12, 6))

    for label, values in distribution_data.items():
        plt.plot(values, label=label)

    plt.xlabel('Fold')
    plt.ylabel('Values')
    plt.title('Distribution of Min, Max, and Mean Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path, dpi=100)
    plt.close()

def plot_confusion_matrix(true_labels, predicted_labels, class_labels, file_path=None, normalize=False):
    """
    Візуалізує Confusion Matrix.

    :param true_labels: Справжні мітки (list або numpy array).
    :param predicted_labels: Передбачені мітки (list або numpy array).
    :param class_labels: Мітки класів (list).
    :param file_path: Шлях для збереження графіка (або None для показу).
    :param normalize: Нормалізація значень (True/False).
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
    else:
        plt.show()

def visualize_graph_with_dot(graph, file_path=None):
    """
    Візуалізація графа NetworkX у стилі BPMN із використанням Graphviz 'dot'.
    У вузлах відображаємо BPMN name або task_subject, фарбуємо вузли за типом.
    На ребрах показуємо conditionExpression або name (якщо є).
    Якщо SEQUENCE_COUNTER_ заповнено, вузол матиме чорну обводку.
     :param graph: Граф NetworkX для візуалізації.
    :param file_path: Шлях до файлу для збереження (якщо None, виводить на екран).
    """

    # Використовуємо layout від Graphviz з алгоритмом 'dot'
    #graph = clean_graph(graph)
    #pos = nx.nx_agraph.graphviz_layout(graph, prog='dot', args='-Grankdir=LR -Goverlap=false -Gnodesep=0.5 -Goutputorder=edgesfirst -Gsize=none -Granksep=1.5')
    #pos = nx.nx_agraph.graphviz_layout(graph, prog='neato', args='-Goverlap=false')
    pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')

    plt.figure(figsize=(180, 180))

    node_labels = {}
    fill_colors = []
    border_colors = []

    for node, data in graph.nodes(data=True):
        # Отримуємо ідентифікатор/назву
        #logger.debug(node, variable_name="node", max_lines=30)
        #logger.debug(data, variable_name="date", max_lines=30)
        #logger.debug(data.get('type', ''), variable_name="node_type", max_lines=3)
        bpmn_name = data.get('name', node)
        node_type = data.get('type', '')
        if isinstance(node_type, pd.Series):
            node_type = node_type.iloc[0]  # Візьмемо перше значення
        node_type = node_type.lower() if isinstance(node_type, str) else ''

        # Формуємо підпис
        #label_text = bpmn_name
        words = bpmn_name.split()
        label_text = '\n'.join([' '.join(words[i:i + 2]) for i in range(0, len(words), 2)])
        #node_labels[node] = f"{label_text}#_{node}"
        node_labels[node] = f"{label_text}"
        #logger.debug(node_type, variable_name="node_type", max_lines=3)
        # Фарбуємо "заливку" вузла
        if 'starteventsp' in node_type:
            fill_colors.append('lightgreen')
        elif 'startevent' in node_type:
            fill_colors.append('green')
        elif 'endeventsp' in node_type:
            fill_colors.append('#f9cfcf')
        elif 'endevent' in node_type:
            fill_colors.append('red')
        elif 'gateway' in node_type:
            fill_colors.append('yellow')
        elif node_type in ['subprocess', 'callactivity']:
            fill_colors.append('cornflowerblue')
        elif node_type in ['usertask']:
            fill_colors.append('blue')
        elif node_type in ['scripttask','servicetask']:
            fill_colors.append('#f2e6fb')
        elif node_type in ['intermediatethrowevent']:
            fill_colors.append('#27c4c4')
        elif node_type in ['boundaryevent']:
            fill_colors.append('#e7d0f7')
        else:
            #if node_type:
                #logger.debug(node_type, variable_name="node_type", max_lines=3)
            fill_colors.append('lightblue')

        # Якщо SEQUENCE_COUNTER_ існує й не порожній, обводка чорна, інакше "none"
        seq_counter = data.get('SEQUENCE_COUNTER_')
        if seq_counter is not None and seq_counter != '':
            border_colors.append('black')
        else:
            border_colors.append('none')

    # Підписи для ребер
    edge_labels = {}
    for u, v, edge_data in graph.edges(data=True):
        cond_expr = edge_data.get('DURATION_', '')
        taskaction = edge_data.get('taskaction_code', '')
        flow_name = edge_data.get('name', '')
        #if cond_expr:
        edge_labels[(u, v)] = f'{cond_expr} \n {taskaction} \n {flow_name}'
        #elif flow_name:
       #     edge_labels[(u, v)] = flow_name
        #else:
       #     edge_labels[(u, v)] = ''

    # Малюємо вузли
    # node_color = fill_colors — це "заливка"
    # edgecolors = border_colors — це колір обводки
    nx.draw(
        graph,
        pos,
        labels=node_labels,
        node_color=fill_colors,
        edgecolors=border_colors,
        with_labels=True,
        node_size=10000,
        font_size=8,
        edge_color='gray',
        arrows=True,
        arrowsize=40,
        width=2,
        linewidths=3  # Товщина бордера вузлів
    )

    # Підписи на ребрах
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("BPMN Graph using Graphviz 'dot' (Sequence Counter Border)", fontsize=14)
    plt.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Граф збережено у {file_path}")
    else:
        plt.show()

def visualize_graph_heatmap(graph: nx.DiGraph, cm_matrix: np.ndarray, class_labels: list[str], normalize: bool = True, figsize=(12, 8)):
    """
    Візуалізує граф із підсвічуванням вузлів відповідно до частоти помилок у матриці плутанини.

    :param graph: граф типу networkx.DiGraph
    :param cm_matrix: матриця плутанини numpy.ndarray
    :param class_labels: список назв вузлів у тому ж порядку, що й у cm_matrix
    :param normalize: нормалізувати чи абсолютні значення
    :param figsize: розмір фігури
    """
    node_errors = defaultdict(int)
    for i in range(len(cm_matrix)):
        node_name = class_labels[i]
        errors = sum(cm_matrix[i]) - cm_matrix[i, i]  # всі передбачення, крім вірних
        node_errors[node_name] = errors

    # нормалізація
    if normalize and node_errors:
        max_error = max(node_errors.values()) or 1
        node_errors = {k: v / max_error for k, v in node_errors.items()}

    # кольорова карта
    cmap = cm.Reds
    node_colors = []
    for node in graph.nodes():
        label = str(node)
        error_score = node_errors.get(label, 0.0)
        color = cmap(error_score)
        node_colors.append(color)

    pos = nx.spring_layout(graph)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, arrows=True, edge_color="gray")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, label="Error Level")
    plt.title("Graph Node Error Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def visualize_train_vs_val_accuracy(activity_train_vs_val_accuracy, file_path=None):
    """
    Візуалізує залежність точності передбачення активностей від їх частоти у тренувальних даних.

    :param activity_train_vs_val_accuracy: Словник формату {node_idx: {"train_count": int, "val_accuracy": float or None}}
    :param file_path: Шлях до файлу для збереження графіка (опціонально).
    """
    train_counts = []
    val_accuracies = []

    for node_idx, stats in activity_train_vs_val_accuracy.items():
        train_count = stats["train_count"]
        val_accuracy = stats["val_accuracy"]

        if val_accuracy is not None:
            train_counts.append(train_count)
            val_accuracies.append(val_accuracy)

    plt.figure(figsize=(18, 8))
    plt.scatter(train_counts, val_accuracies, alpha=0.7, edgecolors='k')
    plt.xlabel('Train Appearance Count')
    plt.ylabel('Validation Accuracy per Activity')
    plt.title('Training Frequency vs Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()


def visualize_activity_train_vs_val_accuracy_with_regression(activity_train_vs_val_accuracy, file_path=None):
    train_counts = []
    val_accuracies = []

    for node_idx, stats in activity_train_vs_val_accuracy.items():
        train_count = stats["train_count"]
        val_acc = stats["val_accuracy"]
        if val_acc is not None:
            train_counts.append(train_count)
            val_accuracies.append(val_acc)

    plt.figure(figsize=(22, 10))
    plt.scatter(train_counts, val_accuracies, alpha=0.7, label="Activities")

    # Побудова лінії регресії
    if len(train_counts) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(train_counts, val_accuracies)
        x_vals = np.array([min(train_counts), max(train_counts)])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f"Regression line\n$R^2$={r_value**2:.2f}")

    plt.xlabel('Train Appearance Count')
    plt.ylabel('Validation Accuracy per Activity')
    plt.title('Training Frequency vs Validation Accuracy with Regression Line')
    plt.legend()
    plt.grid(True)
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()

def plot_regression_by_architecture(
    df,
    chart_title="Regression by Architecture",
    data_type_filter="bpmn",
    arhitec_filter=None,
    seed_filter=None,
    group_seed=False,
    group_arhitec=False,
    figsize=(10, 10),
    poly_level=2,
    ylim=(0, 1),
    xlim=None,
    file_path=None
):
    """
    Побудова scatter + поліноміальної регресії по архітектурах для заданого режиму.

    Параметри:
    - df: DataFrame з колонками: train_count, val_accuracy, architecture, mode, seed
    - data_type_filter: 'bpmn' або 'logs'
    - arhitec_filter: список архітектур (опціонально)
    - seed_filter: фільтр по seed (опціонально)
    - group_seed: якщо True, середнє по seed
    - group_arhitec: якщо True, середнє по architecture
    - poly_level: порядок полінома (1 = лінійна, 2 = квадратична і т.д.)
    - ylim: межі по Y
    - file_path: шлях до збереження графіка
    """

    filtered_df = df[df["mode"] == data_type_filter]
    if seed_filter is not None:
        filtered_df = filtered_df[filtered_df["seed"] == seed_filter]
    if arhitec_filter:
        filtered_df = filtered_df[filtered_df["architecture"].isin(arhitec_filter)]

    # Групування по seed (усереднення)
    if group_seed:
        filtered_df = (
            filtered_df.groupby(["architecture", "train_count"])
            .agg({"val_accuracy": "mean"})
            .reset_index()
        )
        filtered_df["seed"] = "avg"

    # Групування по архітектурі (усереднення по train_count)
    if group_arhitec:
        filtered_df = (
            filtered_df.groupby(["train_count"])
            .agg({"val_accuracy": "mean"})
            .reset_index()
        )
        filtered_df["architecture"] = "Average"

    plt.figure(figsize=figsize)

    architectures = filtered_df["architecture"].unique()

    for arch in architectures:
        sub_df = filtered_df[filtered_df["architecture"] == arch]
        sub_df = sub_df.dropna(subset=["val_accuracy"])
        x = sub_df["train_count"].values
        y = sub_df["val_accuracy"].values
        if len(x) > poly_level:
            coeffs = np.polyfit(x, y, deg=poly_level)
            poly = np.poly1d(coeffs)
            x_vals = np.linspace(min(x), max(x), 200)
            y_vals = poly(x_vals)
            r2 = r2_score(y, poly(x))
            plt.plot(x_vals, y_vals, linestyle="--", label=f"{arch} (R²={r2:.2f})")
            plt.scatter(x, y, alpha=0.4)

    plt.xlabel("Train Appearance Count", fontsize=20)
    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.title(chart_title, fontsize=24)
    plt.tick_params(axis='both', labelsize=18)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()


def visualize_diff_conf_matrix(
    cm_bpmn,
    cm_logs,
    file_path=None,
    top_k=30,
    top_k_mode="diag",  # "diag" або "diff"
    use_log_scale=False,
    title="Δ Confusion Matrix (BPMN - Logs)",
    min_value=0.1,
    cmap="RdBu",
    normalize=False
):
    """
    Візуалізація порівняльної теплокарти (Logs vs BPMN).
    """
    # Узгодження міток
    all_labels = sorted(set(cm_bpmn.index) | set(cm_logs.index))
    cm_bpmn = cm_bpmn.reindex(index=all_labels, columns=all_labels, fill_value=0)
    cm_logs = cm_logs.reindex(index=all_labels, columns=all_labels, fill_value=0)

    # Віднімання ДО або ПІСЛЯ нормалізації
    if normalize:
        cm_bpmn = cm_bpmn.div(cm_bpmn.sum(axis=1), axis=0).fillna(0)
        cm_logs = cm_logs.div(cm_logs.sum(axis=1), axis=0).fillna(0)

    diff_matrix = cm_bpmn - cm_logs

    # Top-K вибір
    if top_k is not None:
        if top_k_mode == "diff":
            diff_score = np.abs(np.diag(diff_matrix))
        else:  # "diag"
            diff_score = np.diag(cm_bpmn)

        top_k_indices = np.argsort(diff_score)[::-1][:top_k]
        diff_matrix = diff_matrix.iloc[top_k_indices, top_k_indices]
        labels = diff_matrix.index.tolist()
    else:
        labels = diff_matrix.index.tolist()

    # Фільтрація шуму
    filtered_diff = diff_matrix.copy()
    filtered_diff[np.abs(filtered_diff) < min_value] = 0
    mask = filtered_diff == 0

    def format_cell(x):
        if abs(x) < min_value:
            return ""
        elif abs(x) >= 100:
            return f"{x:.0f}"
        elif abs(x) >= 10:
            return f"{x:.1f}".rstrip("0").rstrip(".")
        elif abs(x) >= 1:
            return f"{x:.2f}".rstrip("0").rstrip(".")
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")

    annotations = np.vectorize(format_cell)(filtered_diff)

    plt.figure(figsize=(min(1 + 0.5 * len(labels), 20), min(1 + 0.5 * len(labels), 18)))

    if use_log_scale:
        norm = LogNorm(vmin=np.abs(filtered_diff[filtered_diff != 0]).min(), vmax=np.abs(filtered_diff).max())
    else:
        max_abs = np.max(np.abs(filtered_diff.values))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    sns.heatmap(
        filtered_diff,
        annot=annotations,
        fmt="",
        cmap=cmap,
        mask=mask,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Δ BPMN - Logs"}
    )

    plt.title(title, fontsize=24)
    plt.xlabel("Predicted",fontsize=20)
    plt.ylabel("True",fontsize=20)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=120)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()


def plot_regression_logs_vs_bpmn(
    df,
    chart_title="Regression: Logs vs BPMN",
    arhitec_filter=None,
    seed_filter=None,
    group_seed=False,
    group_arhitec=False,
    figsize=(14, 8),
    poly_level=2,
    ylim=(0, 1),
    xlim=None,
    file_path=None
):
    plt.figure(figsize=figsize)

    for mode in ["logs", "bpmn"]:
        filtered_df = df[df["mode"] == mode]

        if seed_filter is not None:
            filtered_df = filtered_df[filtered_df["seed"] == seed_filter]
        if arhitec_filter:
            filtered_df = filtered_df[filtered_df["architecture"].isin(arhitec_filter)]

        if group_seed:
            filtered_df = (
                filtered_df.groupby(["architecture", "train_count"])
                .agg({"val_accuracy": "mean"})
                .reset_index()
            )
            filtered_df["seed"] = "avg"

        if group_arhitec:
            filtered_df = (
                filtered_df.groupby(["train_count"])
                .agg({"val_accuracy": "mean"})
                .reset_index()
            )
            filtered_df["architecture"] = f"Average ({mode})"
            architectures = [f"Average ({mode})"]
        else:
            architectures = filtered_df["architecture"].unique()

        for arch in architectures:
            sub_df = filtered_df[filtered_df["architecture"] == arch]
            sub_df = sub_df.dropna(subset=["val_accuracy"])
            x = sub_df["train_count"].values
            y = sub_df["val_accuracy"].values
            if len(x) > poly_level:
                coeffs = np.polyfit(x, y, deg=poly_level)
                poly = np.poly1d(coeffs)
                x_vals = np.linspace(min(x), max(x), 200)
                y_vals = poly(x_vals)
                r2 = r2_score(y, poly(x))
                label = f"{arch} (R²={r2:.2f})"
                plt.plot(x_vals, y_vals, linestyle="--", label=label)
                plt.scatter(x, y, alpha=0.4)

    plt.xlabel("Train Appearance Count", fontsize=20)
    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.title(chart_title, fontsize=24)
    plt.tick_params(axis='both', labelsize=18)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=14)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()
def plot_architecture_radar_by_metric(
    df,
    chart_title="Radar Chart by Architecture",
    data_type_filter=None,
    seed_filter=None,  # якщо вказаний — фільтруємо, якщо ні — усереднюємо
    metrics=None,
    metric_labels=None,
    figsize=(8, 8),
    normalize=False,
    file_path=None,
    ylim=(0, 1)
):
    """
    Побудова Radar Chart, де вершини — архітектури, а лінії — метрики.

    Параметри:
    - df: DataFrame зі статистикою по архітектурах
    - chart_title: заголовок діаграми
    - data_type_filter: 'logs', 'bpmn' або None
    - seed_filter: int або None. Якщо None — рахується середнє по seed
    - metrics: список метрик (за замовчуванням: accuracy, top-3, 1-OOS)
    - metric_labels: підписи до метрик
    - figsize: розмір полотна
    - normalize: нормалізувати метрики в [0,1]
    - file_path: шлях до збереження (PNG)
    """
    df = df.copy()
    df["1 - OOS Rate"] = 1 - df["val_out_of_scope_rate"]

    if data_type_filter:
        df = df[df["data_type"] == data_type_filter]

    if seed_filter is not None:
        df = df[df["seed"] == seed_filter]
    else:
        df = df.groupby(["architecture", "data_type"])[
            ["val_accuracy", "val_top_k_accuracy", "1 - OOS Rate"]
        ].mean().reset_index()

    if not metrics:
        metrics = ["val_accuracy", "val_top_k_accuracy", "1 - OOS Rate"]
    if not metric_labels:
        metric_labels = ["Accuracy", "Top-3 Accuracy", "1 - OOS Rate"]

    architectures = df["architecture"].tolist()
    num_vars = len(architectures)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for metric, label in zip(metrics, metric_labels):
        values = df[metric].tolist()
        if normalize and max(values) > 0:
            values = [v / max(values) for v in values]
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_title(chart_title, size=24, y=1.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), architectures)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left' ,fontsize=12)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()

def plot_class_bar_chart(
    df,
    class_dict,
    metric_list,
    metric_labels=None,
    class_labels=None,
    data_type_filter=None,
    chart_title="Порівняння класів архітектур",
    figsize=(8, 6),
    file_path=None
):
    """
    Побудова стовпчастої діаграми з середніми метриками по класах архітектур.

    Параметри:
    - df: DataFrame (включно з колонками: architecture, seed, метрики)
    - class_dict: словник {"назва класу": [архітектури]}
    - metric_list: перелік метрик, які будемо відображати
    - metric_labels: підписи до метрик (опційно)
    - class_labels: список назв класів (опційно)
    - chart_title: заголовок графіка
    - figsize: розміри полотна
    - file_path: шлях до збереження (PNG)
    """
    plot_data = []

    if data_type_filter:
        df = df[df["data_type"] == data_type_filter]

    for cls_name, arch_list in class_dict.items():
        class_subset = df[df["architecture"].isin(arch_list)]
        class_avg = class_subset.groupby("architecture")[metric_list].mean().mean()
        for i, metric in enumerate(metric_list):
            label = metric_labels[i] if metric_labels else metric
            plot_data.append({
                "Клас": cls_name,
                "Метрика": label,
                "Значення": class_avg[metric]
            })

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x="Клас", y="Значення", hue="Метрика")
    plt.title(chart_title, fontsize=24)
    plt.ylabel("Значення метрик", fontsize=20)
    plt.xlabel("", fontsize=20)
    plt.legend(title="", bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()

import seaborn as sns
import matplotlib.pyplot as plt


def plot_metric_over_epochs(
        df,
        chart_title="Metric Over Epochs",
        data_type_filter=None,
        seed_filter=None,
        max_epoch=None,
        ylim=(0, 1),
        figsize=(14, 6),
        loc='upper left',
        file_path=None
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    filtered_df = df.copy()

    if data_type_filter:
        filtered_df = filtered_df[filtered_df["data_type"] == data_type_filter]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з data_type фільтрами.")
            return

    if seed_filter is not None:
        filtered_df = filtered_df[filtered_df["seed"] == seed_filter]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з seed фільтрами.")
            return
    else:
        # Усереднення по seed
        filtered_df = (
            filtered_df.groupby(["architecture", "data_type", "epoch"])
            .agg({"metric_value": "mean"})
            .reset_index()
        )

    if max_epoch is not None:
        filtered_df = filtered_df[filtered_df["epoch"] <= max_epoch]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з epoch фільтрами.")
            return

    # Архітектури
    architectures = filtered_df["architecture"].unique()
    num_arch = len(architectures)

    # Палітра + товщина
    palette = sns.color_palette("husl", n_colors=num_arch) #deep
    linewidths = [0.5 if i % 2 == 0 else 1.25 for i in range(num_arch)]

    # Побудова графіку
    plt.figure(figsize=figsize)
    for i, arch in enumerate(architectures):
        arch_df = filtered_df[filtered_df["architecture"] == arch]
        plt.plot(
            arch_df["epoch"],
            arch_df["metric_value"],
            label=arch,
            color=palette[i],
            linewidth=linewidths[i]
        )

    plt.title(chart_title, fontsize=24)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Metric Value", fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.ylim(ylim)
    plt.grid(True)
    plt.legend(loc=loc, fontsize=16)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()


def plot_metric_over_epochs2(
    df,
    chart_title="Metric Over Epochs",
    data_type_filter=None,
    seed_filter=None,
    max_epoch=None,
    figsize=(14, 6),
    file_path=None
):
    """
    Побудова графіку зміни метрики по епохах для різних GNN-архітектур.

    Параметри:
    - df: pandas DataFrame (з колонками: architecture, data_type, seed, epoch, metric_value)
    - chart_title: заголовок графіку
    - data_type_filter: 'logs' або 'bpmn' або None
    - seed_filter: int або None
    - max_epoch: максимальне значення епохи (обрізання по осі X)
    - figsize: розмір графіку
    """
    # Копія і фільтрація
    filtered_df = df.copy()
    if data_type_filter:
        filtered_df = filtered_df[filtered_df["data_type"] == data_type_filter]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з data_type фільтрами.")
            return
    if seed_filter is not None:
        filtered_df = filtered_df[filtered_df["seed"] == seed_filter]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з seed фільтрами.")
            return
    if max_epoch is not None:
        filtered_df = filtered_df[filtered_df["epoch"] <= max_epoch]
        if filtered_df.empty:
            print("⚠️ Немає даних для побудови графіку з epoch фільтрами.")
            return

    # Побудова графіку
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=filtered_df,
        x="epoch",
        y="metric_value",
        hue="architecture",
        marker="o",
        linewidth=0.75
    )
    plt.title(chart_title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()

def visualize_distribution(node_distribution=None, edge_distribution=None, prefix_distribution=None, file_path=None):
    """
    Візуалізує розподіл кількості вузлів, зв'язків та довжини префіксів у графах.

    :param node_distribution: Розподіл кількості вузлів.
    :param edge_distribution: Розподіл кількості зв'язків.
    :param prefix_distribution: Розподіл довжини префіксів (опціонально).
    :param file_path: Шлях до файлу для збереження графіка (опціонально).
    """
    plt.figure(figsize=(22, 12))
    name = 'Node'

    # Розподіл кількості вузлів
    if node_distribution:
        plt.plot(
            sorted(node_distribution.keys()),
            [node_distribution[k] for k in sorted(node_distribution.keys())],
            label='Node Count Distribution',
            linestyle='-',
            color='blue'
        )

    # Розподіл кількості зв'язків
    if edge_distribution:
        plt.plot(
            sorted(edge_distribution.keys()),
            [edge_distribution[k] for k in sorted(edge_distribution.keys())],
            label='Edge Count Distribution',
            linestyle='--',
            color='red'
        )
        name = f'{name}, Edge'

    # Розподіл довжини префіксів (якщо переданий)
    if prefix_distribution:
        plt.plot(
            sorted(prefix_distribution.keys()),
            [prefix_distribution[k] for k in sorted(prefix_distribution.keys())],
            label='Prefix Length Distribution',
            linestyle='-',
            color='green'
        )
        name = f'{name}, Prefix'
    if node_distribution or edge_distribution or prefix_distribution:
        plt.xlabel('Count')
        plt.ylabel('Number of Graphs')
        plt.title(f'{name} Length Distributions')
        plt.legend()
        plt.grid(True)

        if file_path:
            plt.savefig(file_path)
            plt.close()
            print(f"Графік розподілення збережено у {file_path}")
        else:
            plt.show()
    else:
        print("⚠️ Немає даних для побудови графіку розподілення.")

def plot_avg_epoch_time_bar(
        df,
        chart_title="Середній Час Навчання На Епоху",
        data_type_filter="bpmn",
        seed_filter=None,
        figsize=(18, 10),
        file_path=None
):
    """
    Побудова горизонтальної діаграми середнього часу навчання на епоху.

    Параметри:
    - df: DataFrame, що містить spend_time, epoch, architecture, data_type, seed
    - chart_title: заголовок графіку
    - data_type_filter: 'bpmn' або 'logs'
    - seed_filter: int або None (усереднення по seed)
    - figsize: розмір графіку
    - file_path: шлях до збереження
    """

    filtered_df = df[df["data_type"] == data_type_filter].copy()
    if seed_filter is not None:
        filtered_df = filtered_df[filtered_df["seed"] == seed_filter]

    # Обчислення часу на епоху
    filtered_df["time_per_epoch"] = filtered_df["spend_time"] / filtered_df["epoch"]

    # Усереднення по архітектурах
    avg_time_df = (
        filtered_df.groupby("architecture")["time_per_epoch"]
        .mean()
        .reset_index()
        .sort_values("time_per_epoch", ascending=False)
    )

    # Побудова діаграми з колірною палітрою від червоного до зеленого
    norm = plt.Normalize(avg_time_df["time_per_epoch"].min(), avg_time_df["time_per_epoch"].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    colors = [sm.to_rgba(val) for val in avg_time_df["time_per_epoch"]]

    # Побудова графіка
    plt.figure(figsize=figsize)
    sns.barplot(
        data=avg_time_df,
        y="architecture",
        x="time_per_epoch",
        palette=colors
    )
    plt.title(chart_title, fontsize=24)
    plt.xlabel("Середній час навчання на епоху (сек)", fontsize=20)
    plt.ylabel("Архітектура", fontsize=20)
    plt.tick_params(axis='both', labelsize=18)

    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()



def visualize_confusion_matrix(confusion_matrix_object, class_labels=None, file_path=None, top_k=None,
                                global_node_dict=None):
    """
    Візуалізує матрицю плутанини з опціональним скороченням до top_k найчастіших класів та підписами вузлів.

    :param confusion_matrix_object: dict з true/predicted або готова матриця
    :param class_labels: список міток класів (опціонально)
    :param file_path: шлях до файлу для збереження (або None для показу)
    :param top_k: int або ('best' | 'worst', k)
    :param global_node_dict: словник {node_name: index} для побудови підписів
    """
    if isinstance(confusion_matrix_object, dict):
        true_labels = confusion_matrix_object["true_labels"]
        predicted_labels = confusion_matrix_object["predicted_labels"]
        cm = confusion_matrix(true_labels, predicted_labels)
    else:
        cm = confusion_matrix_object
        true_labels = list(range(cm.shape[0]))
        predicted_labels = list(range(cm.shape[1]))

    # Формування підписів на основі глобального словника
    if class_labels is None:
        if global_node_dict is not None:
            inv_node_dict = {v: k for k, v in global_node_dict.items()}
            class_labels = [inv_node_dict.get(i, f"Unknown_{i}")[:25] for i in range(cm.shape[0])]
        else:
            class_labels = [str(i) for i in range(cm.shape[0])]

    # Збереження повної версії матриці
    if file_path:
        save_confusion_matrix_to_csv(cm, class_labels, file_path + '_full')

    # Top-K скорочення
    if top_k is not None and len(class_labels) == cm.shape[0]:
        row_sums = np.sum(cm, axis=1)
        correct = np.diag(cm)
        mistakes = row_sums - correct

        if isinstance(top_k, tuple):
            mode, k = top_k
            if mode == 'best':
                top_k_indices = np.argsort(correct)[::-1][:k]
            elif mode == 'worst':
                top_k_indices = np.argsort(mistakes)[::-1][:k]
            else:
                raise ValueError("top_k повинен бути або int, або ('best' | 'worst', k)")
        elif isinstance(top_k, int):
            top_k_indices = np.argsort(correct)[::-1][:top_k]
        else:
            raise TypeError("top_k повинен бути int або tuple")

        cm = cm[np.ix_(top_k_indices, top_k_indices)]
        class_labels = [class_labels[i] for i in top_k_indices]

    # Обрізаємо довгі підписи
    class_labels = [label[:25] for label in class_labels]

    plt.figure(figsize=(min(1 + 0.5 * len(class_labels), 20), min(1 + 0.5 * len(class_labels), 18)))
    vmax = cm.max()
    norm = LogNorm(vmin=1, vmax=vmax) if vmax >= 1 else None

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", norm=norm,
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Node ID")
    plt.ylabel("True Node ID")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
        #save_confusion_matrix_to_csv(cm, class_labels, file_path + '_topk')
    else:
        plt.show()


def visualize_aggregated_conf_matrix_old(
    cm_df,
    title="Aggregated Confusion Matrix",
    file_path=None,
    top_k=None,
    use_log_scale=True
):
    cm = cm_df.values
    labels = cm_df.index.tolist()

    if top_k is not None:
        correct = np.diag(cm)
        top_k_indices = np.argsort(correct)[::-1][:top_k]
        cm = cm[np.ix_(top_k_indices, top_k_indices)]
        labels = [labels[i] for i in top_k_indices]

    plt.figure(figsize=(min(1 + 0.5 * len(labels), 20), min(1 + 0.5 * len(labels), 18)))

    # Лог-норм з білим кольором для 0
    norm = LogNorm(vmin=cm[cm > 0].min(), vmax=cm.max()) if use_log_scale else None
    fmt = ".2g" if cm.dtype.kind == "f" else "d"  # короткий формат без зайвих нулів

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()

# Оновлення функції: фіксуємо 3 знаки після крапки, навіть для малих значень, і прибираємо нулі
# Оновлена функція форматування значень підписи з умовною точністю
# Оновлена функція: пусті клітинки (нулі) тепер білі на heatmap
def visualize_aggregated_conf_matrix(
    cm_df,
    title="Aggregated Confusion Matrix",
    file_path=None,
    top_k=None,
    use_log_scale=True
):
    cm = cm_df.values
    labels = cm_df.index.tolist()

    if top_k is not None:
        correct = np.diag(cm)
        top_k_indices = np.argsort(correct)[::-1][:top_k]
        cm = cm[np.ix_(top_k_indices, top_k_indices)]
        labels = [labels[i] for i in top_k_indices]

    # Формат аннотацій: умовна точність
    def format_cell(x):
        if x == 0:
            return ""
        elif x >= 100:
            return f"{x:.0f}"
        elif x >= 10:
            return f"{x:.1f}".rstrip("0").rstrip(".")
        elif x >= 1:
            return f"{x:.2f}".rstrip("0").rstrip(".")
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")

    annotations = np.vectorize(format_cell)(cm)

    # Маска для білих (нулевих) клітинок
    mask = cm == 0

    plt.figure(figsize=(min(1 + 0.5 * len(labels), 20), min(1 + 0.5 * len(labels), 18)))

    norm = LogNorm(vmin=cm[cm > 0].min(), vmax=cm.max()) if use_log_scale else None

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray",
        mask=mask,            # приховати нульові клітинки
        cbar_kws={"label": "Value"}
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
        print(f"Графік збережено у {file_path}")
    else:
        plt.show()




def visualize_confusion_matrix_bk(confusion_matrix_object, class_labels, file_path=None):
    """
    Візуалізує матрицю плутанини.

    :param confusion_matrix_object: Об'єкт із true_labels і predicted_labels.
    :param class_labels: Список міток класів.
    :param file_path: Шлях для збереження матриці плутанини (або None для показу).
    """
    true_labels = confusion_matrix_object["true_labels"]
    predicted_labels = confusion_matrix_object["predicted_labels"]

    #print("true_labels", true_labels)
    #print("predicted_labels", predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
    else:
        plt.show()

def generate_model_diagram(model, model_name=" Neural Network"):
    """
    Генерує діаграму для заданої моделі, аналізуючи її шари та метод forward.

    :param model: Модель PyTorch для аналізу.
    :param model_name: Назва моделі для заголовка діаграми.
    :return: Діаграма у вигляді об'єкта Digraph.
    """
    from graphviz import Digraph

    dot = Digraph(format='png', comment='Model Diagram')
    dot.attr(rankdir='LR', fontsize='12')  # Горизонтальна діаграма

    # Додаємо заголовок
    dot.attr(label=model_name, labelloc='t', fontsize='20', fontname='Helvetica')

    # Додаємо вузли для компонентів
    dot.node('Input', 'Input Features (x)', shape='ellipse')
    dot.node('EdgeIndex', 'Edge Index (edge_index)', shape='ellipse')
    dot.node('EdgeAttr', 'Edge Attributes (edge_attr)', shape='ellipse')
    dot.node('Batch', 'Batch Info (batch)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')

    dot.node('Conv1', 'GATConv (input_dim -> hidden_dim)', shape='box')
    dot.node('Activation1', 'ReLU Activation', shape='box')

    dot.node('Conv2', 'GATConv (hidden_dim -> hidden_dim)', shape='box')
    dot.node('GlobalPool', 'Global Mean Pooling', shape='box')

    dot.node('DocFC', 'Linear (doc_dim -> hidden_dim)', shape='box')
    dot.node('DocActivation', 'ReLU Activation (doc_emb)', shape='box')

    dot.node('Concat', 'Concatenation [x, doc_emb]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim + hidden_dim -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')

    # Додаємо зв'язки відповідно до forward
    dot.edges([
        ('Input', 'Conv1'),
        ('Conv1', 'Activation1'),
        ('Activation1', 'Conv2'),
        ('Conv2', 'GlobalPool'),
        ('DocFeatures', 'DocFC'),
        ('DocFC', 'DocActivation'),
        ('GlobalPool', 'Concat'),
        ('DocActivation', 'Concat'),
        ('Concat', 'FC'),
        ('FC', 'Sigmoid')
    ])

    return dot

def create_gnn_diagram():
    """
    Створює діаграму послідовності перетворень у GNN із використанням Graphviz.
    """
    dot = Digraph(format='png', comment='GNN Model Flow')

    # Додаємо вузли для кожного шару
    dot.node('Input', 'Input Features (x)', shape='ellipse')
    dot.node('EdgeIndex', 'Edge Index (edge_index)', shape='ellipse')
    dot.node('EdgeAttr', 'Edge Attributes (edge_attr)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')

    dot.node('Conv1', 'GATConv (input_dim -> hidden_dim)', shape='box')
    dot.node('Activation1', 'ReLU Activation', shape='box')

    dot.node('Conv2', 'GATConv (hidden_dim -> hidden_dim)', shape='box')
    dot.node('GlobalPool', 'Global Mean Pooling', shape='box')

    dot.node('DocFC', 'Linear (doc_dim -> hidden_dim)', shape='box')
    dot.node('DocActivation', 'ReLU Activation (doc_emb)', shape='box')

    dot.node('Concat', 'Concatenation [x, doc_emb]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim + hidden_dim -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')

    # Зв'язки між вузлами
    dot.edge('Input', 'Conv1', label='x')
    dot.edge('EdgeIndex', 'Conv1', label='edge_index')
    dot.edge('EdgeAttr', 'Conv1', label='edge_attr')
    dot.edge('Conv1', 'Activation1', label='hidden_dim')
    dot.edge('Activation1', 'Conv2', label='hidden_dim')
    dot.edge('EdgeIndex', 'Conv2', label='edge_index')
    dot.edge('EdgeAttr', 'Conv2', label='edge_attr')
    dot.edge('Conv2', 'GlobalPool', label='hidden_dim')
    dot.edge('GlobalPool', 'Concat', label='x (graph features)')

    dot.edge('DocFeatures', 'DocFC', label='doc_features')
    dot.edge('DocFC', 'DocActivation', label='hidden_dim')
    dot.edge('DocActivation', 'Concat', label='doc_emb')

    dot.edge('Concat', 'FC', label='hidden_dim + hidden_dim')
    dot.edge('FC', 'Sigmoid', label='output_dim')

    return dot

def create_cnn_diagram():
    """
    Створює діаграму послідовності перетворень у CNN із використанням Graphviz.
    """
    from graphviz import Digraph

    dot = Digraph(format='png', comment='CNN Model Flow')

    # Додаємо вузли для кожного шару
    dot.node('InputNodes', 'Node Features (node_features)', shape='ellipse')
    dot.node('InputEdges', 'Edge Features (edge_features)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')

    dot.node('Concat', 'Concatenation [node_features, edge_features]', shape='parallelogram')
    dot.node('Conv1', 'Conv1D (input_dim -> hidden_dim)', shape='box')
    dot.node('Activation1', 'ReLU Activation', shape='box')

    dot.node('Conv2', 'Conv1D (hidden_dim -> hidden_dim)', shape='box')
    dot.node('Pooling', 'Adaptive Avg Pooling', shape='box')

    dot.node('DocFC', 'Linear (doc_dim -> hidden_dim)', shape='box')
    dot.node('DocActivation', 'ReLU Activation (doc_emb)', shape='box')

    dot.node('ConcatFinal', 'Concatenation [x, doc_emb]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim + hidden_dim -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')

    # Зв'язки між вузлами
    dot.edge('InputNodes', 'Concat', label='node_features')
    dot.edge('InputEdges', 'Concat', label='edge_features')
    dot.edge('Concat', 'Conv1', label='input_dim')
    dot.edge('Conv1', 'Activation1', label='hidden_dim')
    dot.edge('Activation1', 'Conv2', label='hidden_dim')
    dot.edge('Conv2', 'Pooling', label='hidden_dim')
    dot.edge('Pooling', 'ConcatFinal', label='x (graph features)')

    dot.edge('DocFeatures', 'DocFC', label='doc_features')
    dot.edge('DocFC', 'DocActivation', label='hidden_dim')
    dot.edge('DocActivation', 'ConcatFinal', label='doc_emb')

    dot.edge('ConcatFinal', 'FC', label='hidden_dim + hidden_dim')
    dot.edge('FC', 'Sigmoid', label='output_dim')

    return dot

def create_rnn_diagram():
    """
    Створює діаграму послідовності перетворень у RNN із використанням Graphviz.
    """
    from graphviz import Digraph
    dot = Digraph(format='png', comment='RNN Model Flow')
    # Додаємо вузли для кожного шару
    dot.node('InputSequence', 'Input Sequence (node + edge features)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')
    dot.node('LSTM', 'BiLSTM (input_dim -> hidden_dim * 2)', shape='box')
    dot.node('MeanPooling', 'Mean Pooling', shape='box')
    dot.node('DocFC', 'Linear (doc_dim -> hidden_dim)', shape='box')
    dot.node('DocActivation', 'ReLU Activation (doc_emb)', shape='box')
    dot.node('ConcatFinal', 'Concatenation [sequence_emb, doc_emb]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim * 3 -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')
    # Зв'язки між вузлами
    dot.edge('InputSequence', 'LSTM', label='sequence')
    dot.edge('LSTM', 'MeanPooling', label='hidden_dim * 2')
    dot.edge('MeanPooling', 'ConcatFinal', label='sequence_emb')
    dot.edge('DocFeatures', 'DocFC', label='doc_features')
    dot.edge('DocFC', 'DocActivation', label='hidden_dim')
    dot.edge('DocActivation', 'ConcatFinal', label='doc_emb')
    dot.edge('ConcatFinal', 'FC', label='hidden_dim * 3')
    dot.edge('FC', 'Sigmoid', label='output_dim')
    return dot

def create_transformer_diagram():
    """
    Створює діаграму послідовності перетворень у Transformer із використанням Graphviz.
    """
    from graphviz import Digraph

    dot = Digraph(format='png', comment='Transformer Model Flow')

    # Додаємо вузли для кожного шару
    dot.node('InputSequence', 'Input Sequence (node + edge features)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features', shape='ellipse')

    dot.node('PositionalEncoding', 'Positional Encoding', shape='box')

    dot.node('Encoder', 'Transformer Encoder', shape='box')
    dot.node('Mask', 'Padding Mask', shape='parallelogram')

    dot.node('Pooling', 'Pooling (e.g., Mean Pooling)', shape='box')

    dot.node('DocFC', 'Linear (doc_dim -> hidden_dim)', shape='box')
    dot.node('DocActivation', 'ReLU Activation (doc_emb)', shape='box')

    dot.node('ConcatFinal', 'Concatenation [sequence_emb, doc_emb]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim + d_model -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')

    # Зв'язки між вузлами
    dot.edge('InputSequence', 'PositionalEncoding', label='sequence')
    dot.edge('PositionalEncoding', 'Encoder', label='sequence + positions')
    dot.edge('Mask', 'Encoder', label='mask')
    dot.edge('Encoder', 'Pooling', label='encoded_sequence')
    dot.edge('Pooling', 'ConcatFinal', label='sequence_emb')

    dot.edge('DocFeatures', 'DocFC', label='doc_features')
    dot.edge('DocFC', 'DocActivation', label='hidden_dim')
    dot.edge('DocActivation', 'ConcatFinal', label='doc_emb')

    dot.edge('ConcatFinal', 'FC', label='hidden_dim + d_model')
    dot.edge('FC', 'Sigmoid', label='output_dim')

    return dot

def create_autoencoder_diagram():
    """
    Створює діаграму послідовності перетворень у Autoencoder із використанням Graphviz.
    """
    from graphviz import Digraph

    dot = Digraph(format='png', comment='Autoencoder Model Flow')

    # Додаємо вузли для кожного компоненту
    dot.node('InputSequence', 'Input Sequence (node + edge features)', shape='ellipse')
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')

    dot.node('SequenceEncoder', 'Sequence Encoder (Linear -> ReLU -> Linear)', shape='box')
    dot.node('SequenceLatent', 'Sequence Latent Space', shape='parallelogram')

    dot.node('SequenceDecoder', 'Sequence Decoder (Linear -> ReLU -> Linear)', shape='box')
    dot.node('ReconstructedSequence', 'Reconstructed Sequence', shape='ellipse')

    dot.node('DocEncoder', 'Document Encoder (Linear -> ReLU -> Linear)', shape='box')
    dot.node('DocLatent', 'Document Latent Space', shape='parallelogram')

    dot.node('ConcatFinal', 'Concatenation [sequence_latent, doc_latent]', shape='parallelogram')
    dot.node('FC', 'Linear (hidden_dim * 2 -> output_dim)', shape='box')
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box')

    # Зв'язки між вузлами
    dot.edge('InputSequence', 'SequenceEncoder', label='sequence')
    dot.edge('SequenceEncoder', 'SequenceLatent', label='hidden_dim')
    dot.edge('SequenceLatent', 'SequenceDecoder', label='hidden_dim')
    dot.edge('SequenceDecoder', 'ReconstructedSequence', label='reconstructed_sequence')
    dot.edge('ReconstructedSequence', 'SequenceLatent', label='loss computation', style='dashed')

    dot.edge('DocFeatures', 'DocEncoder', label='doc_features')
    dot.edge('DocEncoder', 'DocLatent', label='hidden_dim')

    dot.edge('SequenceLatent', 'ConcatFinal', label='sequence_latent')
    dot.edge('DocLatent', 'ConcatFinal', label='doc_latent')

    dot.edge('ConcatFinal', 'FC', label='hidden_dim * 2')
    dot.edge('FC', 'Sigmoid', label='output_dim')

    return dot

from graphviz import Digraph

def create_cnn_diagram_colors():
    """
    Створює діаграму з пастельними кольорами для CNN із 4 шарами, з адаптацією для друку.
    """
    dot = Digraph(format='png', comment='Compact CNN Model Flow with Pastel Colors')

    # Вхідні дані
    dot.node('InputNodes', 'Node with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('InputEdges', 'Edge with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('DocFeatures', 'Document Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue

    # Перетворення графових ознак
    dot.node('Concat', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0', fontcolor='black')  # Light Grey
    dot.node('Conv1', 'Conv1D (Layer 1)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Activation1', 'ReLU (Layer 1)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    dot.node('Conv2', 'Conv1D (Layer 2)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Activation2', 'ReLU (Layer 2)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    dot.node('Conv3', 'Conv1D (Layer 3)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Activation3', 'ReLU (Layer 3)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    dot.node('Conv4', 'Conv1D (Layer 4)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Activation4', 'ReLU (Layer 4)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow
    dot.node('Pooling', 'Adaptive Avg Pooling', shape='box', style='filled', color='#fce5cd', fontcolor='black')  # Pastel Orange

    # Перетворення текстових ознак
    dot.node('DocFC', 'Linear (Text Embedding)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('DocActivation', 'ReLU (Text Embedding)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    # Інтеграція ознак
    dot.node('ConcatFinal', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0', fontcolor='black')  # Light Grey
    dot.node('FC', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3', fontcolor='black')  # Pastel Blue
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box', style='filled', color='#f4cccc', fontcolor='black')  # Pastel Red

    # Зв'язки між вузлами
    dot.edge('InputNodes', 'Concat',  color='black')
    dot.edge('InputEdges', 'Concat',  color='black')
    dot.edge('Concat', 'Conv1', color='black')
    dot.edge('Conv1', 'Activation1', color='black')
    dot.edge('Activation1', 'Conv2', color='black')
    dot.edge('Conv2', 'Activation2', color='black')
    dot.edge('Activation2', 'Conv3', color='black')
    dot.edge('Conv3', 'Activation3', color='black')
    dot.edge('Activation3', 'Conv4', color='black')
    dot.edge('Conv4', 'Activation4', color='black')
    dot.edge('Activation4', 'Pooling', color='black')
    dot.edge('Pooling', 'ConcatFinal', color='black')

    dot.edge('DocFeatures', 'DocFC', color='black')
    dot.edge('DocFC', 'DocActivation', color='black')
    dot.edge('DocActivation', 'ConcatFinal', color='black')

    dot.edge('ConcatFinal', 'FC', color='black')
    dot.edge('FC', 'Sigmoid', color='black')

    return dot

def create_gnn_diagram_colors():
    """
    Створює діаграму послідовності перетворень у GNN із використанням пастельних кольорів.
    """
    dot = Digraph(format='png', comment='GNN Model Flow with Pastel Colors')

    # Вхідні дані
    dot.node('Input', 'Node with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('EdgeAttr', 'Edge with Attributes', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('DocFeatures', 'Document Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue

    # Перетворення графових ознак
    dot.node('Conv1', 'GATConv (Layer 1)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Activation1', 'ReLU (Layer 1)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    dot.node('Conv2', 'GATConv (Layer 2)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Conv3', 'GATConv (Layer 3)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('GlobalPool', 'Global Mean Pooling', shape='box', style='filled', color='#fce5cd', fontcolor='black')  # Pastel Orange

    # Перетворення текстових ознак
    dot.node('DocFC', 'Linear (Text Embedding)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('DocActivation', 'ReLU (Text Embedding)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    # Інтеграція ознак
    dot.node('Concat', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0', fontcolor='black')  # Light Grey
    dot.node('FC', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3', fontcolor='black')  # Pastel Blue
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box', style='filled', color='#f4cccc', fontcolor='black')  # Pastel Red

    # Зв'язки між вузлами
    dot.edge('Input', 'Conv1',  color='black')
    dot.edge('EdgeAttr', 'Conv1',  color='black')
    dot.edge('Conv1', 'Activation1',  color='black')
    dot.edge('Activation1', 'Conv2', color='black')
    dot.edge('Conv2', 'Conv3', color='black')
    dot.edge('Conv3', 'GlobalPool',  color='black')
    dot.edge('GlobalPool', 'Concat', color='black')

    dot.edge('DocFeatures', 'DocFC',  color='black')
    dot.edge('DocFC', 'DocActivation',  color='black')
    dot.edge('DocActivation', 'Concat',  color='black')

    dot.edge('Concat', 'FC',  color='black')
    dot.edge('FC', 'Sigmoid', color='black')

    return dot

def create_rnn_diagram_colors():
    """
    Створює діаграму послідовності перетворень у RNN із використанням пастельних кольорів.
    """
    dot = Digraph(format='png', comment='RNN Model Flow with Pastel Colors')

    # Вхідні дані
    dot.node('InputSequence', 'Sequence: Node with Features + Edge with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('DocFeatures', 'Document Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue

    # Обробка послідовності
    dot.node('LSTM', 'BiLSTM', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('MeanPooling', 'Mean Pooling', shape='box', style='filled', color='#fce5cd', fontcolor='black')  # Pastel Orange

    # Обробка документних ознак
    dot.node('DocFC', 'Linear (Text Embedding)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('DocActivation', 'ReLU (Text Embedding)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    # Інтеграція ознак
    dot.node('ConcatFinal', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0', fontcolor='black')  # Light Grey
    dot.node('FC', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3', fontcolor='black')  # Pastel Blue
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box', style='filled', color='#f4cccc', fontcolor='black')  # Pastel Red

    # Зв'язки між вузлами
    dot.edge('InputSequence', 'LSTM',  color='black')
    dot.edge('LSTM', 'MeanPooling', color='black')
    dot.edge('MeanPooling', 'ConcatFinal',  color='black')

    dot.edge('DocFeatures', 'DocFC',  color='black')
    dot.edge('DocFC', 'DocActivation',  color='black')
    dot.edge('DocActivation', 'ConcatFinal', color='black')

    dot.edge('ConcatFinal', 'FC', color='black')
    dot.edge('FC', 'Sigmoid', color='black')

    return dot

def create_transformer_diagram_colors():
    """
    Створює діаграму послідовності перетворень у Transformer із 3 шарами та використанням пастельних кольорів.
    """
    dot = Digraph(format='png', comment='Transformer Model Flow with 3 Layers and Pastel Colors')

    # Вхідні дані
    dot.node('InputSequence', 'Sequence: Node with Features + Edge with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue
    dot.node('DocFeatures', 'Document Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue

    # Positional Encoding
    dot.node('PositionalEncoding', 'Positional Encoding', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green

    # Transformer Layers
    dot.node('Encoder1', 'Transformer Encoder (Layer 1)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Encoder2', 'Transformer Encoder (Layer 2)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Encoder3', 'Transformer Encoder (Layer 3)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green

    # Pooling Layer
    dot.node('Pooling', 'Pooling (e.g., Mean Pooling)', shape='box', style='filled', color='#fce5cd', fontcolor='black')  # Pastel Orange

    # Обробка документних ознак
    dot.node('DocFC', 'Linear (Text Embedding)', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('DocActivation', 'ReLU (Text Embedding)', shape='box', style='filled', color='#fff2cc', fontcolor='black')  # Pastel Yellow

    # Інтеграція ознак
    dot.node('ConcatFinal', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0', fontcolor='black')  # Light Grey
    dot.node('FC', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3', fontcolor='black')  # Pastel Blue
    dot.node('Sigmoid', 'Sigmoid Activation', shape='box', style='filled', color='#f4cccc', fontcolor='black')  # Pastel Red

    # Зв'язки між вузлами
    dot.edge('InputSequence', 'PositionalEncoding', color='black')
    dot.edge('PositionalEncoding', 'Encoder1', color='black')
    dot.edge('Encoder1', 'Encoder2', color='black')
    dot.edge('Encoder2', 'Encoder3', color='black')
    dot.edge('Encoder3', 'Pooling',  color='black')
    dot.edge('Pooling', 'ConcatFinal',  color='black')

    dot.edge('DocFeatures', 'DocFC',  color='black')
    dot.edge('DocFC', 'DocActivation',  color='black')
    dot.edge('DocActivation', 'ConcatFinal',  color='black')

    dot.edge('ConcatFinal', 'FC',  color='black')
    dot.edge('FC', 'Sigmoid',  color='black')

    return dot

def create_autoencoder_diagram_colors():
    """
    Створює діаграму для Autoencoder із пастельними кольорами та зв'язком Reconstructed Sequence.
    """
    dot = Digraph(format='png', comment='Autoencoder Model Flow with Reconstructed Sequence Link')

    # Вхідні дані
    dot.node('InputSequence', 'Sequence: Node with Features + Edge with Features', shape='ellipse', style='filled', color='#add8e6', fontcolor='black')  # Light Blue

    dot.node('DocFeatures', 'Document Features', shape='ellipse', style='filled', color='#add8e6',
             fontcolor='black')  # Light Blue

    # Encoder-Decoder Layers
    dot.node('Encoder', 'Encoder', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('Decoder', 'Decoder', shape='box', style='filled', color='#d9ead3', fontcolor='black')  # Pastel Green
    dot.node('ReconstructedSeq', 'Reconstructed Sequence', shape='box', style='filled', color='#add8e6',
             fontcolor='black')  # Light Blue

    # Aggregation Layer
    dot.node('Aggregation', 'Mean Aggregation', shape='box', style='filled', color='#fce5cd',
             fontcolor='black')  # Pastel Orange

    # Document Encoding
    dot.node('DocEncoder', 'Document Encoder', shape='box', style='filled', color='#d9ead3',
             fontcolor='black')  # Pastel Green

    # Latent Space Integration
    dot.node('ConcatLatent', 'Concatenation', shape='parallelogram', style='filled', color='#f0f0f0',
             fontcolor='black')  # Light Grey

    # Classification
    dot.node('FinalFC', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3',
             fontcolor='black')  # Pastel Blue
    dot.node('Output', 'Classification Output', shape='box', style='filled', color='#f4cccc',
             fontcolor='black')  # Pastel Red

    # Зв'язки між вузлами
    dot.edge('InputSequence', 'Encoder',  color='black')
    dot.edge('Encoder', 'Decoder', color='black')
    dot.edge('Decoder', 'ReconstructedSeq',  color='black')
    dot.edge('ReconstructedSeq', 'Aggregation',  color='black')
    dot.edge('Encoder', 'Aggregation',  color='black')
    dot.edge('Aggregation', 'ConcatLatent',  color='black')

    dot.edge('DocFeatures', 'DocEncoder',  color='black')
    dot.edge('DocEncoder', 'ConcatLatent',  color='black')

    dot.edge('ConcatLatent', 'FinalFC',  color='black')
    dot.edge('FinalFC', 'Output',  color='black')

    return dot


def create_generalized_diagram():
    dot = Digraph(format='png', comment='General Architecture for Graph and Process Features')

    # Вхідні дані
    dot.node('InputGraph', 'Graph Data (Nodes + Edges) Features', shape='ellipse', style='filled', color='#add8e6',
             fontcolor='black')  # Light Blue
    dot.node('ProcessFeatures', 'General Process Features', shape='ellipse', style='filled', color='#add8e6',
             fontcolor='black')  # Light Blue

    # Графова гілка
    dot.node('GraphInputLayer', 'Input Layer', shape='box', style='filled', color='#d9ead3',
             fontcolor='black')  # Pastel Green
    dot.node('GraphHidden1', 'Hidden Layer 1', shape='box', style='filled', color='#fff2cc',
             fontcolor='black')  # Pastel Yellow
    dot.node('GraphHiddenN', 'Hidden Layer N', shape='box', style='filled', color='#fff2cc',
             fontcolor='black')  # Pastel Yellow

    # Гілка загальних ознак
    dot.node('ProcessInputLayer', 'Input Layer (Process Features)', shape='box', style='filled', color='#d9ead3',
             fontcolor='black')  # Pastel Green
    dot.node('ProcessHidden1', 'Hidden Layer', shape='box', style='filled', color='#fff2cc',
             fontcolor='black')  # Pastel Yellow

    # Об'єднання ознак
    dot.node('FeatureConcat', 'Feature Concatenation', shape='parallelogram', style='filled', color='#f0f0f0',
             fontcolor='black')  # Light Grey

    # Повнозв'язний шар
    dot.node('FullyConnected', 'Fully Connected Layer', shape='box', style='filled', color='#cfe2f3',
             fontcolor='black')  # Pastel Blue
    dot.node('Output', 'Output (Prediction)', shape='ellipse', style='filled', color='#f4cccc',
             fontcolor='black')  # Pastel Red

    # Зв’язки для графової гілки
    dot.edge('InputGraph', 'GraphInputLayer', color='black')
    dot.edge('GraphInputLayer', 'GraphHidden1', color='black')
    dot.edge('GraphHidden1', 'GraphHiddenN', color='black')
    dot.edge('GraphHiddenN', 'FeatureConcat', color='black')

    # Зв’язки для гілки загальних ознак
    dot.edge('ProcessFeatures', 'ProcessInputLayer', color='black')
    dot.edge('ProcessInputLayer', 'ProcessHidden1', color='black')
    dot.edge('ProcessHidden1', 'FeatureConcat', color='black')

    # Об’єднання та вихід
    dot.edge('FeatureConcat', 'FullyConnected', color='black')
    dot.edge('FullyConnected', 'Output', color='black')

    return dot

    # Створення та рендеринг діаграми
    diagram = create_general_architecture_diagram()
    diagram.render('general_architecture_diagram', view=True)
