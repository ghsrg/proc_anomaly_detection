import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.utils.logger import get_logger
from src.utils.graph_utils import clean_graph
from src.utils.file_utils import save2csv, save_confusion_matrix_to_csv
import pandas as pd
from graphviz import Digraph
import inspect
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        'MAE': ('val_mae', 'purple', 0.5),
        'RMSE': ('val_rmse', 'red', 2.5),
        'R2': ('val_r2', 'brown', 1)
    }

    num_epochs = len(stats['epochs'])
    step = max(1, num_epochs // 6)  # Крок для підписів

    for label, (key, color, lw) in metrics.items():
        if key not in stats:
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


def visualize_distribution(node_distribution, edge_distribution, file_path=None):
    """
    Візуалізує розподіл кількості вузлів та зв'язків у графах.

    :param node_distribution: Розподіл кількості вузлів.
    :param edge_distribution: Розподіл кількості зв'язків.
    """
    plt.figure(figsize=(22, 12))

    # Розподіл кількості вузлів
    plt.plot(
        sorted(node_distribution.keys()),
        [node_distribution[k] for k in sorted(node_distribution.keys())],
        label='Node Count Distribution',
        linestyle='-',
        color='blue'
    )

    # Розподіл кількості зв'язків
    plt.plot(
        sorted(edge_distribution.keys()),
        [edge_distribution[k] for k in sorted(edge_distribution.keys())],
        label='Edge Count Distribution',
        linestyle='--',
        color='green'
    )

    plt.xlabel('Count')
    plt.ylabel('Number of Graphs')
    plt.title('Node and Edge Count Distributions')
    plt.legend()
    plt.grid(True)
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f"Графік розподілення збережено у {file_path}")
    else:
        plt.show()


def visualize_confusion_matrix(confusion_matrix_object, class_labels=None, file_path=None, top_k=None,
                                true_node_ids=None):
    if isinstance(confusion_matrix_object, dict):
        true_labels = confusion_matrix_object["true_labels"]
        predicted_labels = confusion_matrix_object["predicted_labels"]
        cm = confusion_matrix(true_labels, predicted_labels)
    else:
        cm = confusion_matrix_object
        true_labels = list(range(cm.shape[0]))
        predicted_labels = list(range(cm.shape[1]))

    if class_labels is None:
        if true_node_ids is not None:
            unique_indices = sorted(set(true_labels + predicted_labels))
            try:
                class_labels = [
                    true_node_ids[i][:25] if i < len(true_node_ids) else str(i)
                    for i in unique_indices
                ]
            except Exception as e:
                print("ERROR mapping class_labels from node_ids:", e)
                class_labels = [str(i) for i in unique_indices]
        else:
            class_labels = [str(i) for i in range(cm.shape[0])]

        # Збереження повної матриці перед top_k
    if file_path:
        save_confusion_matrix_to_csv(cm, class_labels, file_path+ '_full')

        # Обробка top_k
    if top_k is not None and len(class_labels) == cm.shape[0]:
        if isinstance(top_k, tuple):
            mode, k = top_k
            row_sums = np.sum(cm, axis=1)
            correct = np.diag(cm)
            mistakes = row_sums - correct

            if mode == 'best':
                top_k_indices = np.argsort(correct)[::-1][:k]  # Найбільше правильних
            elif mode == 'worst':
                top_k_indices = np.argsort(mistakes)[::-1][:k]  # Найбільше помилок
            else:
                raise ValueError("top_k повинен бути або int, або ('best' | 'worst', k)")
        elif isinstance(top_k, int):
            correct = np.diag(cm)
            top_k_indices = np.argsort(correct)[::-1][:top_k]  # Стандартна поведінка
        else:
            raise TypeError("top_k повинен бути int або tuple")

        cm = cm[np.ix_(top_k_indices, top_k_indices)]
        class_labels = [class_labels[i] for i in top_k_indices]

        # Обрізка назв
    class_labels = [label[:25] for label in class_labels]

    plt.figure(figsize=(min(1 + 0.5 * len(class_labels), 20), min(1 + 0.5 * len(class_labels), 18)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", norm=LogNorm(vmin=1, vmax=cm.max()), xticklabels=class_labels, yticklabels=class_labels)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Node ID")
    plt.ylabel("True Node ID")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=100)
        plt.close()
        # Також зберігаємо з актуальною top_k матрицею
        save_confusion_matrix_to_csv(cm, class_labels, file_path + '_topk')
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
