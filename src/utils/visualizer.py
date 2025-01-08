import networkx as nx
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from src.utils.graph_utils import clean_graph
import pandas as pd
from graphviz import Digraph
import inspect
import torch.nn as nn

logger = get_logger(__name__)

import matplotlib.pyplot as plt


def save_training_diagram(stats, file_path, test_stats=None):
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
    pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')

    plt.figure(figsize=(98, 55))

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
    dot.node('DocFeatures', 'Document Features (doc_features)', shape='ellipse')

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
