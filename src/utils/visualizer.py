import networkx as nx
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from src.utils.graph_utils import clean_graph
import pandas as pd

logger = get_logger(__name__)


def save_training_diagram(stats, file_path, test_stats=None):
    """
    Зберігає графік навчання та валідації, додаючи, за потреби, тестові метрики.

    :param stats: Статистика навчання (словник зі списками по епохах).
    :param file_path: Шлях для збереження графіка.
    :param test_stats: Метрики тестових даних (словник).
    """
    plt.figure()

    # Графіки навчання та валідації
    plt.plot(stats['epochs'], stats['train_loss'], label='Train Loss')
    plt.plot(stats['epochs'], stats['val_precision'], label='Validation Precision')
    plt.plot(stats['epochs'], stats['val_recall'], label='Validation Recall')
    plt.plot(stats['epochs'], stats['val_roc_auc'], label='Validation ROC AUC')

    # Тестові метрики (додаються, якщо значення не None)
    if test_stats:
        if test_stats.get('precision') is not None:
            plt.axhline(y=test_stats['precision'], color='blue', linestyle='--', label='Test Precision')
        if test_stats.get('recall') is not None:
            plt.axhline(y=test_stats['recall'], color='green', linestyle='--', label='Test Recall')
        if test_stats.get('roc_auc') is not None:
            plt.axhline(y=test_stats['roc_auc'], color='red', linestyle='--', label='Test ROC AUC')

    # Оформлення графіка
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(file_path)
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

    plt.figure(figsize=(140, 90))

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
