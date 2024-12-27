import networkx as nx
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from src.utils.graph_utils import clean_graph
import pandas as pd

logger = get_logger(__name__)

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

def visualize_graph(graph):
    """
    Візуалізація графа NetworkX із більш зрозумілими підписами вузлів/ребер.
    У вузлах відображаємо BPMN name або task_subject, а на ребрах – назву переходу чи conditionExpression.
    """

    plt.figure(figsize=(15, 10))
    # Spring layout можна замінити на інший (наприклад, shell_layout, kamada_kawai_layout)
    pos = nx.spring_layout(graph, k=0.7, iterations=50)

    # Формуємо підписи для вузлів
    node_labels = {}
    colors = []
    for node, data in graph.nodes(data=True):
        # Підпис вузла: BPMN-ім'я або тема задачі
        bpmn_name = data.get('name', node)
        subject = data.get('task_subject', '')

        # Можна виводити і те, й інше:
        # label_text = f"{bpmn_name}\n({subject})" if subject else bpmn_name
        # або лише BPMN name, якщо так зручніше
        label_text = bpmn_name

        node_labels[node] = label_text

        # Фарбування вузлів за типом
        node_type = data.get('type', '').lower()
        if 'startevent' in node_type:
            colors.append('lightgreen')
        elif 'endevent' in node_type:
            colors.append('lightcoral')
        elif 'gateway' in node_type:
            colors.append('yellow')
        else:
            colors.append('lightblue')

    # Формуємо підписи для ребер
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        # Спробуємо витягнути conditionExpression або назву переходу
        cond_expr = data.get('conditionExpression', '')
        flow_name = data.get('name', '')

        # Якщо є conditionExpression, пріоритетніше його показати
        if cond_expr:
            edge_labels[(u, v)] = cond_expr
        elif flow_name:
            edge_labels[(u, v)] = flow_name
        else:
            edge_labels[(u, v)] = ''  # порожній підпис

    # Малюємо сам граф
    nx.draw(
        graph,
        pos,
        labels=node_labels,
        node_color=colors,
        with_labels=True,
        node_size=1500,
        font_size=8,
        edge_color='gray',
        arrowsize=12
    )

    # Підписи на ребрах
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Visualization of BPMN Process Graph", fontsize=14)
    plt.axis("off")  # Прибираємо осі
    plt.tight_layout()
    plt.show()
