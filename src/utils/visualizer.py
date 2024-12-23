import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph_with_dot(graph):
    """
    Візуалізація графа NetworkX у стилі BPMN із використанням Graphviz 'dot'.
    У вузлах відображаємо BPMN name або task_subject, фарбуємо вузли за типом.
    На ребрах показуємо conditionExpression або name (якщо є).
    """

    # Використовуємо layout від Graphviz з алгоритмом 'dot'
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')

    # Підготовка полотна
    plt.figure(figsize=(15, 10))

    # Підписи та кольори для вузлів
    node_labels = {}
    colors = []

    for node, data in graph.nodes(data=True):
        # Витягаємо назву й тип
        bpmn_name = data.get('name', node)
        subject = data.get('task_subject', '')
        node_type = data.get('type', '').lower()

        # Формуємо напис
        label_text = bpmn_name  # або f"{bpmn_name}\n({subject})" якщо треба обидва
        node_labels[node] = label_text

        # Логіка фарбування:
        if 'startevent' in node_type:
            colors.append('lightgreen')
        elif 'endevent' in node_type:
            colors.append('lightcoral')
        elif 'gateway' in node_type:
            colors.append('yellow')
        # Якщо це userTask або (callActivity / subProcess):
        elif node_type in ['usertask', 'subprocess', 'callactivity']:
            colors.append('cornflowerblue')  # Інший відтінок синього
        else:
            colors.append('lightblue')  # Решта задач у стандартному блакитному

    # Підписи для ребер
    edge_labels = {}
    for u, v, edge_data in graph.edges(data=True):
        cond_expr = edge_data.get('conditionExpression', '')
        flow_name = edge_data.get('name', '')
        if cond_expr:
            edge_labels[(u, v)] = cond_expr
        elif flow_name:
            edge_labels[(u, v)] = flow_name
        else:
            edge_labels[(u, v)] = ''

    # Малюємо вузли
    nx.draw(
        graph,
        pos,
        labels=node_labels,
        node_color=colors,
        with_labels=True,
        node_size=1500,
        font_size=8,
        edge_color='gray',
        arrows=True,
        arrowsize=12
    )

    # Додаємо підписи ребер
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("BPMN Graph using Graphviz 'dot'", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
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
