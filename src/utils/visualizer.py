# Функції для візуалізації результатів
from matplotlib import pyplot as plt
import networkx as nx


def visualize_graph1(graph, title="Process Graph", path=""):
    nx.draw(
        graph,
        with_labels=True,
        labels=nx.get_node_attributes(graph, 'name'),
        node_color="lightblue"
    )
    plt.title(title)
    if path and path.strip():
        plt.savefig(f"{path}.png", dpi=300)
    else:
        plt.show()


def visualize_graph(graph):
    """
    Візуалізація графа NetworkX з атрибутами task_subject та duration_full.
    :param graph: Граф NetworkX.
    """
    pos = nx.spring_layout(graph)  # Розташування вузлів
    plt.figure(figsize=(15, 10))

    # Відображення вузлів
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color="gray",
    )

    # Додавання підписів до вузлів (task_subject)
    node_labels = nx.get_node_attributes(graph, 'task_subject')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)

    # Додавання підписів до ребер (duration_full)
    edge_labels = nx.get_edge_attributes(graph, 'duration_full')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Visualization of the Process Graph")
    plt.show()