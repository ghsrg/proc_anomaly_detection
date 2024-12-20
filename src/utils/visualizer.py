# Функції для візуалізації результатів
from matplotlib import pyplot as plt
import networkx as nx


def visualize_graph(graph, title="Process Graph", path=""):
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