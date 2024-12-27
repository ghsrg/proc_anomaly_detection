from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def clean_graph(graph):
    """
    Видаляє всі вузли (крім "startevent" та "endevent"), якщо параметр active_executions дорівнює 0 або відсутній.
    :param graph: NetworkX граф.
    """
    nodes_to_remove = []

    # Перебираємо вузли графа
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', '').lower()
        active_executions = attrs.get('active_executions', None)

        # Перевіряємо умови для видалення
        if node_type not in ['startevent-', 'endevent-','endeventsp-','starteventsp-'] and (active_executions is None or active_executions == 0):
            #logger.debug(attrs, variable_name="attrs", max_lines=10)
            nodes_to_remove.append(node_id)

    # Видаляємо вузли та їхні зв'язки
    graph.remove_nodes_from(nodes_to_remove)

    return graph


def inspect_graph(graph):
    #logger.info(f" Інспектація графа ...")
    for node, attrs in graph.nodes(data=True):
        for key, value in attrs.items():
            if isinstance(value, pd.Series):
                logger.error(f"Вузол {node} має атрибут {key} з типом pandas.Series: {value}")
            elif isinstance(value, (dict, set)):
                logger.error(f"Вузол {node} має атрибут {key} з несумісним типом: {type(value)}")
    for u, v, attrs in graph.edges(data=True):
        for key, value in attrs.items():
            if isinstance(value, pd.Series):
                logger.error(f"Ребро {u}->{v} має атрибут {key} з типом pandas.Series: {value}")
            elif isinstance(value, (dict, set)):
                logger.error(f"Ребро {u}->{v} має атрибут {key} з несумісним типом: {type(value)}")

