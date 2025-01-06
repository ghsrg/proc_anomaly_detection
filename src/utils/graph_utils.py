from src.utils.logger import get_logger
import pandas as pd
from datetime import datetime

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


def format_graph_values(graph, numeric_attrs=None, date_attrs=None, default_numeric=0.0, default_date="1970-01-01T00:00:00.0", default_string=" "):
    """
    Форматує значення атрибутів графа, забезпечуючи правильний формат чисел, дат і рядків.

    :param graph: Граф NetworkX.
    :param numeric_attrs: Список атрибутів, які повинні бути числовими.
    :param date_attrs: Список атрибутів, які повинні бути датами.
    :param default_numeric: Значення за замовчуванням для некоректних числових атрибутів.
    :param default_date: Значення за замовчуванням для некоректних датованих атрибутів.
    :param default_string: Значення за замовчуванням для некоректних рядкових атрибутів.
    :return: Відформатований граф.
    """
    formatted_graph = graph.copy()

    numeric_attrs = numeric_attrs or []
    date_attrs = date_attrs or []
    default_date_obj = datetime.strptime(default_date, "%Y-%m-%dT%H:%M:%S.%f")

    for node, data in formatted_graph.nodes(data=True):
        # Форматування числових значень
        for attr in numeric_attrs:
            if attr in data:
                value = data[attr]
                try:
                    if isinstance(value, str):
                        # Видаляємо пробіли, коми, валюти та перетворюємо в float
                        value = value.replace(" ", "").replace(",", ".").split()[0]
                    data[attr] = float(value)
                except (ValueError, TypeError):
                    logger.debug(f"Атрибут numeric '{attr}' не вдалося обробити: {value}")
                    data[attr] = default_numeric

        # Форматування дат
        for attr in date_attrs:
            if attr in data:
                value = data[attr]
                try:
                    logger.debug(f"Атрибут value {value}, має тип '{type(value)}'.")
                    if value and isinstance(value, str):
                        logger.debug(f"Атрибут value '{value}'.")
                        date_obj = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
                        logger.debug(f"Атрибут дати '{attr}' date_obj =  {date_obj}.")
                        data[attr] = int(date_obj.timestamp())
                    else:
                        raise ValueError
                except (ValueError, TypeError):
                    logger.debug(f"Атрибут дати '{attr}' не вдалося обробити: {value}. Використовуємо дефолт.")
                    try:
                        data[attr] = int(default_date_obj.timestamp())
                    except (OSError, OverflowError):
                        logger.debug(f"Дефолтну дату '{default_date}' неможливо конвертувати в Unix time. Використовуємо 0.")
                        data[attr] = 0
                logger.debug(f"Атрибут '{value}' -> : {data[attr]}")

        # Замінюємо None значення для всіх інших атрибутів
        for attr, value in data.items():
            if value is None:
                logger.debug(f"Атрибут '{attr}' має значення None. Замінюємо на '{default_string}'.")
                data[attr] = default_string

    return formatted_graph

def format_doc_values(doc_json, numeric_attrs=None, date_attrs=None, default_numeric=0.0, default_date="2000-01-01T00:00:00", default_string=" "):
    """
    Форматує значення атрибутів документа (JSON), забезпечуючи правильний формат чисел, дат і рядків.

    :param doc_json: Словник атрибутів документа.
    :param numeric_attrs: Список атрибутів, які повинні бути числовими.
    :param date_attrs: Список атрибутів, які повинні бути датами.
    :param default_numeric: Значення за замовчуванням для некоректних числових атрибутів.
    :param default_date: Значення за замовчуванням для некоректних датованих атрибутів.
    :param default_string: Значення за замовчуванням для некоректних рядкових атрибутів.
    :return: Відформатований словник документа.
    """
    numeric_attrs = numeric_attrs or []
    date_attrs = date_attrs or []

    # Спроба конвертації default_date
    # Спроба конвертації default_date
    try:
        default_date_obj = datetime.strptime(default_date, "%Y-%m-%dT%H:%M:%S")
        default_date_timestamp = int(default_date_obj.timestamp())
    except (ValueError, OSError):
        logger.error(f"Некоректна дефолтна дата: {default_date}. Використовуємо 0.")
        default_date_timestamp = 0

    formatted_doc = {}

    for attr, value in doc_json.items():
        # Форматування числових значень
        if attr in numeric_attrs:
            try:
                if isinstance(value, str):
                    # Видаляємо пробіли, коми, валюти та перетворюємо в float
                    value = value.replace(" ", "").replace(",", ".").split()[0]
                formatted_doc[attr] = float(value)
            except (ValueError, TypeError):
                logger.debug(f"Атрибут numeric '{attr}' не вдалося обробити: {value}")
                formatted_doc[attr] = default_numeric

        # Форматування дат
        elif attr in date_attrs:
            try:
                if value and isinstance(value, str):
                    date_obj = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
                    formatted_doc[attr] = int(date_obj.timestamp())
                elif value is None:
                    formatted_doc[attr] = default_date_timestamp
                else:
                    raise ValueError
            except (ValueError, TypeError):
                logger.debug(f"Атрибут дати '{attr}' не вдалося обробити: {value}. Використовуємо дефолт.")
                formatted_doc[attr] = default_date_timestamp

        # Замінюємо None значення для всіх інших атрибутів
        else:
            if value is None:
                formatted_doc[attr] = default_string
            else:
                formatted_doc[attr] = value

    return formatted_doc
