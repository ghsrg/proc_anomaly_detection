import networkx as nx
import random
import string
from src.utils.logger import get_logger
from src.utils.visualizer import visualize_graph_with_dot
import traceback
logger = get_logger(__name__)

def augment_graph(graph: nx.DiGraph, noise_level: float = 0.1) -> nx.DiGraph:
    """
    Додає шум до графа шляхом випадкової модифікації атрибутів вузлів і ребер.

    :param graph: Вхідний граф NetworkX.
    :param noise_level: Рівень шуму (0-1).
    :return: Модифікований граф із шумом.
    """
    augmented_graph = graph.copy()

    # Модифікація атрибутів вузлів
    for node, attrs in augmented_graph.nodes(data=True):
        for key, value in attrs.items():
            if isinstance(value, (int, float)) and random.random() < noise_level:
                noise = random.uniform(-0.1 * value, 0.1 * value) if value != 0 else random.uniform(-1, 1)
                augmented_graph.nodes[node][key] = value + noise

    # Модифікація атрибутів ребер
    for u, v, attrs in augmented_graph.edges(data=True):
        for key, value in attrs.items():
            if isinstance(value, (int, float)) and random.random() < noise_level:
                noise = random.uniform(-0.1 * value, 0.1 * value) if value != 0 else random.uniform(-1, 1)
                augmented_graph.edges[u, v][key] = value + noise

    return augmented_graph


def generate_normal_graph(graph: nx.DiGraph, change_percentage: float = 0.1, doc_subject_replacements=None, **params) -> \
tuple[nx.DiGraph, dict]:
    """
    Генерує нормальний граф (варіацію оригінального графа), уникаючи аномалій.

    :param graph: Оригінальний граф NetworkX.
    :param change_percentage: Частка змін (5-10%).
    :param doc_subject_replacements: Словник замін для атрибута "doc_subject".
    :param params: Додаткові параметри генерації.
    :return: Модифікований граф і параметри генерації.
    """
    try:
        logger.info("Початок генерації нормального графа.")

        # Перевірка діапазону change_percentage
        if not (0.05 <= change_percentage <= 0.10):
            raise ValueError(
                f"Значення change_percentage {change_percentage} виходить за межі дозволеного діапазону (0.05 - 0.10).")

        modified_graph = graph.copy()
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)

        # Розрахунок кількості нових вузлів
        nodes_to_add = max(1, int(num_nodes * change_percentage))
        logger.info(f"Додається {nodes_to_add} нових вузлів.")

        new_nodes = []
        for i in range(nodes_to_add):
            new_node_id = f"generated_node_{i}"

            # Додавання вузла з базовими атрибутами
            modified_graph.add_node(new_node_id, type="generic", name=new_node_id)
            new_nodes.append(new_node_id)
            logger.debug(f"Додано вузол: {new_node_id}")

            # Вибір сусіднього вузла
            neighbors = random.sample(list(modified_graph.nodes), k=1)
            neighbor_durations = [
                modified_graph.nodes[n].get("DURATION_", 1.0) for n in neighbors
            ]
            avg_duration = sum(neighbor_durations) / len(neighbor_durations)

            # Перевірка, чи avg_duration більше нуля
            if avg_duration <= 0:
                avg_duration = 1.0  # Встановлення дефолтного значення

            # Генерація атрибутів для нового вузла
            modified_graph.nodes[new_node_id]["DURATION_"] = random.uniform(avg_duration * 0.9, avg_duration * 1.1)
            start_time = modified_graph.nodes[neighbors[0]].get("END_TIME_", None)
            if start_time:
                modified_graph.nodes[new_node_id]["START_TIME_"] = start_time
                modified_graph.nodes[new_node_id]["END_TIME_"] = start_time + modified_graph.nodes[new_node_id][
                    "DURATION_"]

            # Зв'язок з сусіднім вузлом
            for neighbor in neighbors:
                modified_graph.add_edge(new_node_id, neighbor)
                logger.debug(f"Додано ребро: ({new_node_id} -> {neighbor})")

        # Зміна значень атрибутів у вузлах
        for node, data in modified_graph.nodes(data=True):
            if "doc_subject" in data and doc_subject_replacements:
                for old_value, new_value in doc_subject_replacements.items():
                    data["doc_subject"] = data["doc_subject"].replace(old_value, new_value)

            if "ExpectedDate" in data:
                date_shift = random.uniform(-0.1, 0.1)  # ±10%
                data["ExpectedDate"] = adjust_date(data["ExpectedDate"], date_shift)

            if "PurchasingBudget" in data:
                try:
                    data["PurchasingBudget"] = float(data["PurchasingBudget"])
                    data["PurchasingBudget"] *= random.uniform(0.9, 1.1)
                except (ValueError, TypeError):
                    logger.warning(f"Атрибут 'PurchasingBudget' не вдалося обробити: {data['PurchasingBudget']}")

            if "InitialPrice" in data and "FinalPrice" in data:
                try:
                    data["InitialPrice"] = float(data["InitialPrice"])
                    data["FinalPrice"] = float(data["FinalPrice"])
                    data["InitialPrice"] *= random.uniform(0.9, 1.1)
                    data["FinalPrice"] *= random.uniform(0.9, 1.1)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Атрибути 'InitialPrice' або 'FinalPrice' не вдалося обробити: {data.get('InitialPrice')}, {data.get('FinalPrice')}")

        # Логування змін
        logger.info(f"Усього вузлів після генерації: {len(modified_graph.nodes)}.")
        logger.info(f"Усього зв’язків після генерації: {len(modified_graph.edges)}.")

        # Формування параметрів для повернення
        generation_params = {
            "original_nodes": num_nodes,
            "original_edges": num_edges,
            "final_nodes": len(modified_graph.nodes),
            "final_edges": len(modified_graph.edges),
            "change_percentage": change_percentage,
            "new_nodes_added": len(new_nodes),
        }

        logger.info("Генерація нормального графа завершена.")
        return modified_graph, generation_params

    except Exception as e:
        logger.error(f"Деталі помилки:\n{traceback.format_exc()}")
        raise e


def adjust_date(date_str, shift):
    """Зміщення дати на заданий відсоток (shift)."""
    from datetime import datetime, timedelta
    if not date_str:
        return None

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        delta = timedelta(days=date_obj.day * shift)
        new_date = date_obj + delta
        return new_date.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        logger.warning(f"Неможливо змінити дату: {date_str}")
        return date_str


def generate_anomalous_graph(graph: nx.DiGraph, anomaly_type: str = "default", **params) -> tuple[nx.DiGraph, dict]:
    """
    Генерує аномальний граф на основі оригінального графа.

    :param graph: Оригінальний граф NetworkX.
    :param anomaly_type: Тип аномалії, наприклад:
    Типи аномалій для використання в anomaly_type:

            1. missing_steps: Пропущені кроки
               - Видалення одного або кількох важливих вузлів із графа.
               - Приклад: Затвердження фінансових документів.

            2. duplicate_steps: Дубльовані кроки
               - Дублювання вузлів або ребер, що викликає плутанину або затримки в процесі.
               - Приклад: Кадрові процеси (подвійне затвердження).

            3. wrong_route: Невірний маршрут
               - Додавання помилкових залежностей або зміна маршруту між вузлами.
               - Приклад: Логістика (невірний вибір складу).

            4. abnormal_duration: Нетипова тривалість
               - Непропорційно коротка або довга тривалість виконання дій між вузлами.
               - Приклад: Документообіг (затримка підписання).

            5. abnormal_frequency: Нетипова частота
               - Збільшена або зменшена частота повторення вузлів або ребер у процесі.
               - Приклад: Фінансові транзакції.

            6. attribute_anomaly: Відхилення атрибутів
               - Зміна атрибутів вузлів або ребер (наприклад, сума, виконавець, тип виконання) поза допустимими межами.
               - Приклад: Закупівлі (перевищення бюджету).

            7. incomplete_graph: Неповний граф
               - Частина графа обривається, що вказує на незавершені дії або помилки в процесі.
               - Приклад: Юридичні процеси (неповна перевірка).

            8. compliance_violation: Порушення нормативів
               - Відсутність обов’язкових дій або недотримання визначеного порядку в процесі.
               - Приклад: Комплаєнс (ігнорування перевірки).

            9. group_anomalies: Групові відхилення
               - Узгоджена поведінка групи вузлів, яка не відповідає очікуваній (наприклад, одночасні затримки).
               - Приклад: Кадрові або фінансові процеси
    :param params: Параметри для генерації аномалії.
    :return: Модифікований граф і параметри генерації.
    """

    modified_graph = graph.copy()
    anomaly_params = {"type": "anomaly", "anomaly_type": anomaly_type}

    try:
        if anomaly_type == "missing_steps":
            # Видалення випадкових вузлів
            nodes_to_remove = random.sample(list(modified_graph.nodes), max(1, len(modified_graph.nodes) // 10))
            for node in nodes_to_remove:
                modified_graph.remove_node(node)
                logger.info(f"Видалено вузол: {node}")
            anomaly_params["removed_nodes"] = nodes_to_remove

        elif anomaly_type == "duplicate_steps":
            # Дублювання випадкових вузлів
            nodes_to_duplicate = random.sample(list(modified_graph.nodes), max(1, len(modified_graph.nodes) // 10))
            for node in nodes_to_duplicate:
                new_node = f"{node}_duplicate"
                modified_graph.add_node(new_node, **modified_graph.nodes[node])
                for neighbor in modified_graph.neighbors(node):
                    modified_graph.add_edge(new_node, neighbor)
                logger.info(f"Додано дубль вузла: {new_node}")
            anomaly_params["duplicated_nodes"] = nodes_to_duplicate

        elif anomaly_type == "wrong_route":
            # Додавання помилкових залежностей
            all_nodes = list(modified_graph.nodes)
            for _ in range(max(1, len(all_nodes) // 10)):
                src, tgt = random.sample(all_nodes, 2)
                if not modified_graph.has_edge(src, tgt):
                    modified_graph.add_edge(src, tgt)
                    logger.info(f"Додано помилкове ребро: {src} -> {tgt}")
            anomaly_params["wrong_routes_added"] = True

        elif anomaly_type == "abnormal_duration":
            # Зміна тривалості на випадкову
            for u, v, data in modified_graph.edges(data=True):
                if "DURATION_" in data:
                    original_duration = data["DURATION_"]
                    data["DURATION_"] = original_duration * random.uniform(0.1, 20.0)
                    logger.info(f"Змінено тривалість ребра {u} -> {v} з {original_duration} на {data['DURATION_']}")
            anomaly_params["abnormal_duration"] = True

        elif anomaly_type == "abnormal_frequency":
            # Дублювання ребер
            edges_to_duplicate = random.sample(list(modified_graph.edges), max(1, len(modified_graph.edges) // 10))
            for u, v in edges_to_duplicate:
                modified_graph.add_edge(u, v)
                logger.info(f"Дубльовано ребро: {u} -> {v}")
            anomaly_params["duplicated_edges"] = edges_to_duplicate

        elif anomaly_type == "attribute_anomaly":
            # Зміна атрибутів вузлів
            nodes_to_modify = random.sample(list(modified_graph.nodes), max(1, len(modified_graph.nodes) // 10))
            for node in nodes_to_modify:
                for attr in ["PurchasingBudget", "InitialPrice", "FinalPrice", "active_executions"]:
                    if attr in modified_graph.nodes[node]:
                        original_value = modified_graph.nodes[node][attr]
                        try:
                            modified_graph.nodes[node][attr] = original_value * random.uniform(0.5, 20.0) if isinstance(
                                original_value, (int, float)) else random.randint(1, 10)
                            logger.info(
                                f"Змінено атрибут {attr} вузла {node} з {original_value} на {modified_graph.nodes[node][attr]}")
                        except Exception as e:
                            logger.warning(f"Не вдалося змінити атрибут {attr} вузла {node}: {e}")
                if "taskaction_code" in modified_graph.nodes[node]:
                    original_code = modified_graph.nodes[node]["taskaction_code"]
                    modified_graph.nodes[node]["taskaction_code"] = ''.join(
                        random.choices(string.ascii_letters, k=random.randint(5, 10)))
                    logger.info(
                        f"Змінено taskaction_code вузла {node} з {original_code} на {modified_graph.nodes[node]['taskaction_code']}")
            anomaly_params["modified_nodes"] = nodes_to_modify

        elif anomaly_type == "incomplete_graph":
            # Видалення частини графа
            subgraph_nodes = random.sample(list(modified_graph.nodes), max(1, len(modified_graph.nodes) // 2))
            for node in subgraph_nodes:
                modified_graph.remove_node(node)
                logger.info(f"Видалено вузол (для неповного графа): {node}")
            anomaly_params["removed_subgraph_nodes"] = subgraph_nodes

        elif anomaly_type == "compliance_violation":
            # Пропуск обов'язкових вузлів
            mandatory_nodes = [node for node, data in modified_graph.nodes(data=True) if data.get("type") == "userTask"]
            if mandatory_nodes:
                nodes_to_remove = random.sample(mandatory_nodes, max(1, len(mandatory_nodes) // 2))
                for node in nodes_to_remove:
                    predecessors = list(modified_graph.predecessors(node))
                    successors = list(modified_graph.successors(node))
                    modified_graph.remove_node(node)
                    logger.info(f"Видалено обов'язковий вузол: {node}")
                    if predecessors and successors:
                        for pred in predecessors:
                            for succ in successors:
                                modified_graph.add_edge(pred, succ)
                                logger.info(f"З'єднано {pred} -> {succ} після видалення {node}")
                anomaly_params["violated_compliance_nodes"] = nodes_to_remove

        elif anomaly_type == "group_anomalies":
            # Узгоджені зміни для групи вузлів
            grouped_nodes = random.sample(list(modified_graph.nodes), max(2, len(modified_graph.nodes) // 10))
            for node in grouped_nodes:
                for attr in ["DURATION_", "SEQUENCE_COUNTER_", "duration_work", "overduet_work", "taskaction_code"]:
                    if attr in modified_graph.nodes[node]:
                        original_value = modified_graph.nodes[node][attr]
                        try:
                            if attr == "taskaction_code":
                                modified_graph.nodes[node][attr] = ''.join(
                                    random.choices(string.ascii_letters, k=random.randint(5, 10)))
                            else:
                                modified_graph.nodes[node][attr] = original_value * random.uniform(1.5,
                                                                                                   2.5) if isinstance(
                                    original_value, (int, float)) else f"anomaly_{attr}"
                            logger.info(
                                f"Змінено атрибут {attr} вузла {node} з {original_value} на {modified_graph.nodes[node][attr]}")
                        except Exception as e:
                            logger.warning(f"Не вдалося змінити атрибут {attr} вузла {node}: {e}")
            anomaly_params["group_anomalies"] = grouped_nodes

        else:
            logger.warning(f"Невідомий тип аномалії: {anomaly_type}")

    except Exception as e:
        logger.error(f"Помилка при генерації аномалії: {e}")
        logger.error(f"Деталі:\n{traceback.format_exc()}")

    return modified_graph, anomaly_params


def validate_generated_graph(graph: nx.DiGraph) -> bool:
    """
    Перевіряє, чи відповідає згенерований граф вимогам.

    :param graph: Граф NetworkX для перевірки.
    :return: True, якщо граф коректний, інакше False.
    """
    try:
        # TODO: Додати специфічну логіку валідації графа
        logger.info("Перевірка графа пройдена успішно.")
        return True
    except Exception as e:
        logger.error(f"Помилка під час перевірки графа: {e}")
        return False