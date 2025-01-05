import xml.etree.ElementTree as ET
import networkx as nx
from src.utils.logger import get_logger
from src.utils.visualizer import visualize_graph_with_dot
from src.utils.graph_utils import inspect_graph
from src.utils.string_utils import date2str
import pandas as pd
import inspect
from collections import Counter
import traceback

logger = get_logger(__name__)

def parse_bpmn_and_find_elements(bpmn_model):
    """
    Парсинг BPMN-моделі й повернення словника активних елементів:
    - nodes: {node_id: {type, name, ...}}
    - edges: [{source, target, attributes}]
    """
    try:
        tree = ET.ElementTree(ET.fromstring(bpmn_model))
        root = tree.getroot()

        namespace = {
            'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        elements = {
            'nodes': {},
            'edges': []
        }

        node_tags = [
            'task', 'userTask', 'scriptTask', 'serviceTask',
            'startEvent', 'endEvent', 'intermediateThrowEvent',
            'boundaryEvent', 'exclusiveGateway', 'parallelGateway',
            'subProcess', 'callActivity', 'intermediateCatchEvent',
            'manualTask', 'businessRuleTask', 'receiveTask', 'sendTask',
            'complexGateway', 'eventBasedGateway'
        ]

        # Додавання вузлів і потоків до елементів
        for process in root.findall('.//bpmn:process', namespace):
            for element in process:
                tag = element.tag.split('}')[-1]
                node_id = element.attrib.get('id')
                if not node_id:
                    continue

                # Додаємо вузли
                if tag in node_tags:
                    node_name = element.attrib.get('name', tag)
                    node_info = {
                        'type': tag,
                        'name': node_name
                    }

                    # Обробка boundaryEvent
                    if tag == 'boundaryEvent':
                        attached_to = element.attrib.get('attachedToRef')
                        if attached_to:
                            node_info['attachedToRef'] = attached_to
                            # Додаємо зв'язок між вузлом і boundaryEvent
                            elements['edges'].append({
                                'source': attached_to,
                                'target': node_id,
                                'attributes': {
                                    'type': 'boundaryLink'
                                }
                            })
                    # Обробка callActivity
                    if tag == 'callActivity':
                        called_element = element.attrib.get('calledElement')
                        if called_element:
                            node_info['calledElement'] = called_element

                    elements['nodes'][node_id] = node_info

                # Додаємо потоки
                if tag == 'sequenceFlow':
                    source_ref = element.attrib.get('sourceRef')
                    target_ref = element.attrib.get('targetRef')
                    edge_id = element.attrib.get('id')
                    if not (source_ref and target_ref and edge_id):
                        continue

                    flow_name = element.attrib.get('name', '')
                    elements['edges'].append({
                        'source': source_ref,
                        'target': target_ref,
                        'attributes': {
                            'id': edge_id,
                            'name': flow_name
                        }
                    })

        def process_subprocess(subprocess_element, parent_id):
            """
            Рекурсивна обробка підпроцесів та їх інтеграція в глобальний граф.
            """
            subprocess_nodes = {}
            subprocess_edges = []

            # Зчитуємо вузли підпроцесу
            for element in subprocess_element:
                tag = element.tag.split('}')[-1]
                node_id = element.attrib.get('id')
                if not node_id:
                    continue

                if tag in node_tags:
                    node_name = element.attrib.get('name', tag)
                    subprocess_nodes[node_id] = {
                        'type': tag,
                        'name': node_name
                    }

            # Збираємо потоки підпроцесу
            for sequence_flow in subprocess_element.findall('.//bpmn:sequenceFlow', namespace):
                source_ref = sequence_flow.attrib.get('sourceRef')
                target_ref = sequence_flow.attrib.get('targetRef')
                edge_id = sequence_flow.attrib.get('id')
                if not (source_ref and target_ref and edge_id):
                    continue

                flow_name = sequence_flow.attrib.get('name', '')
                subprocess_edges.append({
                    'source': source_ref,
                    'target': target_ref,
                    'attributes': {
                        'id': edge_id,
                        'name': flow_name
                    }
                })

            start_nodes = [n for n in subprocess_nodes if not any(e['target'] == n for e in subprocess_edges)]
            end_nodes = [n for n in subprocess_nodes if not any(e['source'] == n for e in subprocess_edges)]

            # Замінюємо типи вузлів startEvent і endEvent для підпроцесів
            for start_node in start_nodes:
                if subprocess_nodes[start_node]['type'] == 'startEvent':
                    subprocess_nodes[start_node]['type'] = 'startEventsp'

            for end_node in end_nodes:
                if subprocess_nodes[end_node]['type'] == 'endEvent':
                    subprocess_nodes[end_node]['type'] = 'endEventsp'

            # Отримуємо <incoming> та <outgoing> вузли підпроцесу
            incoming_links = [incoming.text for incoming in subprocess_element.findall('bpmn:incoming', namespace)]
            outgoing_links = [outgoing.text for outgoing in subprocess_element.findall('bpmn:outgoing', namespace)]

            # Видаляємо вузол subProcess з основного графа
            elements['nodes'].pop(parent_id, None)

            # Оновлюємо потоки у глобальному контексті
            for edge in elements['edges']:
                if edge['attributes']['id'] in incoming_links:
                    edge['target'] = start_nodes[0] if start_nodes else edge['target']  # Замінюємо target
                if edge['attributes']['id'] in outgoing_links:
                    edge['source'] = end_nodes[0] if end_nodes else edge['source']  # Замінюємо source

            return subprocess_nodes, subprocess_edges

        # Обробка підпроцесів
        for process in root.findall('.//bpmn:process', namespace):
            for element in process:
                tag = element.tag.split('}')[-1]
                node_id = element.attrib.get('id')
                if not node_id:
                    continue

                if tag == 'subProcess':
                    subprocess_nodes, subprocess_edges = process_subprocess(element, node_id)
                    elements['nodes'].update(subprocess_nodes)
                    elements['edges'].extend(subprocess_edges)

        return elements
    except Exception as e:
        logger.error(f"Помилка парсингу BPMN: {e}")
        logger.error(f"Деталі помилки:\n{traceback.format_exc()}")
        return None


def build_graph_for_group(grouped_instances_with_bpmn, bpm_tasks, camunda_actions, bpm_doc_purch):
    """
    Побудова графів для кожної групи (ROOT_PROC_ID) із документів.
    :param grouped_instances_with_bpmn: Словник {doc_id: {root_proc_id: DataFrame}}.
    :param bpm_tasks: DataFrame з історією задач Camunda.
    :param camunda_actions: DataFrame з історією подій в процесі Camunda.
    :return: Словник {doc_id: {root_proc_id: networkx.DiGraph}}.
    """
    result_graphs = {}
    total_cycles = len(grouped_instances_with_bpmn)  # Загальна кількість циклів
    try:
        for current_cycle, (doc_id, root_groups) in enumerate(grouped_instances_with_bpmn.items(), start=1):
            progress = (current_cycle / total_cycles) * 100
            print(f"Цикл {current_cycle}/{total_cycles} - Обробляється doc_id: {doc_id} : {progress:.2f}%")

            doc_graphs = {}

            for root_proc_id, root_group in root_groups.items():
                logger.info(f"Побудова графа ROOT_PROC_ID: {root_proc_id} у документі {doc_id}")

                # Знаходимо рядок із BPMN
                root_process_row = root_group[root_group['ID_'] == root_proc_id]
               # root_process_row = root_group[root_group['ID_'] == '981d87df-5588-11ef-86d8-0242ac111804' ]
                if root_process_row.empty:
                    logger.warning(f"Немає кореневого процесу в групі: {root_proc_id}")
                    continue

                bpmn_model = root_process_row['bpmn_model'].iloc[0]
                if not bpmn_model:
                    logger.warning(f"BPMN відсутня для ROOT_PROC_ID: {root_proc_id}")
                    continue

                # Будуємо граф
                graph = build_process_graph(bpmn_model, root_proc_id, root_group, bpm_tasks, camunda_actions)

                if graph:
                    doc_graphs[root_proc_id] = graph

                # Можна візуалізувати відразу або окремо
                from src.utils.graph_utils import clean_graph
                #inspect_graph(graph)
                #visualize_graph_with_dot(graph)

            if doc_graphs:
                result_graphs[doc_id] = doc_graphs

        return result_graphs
    except Exception as e:
        logger.error(f"Помилка під час побудови графів: {e}")
        logger.error(f"Деталі помилки:\n{traceback.format_exc()}")
        return None


def build_process_graph(bpmn_model, proc_id, group, bpm_tasks, camunda_actions):
    """
    Побудова графа для одного процесу + прив'язка задач (через ACT_ID_).
    :param camunda_actions:
    :param proc_id:
    :param bpmn_model: BPMN XML (рядок).
    :param group: DataFrame з процесами (у т.ч. підпроцесами, якщо є). для рекурсивного проходу по підпроцесам
    :param bpm_tasks: DataFrame з історичними записами задач.
    :return: networkx.DiGraph
    """
    level = len(inspect.stack()) - 5  # Рівень стека
    #print(level)
    try:
        graph = nx.DiGraph()
        elements = parse_bpmn_and_find_elements(bpmn_model)
        if not elements:
            return None

        nodes = elements['nodes']
        edges = elements['edges']

        # Додаємо вузли з BPMN
        for node_id, attr in nodes.items():
            attr['active_executions'] = 0  # Додаємо атрибут кількість виконань зі значенням 0
            if node_id in graph.nodes:
                logger.warning(f"Дублювання вузла з ідентифікатором: {node_id}")
            else:
                graph.add_node(node_id, **attr)

        # Зв'язуємо boundary events та інші
        for node_id, attr in nodes.items():
            attached_to = attr.get('attachedToRef')  # Шукаємо вузол, до якого прив'язаний елемент
            if attached_to and attached_to in graph.nodes:
                # Додаємо ребро між вузлом-власником і прив'язаним елементом
                if graph.has_edge(attached_to, node_id):
                    logger.warning(f"Дублювання зв'язку між {attached_to} і {node_id}")
                else:
                    graph.add_edge(attached_to, node_id, type='attached')

        # Додаємо ребра (sequenceFlow) без оновлення вузлів
        for edge in edges:
            source = edge['source']
            target = edge['target']
            attrs = edge.get('attributes', {})
            graph.add_edge(source, target, **attrs)

            # Мапимо дії з Camunda, щоб визначити що виконувалось, та як
        filtered_camunda_actions = camunda_actions[camunda_actions['PROC_INST_ID_'] == proc_id]
        camunda_mapping = filtered_camunda_actions.set_index('ACT_ID_')
        for node_id in list(graph.nodes):

            if node_id in camunda_mapping.index:
                camunda_row = camunda_mapping.loc[node_id]

                if isinstance(camunda_row, pd.DataFrame):
                    # Групування по SEQUENCE_COUNTER_
                    # logger.debug(camunda_row, variable_name=f"{level} camunda_row", max_lines=10)
                    grouped = camunda_row.groupby('SEQUENCE_COUNTER_').agg({
                        'DURATION_': 'max',
                        'SEQUENCE_COUNTER_': 'max',
                        'TASK_ID_': lambda x: x.iloc[0] if x.notna().any() else ''
                        # Беремо випадковий TASK_ID_ задачі, що мала декілька виконань. Тут втрачаємо деталі по повторних задачах
                    })
                    #logger.debug(grouped, variable_name=f"{level} grouped", max_lines=10)
                    # Логіка для технічних вузлів (без TASK_ID_)
                    if grouped['TASK_ID_'].isnull().all() or (grouped['TASK_ID_'] == '').all():
                        # Якщо TASK_ID_ відсутній, то це не користувацькі задачі - всі записи технічні, використовуємо MAX
                        group_row = grouped.iloc[0]
                        node_params = {
                            # 'type':  camunda_row.get('ACT_TYPE_'),
                            'DURATION_': group_row.get('DURATION_', '').max(),
                            'START_TIME_': date2str(group_row.get('START_TIME_', '')),
                            'END_TIME_': date2str(group_row.get('END_TIME_', '')),
                            'SEQUENCE_COUNTER_': group_row.get('SEQUENCE_COUNTER_', '').max(),
                            'active_executions': len(grouped)  # Кількість виконань
                        }
                        logger.debug(node_params, variable_name=f"{level} технічних node_params", depth=10, max_lines=30)
                        graph.nodes[node_id].update(node_params)
                    else:
                        # Логіка для користувацьких вузлів із TASK_ID_
                        # Оновлюємо перший вузол
                        first_row = grouped.iloc[0]
                        node_params = {
                            'DURATION_': first_row.get('DURATION_', ''),
                            'START_TIME_': date2str(first_row.get('START_TIME_', '')),
                            'END_TIME_': date2str(first_row.get('END_TIME_', '')),
                            'SEQUENCE_COUNTER_': first_row.get('SEQUENCE_COUNTER_', ''),
                            'TASK_ID_': first_row.get('TASK_ID_', ''),
                            'active_executions': 1
                            }
                        logger.debug(node_params, variable_name=f"{level} користувацьких node_params", depth=10, max_lines=30)
                        graph.nodes[node_id].update(node_params)
                        # Додаємо решту вузлів
                        for _, group_row in grouped.iloc[1:].iterrows():
                            task_id = group_row['TASK_ID_']
                            if pd.notna(task_id):
                                # Створюємо унікальний ID для нового вузла
                                new_node_id = f"{node_id}_{task_id}"

                                # Копіюємо вузол із новим ID
                                graph.add_node(new_node_id, **graph.nodes[node_id])

                                # Оновлюємо параметри вузла
                                node_params = {
                                    'DURATION_': group_row.get('DURATION_', ''),
                                    'START_TIME_': date2str(group_row.get('START_TIME_', '')),
                                    'END_TIME_': date2str(group_row.get('END_TIME_', '')),
                                    'SEQUENCE_COUNTER_': group_row.get('SEQUENCE_COUNTER_', ''),
                                    'TASK_ID_': task_id,
                                    'active_executions': 1
                                }
                                logger.debug(node_params, variable_name=f"{level} решта  node_params",
                                             depth=10, max_lines=30)
                                graph.nodes[new_node_id].update(node_params)

                                # Копіюємо вхідні зв'язки
                                for predecessor in graph.predecessors(node_id):
                                    if not graph.has_edge(predecessor, new_node_id):
                                        graph.add_edge(predecessor, new_node_id, **graph.edges[predecessor, node_id])

                                # Копіюємо вихідні зв'язки
                                for successor in graph.successors(node_id):
                                    if not graph.has_edge(new_node_id, successor):
                                        graph.add_edge(new_node_id, successor, **graph.edges[node_id, successor])
                else:
                    # Якщо лише один запис, оновлюємо вузол напряму
                    node_params = {
                        key: value
                        for key, value in {
                            # 'type':  camunda_row.get('ACT_TYPE_'),
                            'DURATION_': camunda_row.get('DURATION_'),
                            'START_TIME_': date2str(camunda_row.get('START_TIME_', '')),
                            'END_TIME_': date2str(camunda_row.get('END_TIME_', '')),
                            'SEQUENCE_COUNTER_': camunda_row.get('SEQUENCE_COUNTER_'),
                            'TASK_ID_': camunda_row.get('TASK_ID_'),
                            'active_executions': 1  # Це статичне значення завжди додається
                        }.items()
                        if pd.notna(value)  # Додаємо тільки значення, які існують
                    }
                    logger.debug(node_params, variable_name=f"{level} node_NO_GROUP_params", depth=10, max_lines=30)
                    graph.nodes[node_id].update(node_params)

        # Мапимо задачі з bpm_tasks, щоб наповнити бізнес атрибутами

        tasks_mapping = bpm_tasks.set_index('externalid')
        for node_id, node_data in graph.nodes(data=True):
            task_id = node_data.get('TASK_ID_')  # Отримуємо TASK_ID_ з атрибутів вузла
            if task_id and task_id in tasks_mapping.index:  # Перевіряємо наявність TASK_ID_ у tasks_mapping
                task_row = tasks_mapping.loc[task_id]
                task_params = {
                    key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
                    for key, value in {
                        'task_status': task_row.get('task_status'),
                        'taskaction_code': task_row.get('taskaction_code'),
                        'user_compl_login': task_row.get('user_compl_login'),
                        'user_compl_position': task_row.get('user_compl_position'),
                        'first_view': task_row.get('first_view_compluser_date'),
                        'overdue_work': task_row.get('overduet_work'),
                        'duration_work': task_row.get('durationt_work')
                    }.items()
                    if pd.notna(value)  # Додаємо тільки значення, які існують
                }
                # logger.debug(task_params, variable_name=f"{level} task_params", max_lines=10)
                graph.nodes[node_id].update(task_params)

        # Оновлюємо атрибути ребер на основі вузлів
        for edge in edges:
            source = edge['source']
            target = edge['target']

            if source in graph.nodes and target in graph.nodes:
                source_node = graph.nodes[source]
                target_node = graph.nodes[target]

                # Оновлюємо атрибути ребра
                graph.edges[source, target].update({
                    'DURATION_E': source_node.get('DURATION_', ''),
                    'taskaction_code_E': source_node.get('taskaction_code', ''),
                    'duration_work_E': target_node.get('duration_work', '')
                })

        # Обробка callActivity
        for node_id, attr in nodes.items():
            if attr.get('type') == 'callActivity' and 'calledElement' in attr:
                extprocess_key = attr['calledElement']
                if 'KEY_' not in group.columns:
                    logger.debug(
                        f"Колонка 'KEY_' відсутня у групі для зовнішнього процесу з ключем {extprocess_key}. Процес буде пропущений."
                    )
                    continue

                matching_extprocess = group[group['KEY_'] == extprocess_key]
                if matching_extprocess.empty:
                    logger.debug(f"Не знайдено зовнішнього процесу з ключем {extprocess_key}. Процес буде пропущений.")
                    continue

                extprocess_row = matching_extprocess.iloc[0]
                extprocess_bpmn_model = extprocess_row.get('bpmn_model')

                if not extprocess_bpmn_model:
                    logger.warning(f"BPMN модель відсутня для зовнішнього процесу з ключем {extprocess_key}.")
                    continue

                extprocess_graph = build_process_graph(
                    extprocess_bpmn_model,
                    extprocess_row['ID_'],
                    group,
                    bpm_tasks,
                    camunda_actions
                )

                if extprocess_graph:
                    # Перевіряємо і додаємо унікальні ID до вузлів підграфа
                    mapping = {
                        n: f"{n}_ext" if not n.endswith("_ext") else n
                        for n in extprocess_graph.nodes()
                    }
                    extprocess_graph = nx.relabel_nodes(extprocess_graph, mapping)

                    start_nodes = [
                        n for n, d in extprocess_graph.in_degree()
                        if d == 0 and extprocess_graph.nodes[n].get('type') == 'startEvent'
                    ]
                    end_nodes = [
                        n for n, d in extprocess_graph.out_degree()
                        if d == 0 and extprocess_graph.nodes[n].get('type') == 'endEvent'
                    ]

                    for start_node in start_nodes:
                        if extprocess_graph.nodes[start_node].get('type') == 'startEvent':
                            extprocess_graph.nodes[start_node]['type'] = 'startEventsp'
                    for end_node in end_nodes:
                        if extprocess_graph.nodes[end_node].get('type') == 'endEvent':
                            extprocess_graph.nodes[end_node]['type'] = 'endEventsp'

                    predecessors = list(graph.predecessors(node_id))
                    successors = list(graph.successors(node_id))

                    graph.remove_node(node_id)

                    graph.update(extprocess_graph)

                    # Уникаємо дублювання зв'язків
                    for pred in predecessors:
                        for start_node in start_nodes:
                            if not graph.has_edge(pred, start_node):
                                graph.add_edge(pred, start_node)

                    for succ in successors:
                        for end_node in end_nodes:
                            if not graph.has_edge(end_node, succ):
                                graph.add_edge(end_node, succ)

                else:
                    logger.warning(f"Не вдалося побудувати граф для зовнішнього процесу {extprocess_key}.")

        #visualize_graph_with_dot(graph)
        logger.info(f"Граф для {proc_id} успішно побудований.")
        return graph
    except Exception as e:
        logger.error(f"Помилка побудови графа: {e}")
        logger.error(f"Деталі помилки:\n{traceback.format_exc()}")
        return None



   #visualize_graph_with_dot(graph)
  # Замінюємо тип стартових подій підпроцесу
                   #for start_node in start_nodes:
                   #    if subprocess_graph.nodes[start_node].get('type') == 'startEvent':
                   #        subprocess_graph.nodes[start_node]['type'] = 'startEventsp'
                   #for end_node in end_nodes:
                   #    if subprocess_graph.nodes[end_node].get('type') == 'endEvent':
                   #        subprocess_graph.nodes[end_node]['type'] = 'endEventsp'
                   #