import xml.etree.ElementTree as ET
import networkx as nx
from src.utils.logger import get_logger
from src.utils.visualizer import visualize_graph, visualize_graph_with_dot
import pandas as pd

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

        # Теги активних елементів BPMN
        node_tags = [
            'task', 'userTask', 'scriptTask', 'serviceTask',
            'startEvent', 'endEvent', 'intermediateThrowEvent',
            'boundaryEvent', 'exclusiveGateway', 'parallelGateway',
            'subProcess', 'callActivity'
        ]

        for process in root.findall('.//bpmn:process', namespace):
            for element in process:
                tag = element.tag.split('}')[-1]  # Назва без простору імен
                node_id = element.attrib.get('id')
                if not node_id:
                    continue  # Пропускаємо елементи без ідентифікатора

                if tag in node_tags:
                    node_name = element.attrib.get('name', tag)
                    node_type = tag
                    node_info = {
                        'type': node_type,
                        'name': node_name
                    }

                    # Специфічні атрибути для callActivity
                    if tag == 'callActivity':
                        node_info['calledElement'] = element.attrib.get('calledElement', '')

                    # Специфічні атрибути для boundaryEvent
                    if tag == 'boundaryEvent':
                        node_info['attachedToRef'] = element.attrib.get('attachedToRef', '')

                    elements['nodes'][node_id] = node_info

        # Збираємо sequenceFlow
        for sequence_flow in root.findall('.//bpmn:sequenceFlow', namespace):
            source_ref = sequence_flow.attrib.get('sourceRef')
            target_ref = sequence_flow.attrib.get('targetRef')
            edge_id = sequence_flow.attrib.get('id')
            if not (source_ref and target_ref and edge_id):
                continue

            # Читаємо назву переходу (може бути label на стрілці) і conditionExpression
            flow_name = sequence_flow.attrib.get('name', '')
            condition_expr = ''
            cond_elem = sequence_flow.find('.//bpmn:conditionExpression', namespace)
            if cond_elem is not None and cond_elem.text:
                condition_expr = cond_elem.text.strip()

            elements['edges'].append({
                'source': source_ref,
                'target': target_ref,
                'attributes': {
                    'id': edge_id,
                    'name': flow_name,
                    'conditionExpression': condition_expr
                }
            })

        return elements
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Помилка парсингу BPMN: {e}")
        return None


def build_graph_for_group(grouped_instances_with_bpmn, bpm_tasks, camunda_actions):
    """
    Побудова графів для кожної групи (ROOT_PROC_ID) із документів.
    :param grouped_instances_with_bpmn: Словник {doc_id: {root_proc_id: DataFrame}}.
    :param bpm_tasks: DataFrame з історією задач Camunda.
    :param camunda_actions: DataFrame з історією подій в процесі Camunda.
    :return: Словник {doc_id: {root_proc_id: networkx.DiGraph}}.
    """
    result_graphs = {}

    try:
        for doc_id, root_groups in grouped_instances_with_bpmn.items():
            doc_graphs = {}

            for root_proc_id, root_group in root_groups.items():
                logger.info(f"Побудова графа для ROOT_PROC_ID: {root_proc_id} у документі {doc_id}")

                # Знаходимо рядок із BPMN
                root_process_row = root_group[root_group['ID_'] == root_proc_id]
                if root_process_row.empty:
                    logger.warning(f"Нема кореневого процесу в групі: {root_proc_id}")
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
                visualize_graph_with_dot(graph)

            if doc_graphs:
                result_graphs[doc_id] = doc_graphs

        return result_graphs
    except Exception as e:
        logger.error(f"Помилка під час побудови графів: {e}")
        return None


def build_process_graph(bpmn_model, proc_id, group, bpm_tasks, camunda_actions):
    """
    Побудова графа для одного процесу + прив'язка задач (через ACT_ID_).
    :param bpmn_model: BPMN XML (рядок).
    :param group: DataFrame з процесами (у т.ч. підпроцесами, якщо є). для рекурсивного проходу по підпроцесам
    :param bpm_tasks: DataFrame з історичними записами задач.
    :return: networkx.DiGraph
    """
    logger = get_logger(__name__)

    try:
        graph = nx.DiGraph()
        elements = parse_bpmn_and_find_elements(bpmn_model)
        if not elements:
            return None
        #logger.debug(elements, variable_name="elements", max_lines=3)
        nodes = elements['nodes']
        edges = elements['edges']

        # Додаємо вузли з BPMN
        for node_id, attr in nodes.items():
            graph.add_node(node_id, **attr)


        # Додаємо прив'язані елементи (boundary events та інші)
        for node_id, attr in nodes.items():
            attached_to = attr.get('attachedToRef')  # Шукаємо вузол, до якого прив'язаний елемент
            if attached_to and attached_to in graph.nodes:
                # Додаємо ребро між вузлом-власником і прив'язаним елементом
                graph.add_edge(attached_to, node_id, type='attached')

        # Мапимо дії з Camunda
        filtered_camunda_actions = camunda_actions[camunda_actions['PROC_INST_ID_'] == proc_id]
        camunda_mapping = filtered_camunda_actions.set_index('ACT_ID_')
        for node_id in graph.nodes:
            if node_id in camunda_mapping.index:
                camunda_row = camunda_mapping.loc[node_id]

                if isinstance(camunda_row, pd.DataFrame):
                    graph.nodes[node_id].update({
                        'ACT_TYPE_': camunda_row.get('ACT_TYPE_', '').iloc[0],
                        'SEQUENCE_COUNTER_': camunda_row.get('SEQUENCE_COUNTER_', '').max(),
                        'DURATION_': camunda_row.get('DURATION_', '').max()
                    })
                else:
                    graph.nodes[node_id].update({
                        'ACT_TYPE_': camunda_row.get('ACT_TYPE_', ''),
                        'SEQUENCE_COUNTER_': camunda_row.get('SEQUENCE_COUNTER_', ''),
                        'DURATION_': camunda_row.get('DURATION_', '')
                    })

            # Мапимо задачі з bpm_tasks
        tasks_mapping = bpm_tasks.set_index('TASK_DEF_KEY_')
        for node_id in graph.nodes:
            if node_id in tasks_mapping.index:
                task_row = tasks_mapping.loc[node_id]
                #logger.debug(task_row, variable_name="task_row", max_lines=3)
         #       graph.nodes[node_id].update({
         #           'task_subject': task_row.get('task_subject', ''),
        #            'user_compl_login': task_row.get('user_compl_login', ''),
         #           'taskaction_code': task_row.get('taskaction_code', ''),
          #          'duration_work': task_row.get('duration_work', '')
         #       })

        # Додаємо ребра (sequenceFlow) з атрибутами
        for edge in edges:
            source = edge['source']
            target = edge['target']
            attrs = edge.get('attributes', {})

            # Додаємо ACT_TYPE_ і duration_work до ребер
            source_node = graph.nodes.get(source, {})
            target_node = graph.nodes.get(target, {})
            attrs.update({
                'DURATION_': source_node.get('DURATION_', ''),
                'duration_work': target_node.get('duration_work', '')
            })
            graph.add_edge(source, target, **attrs)

        # Додаємо підпроцеси
        for node_id, attr in nodes.items():
            if attr.get('type') == 'callActivity' and 'calledElement' in attr:
                subprocess_key = attr['calledElement']
                #logger.debug(subprocess_key, variable_name="subprocess_key")
                #logger.debug(group, variable_name="group")

                # Перевірка наявності колонки 'KEY_'
                if 'KEY_' not in group.columns:
                    logger.warning(
                        f"Колонка 'KEY_' відсутня у групі для підпроцесу з ключем {subprocess_key}. Підпроцес буде пропущений."
                    )
                    continue

                matching_subprocess = group[group['KEY_'] == subprocess_key]
                if matching_subprocess.empty:
                    logger.warning(f"Не знайдено підпроцес з ключем {subprocess_key}. Підпроцес буде пропущений.")
                    continue

                subprocess_row = matching_subprocess.iloc[0]
                subprocess_bpmn_model = subprocess_row.get('bpmn_model')

                if not subprocess_bpmn_model:
                    logger.warning(f"BPMN модель відсутня для підпроцесу з ключем {subprocess_key}.")
                    continue

                subprocess_graph = build_process_graph(
                    subprocess_bpmn_model,
                    subprocess_row['ID_'],
                    group,
                    bpm_tasks,
                    camunda_actions
                )

                if subprocess_graph:
                    # Знаходимо вхідні та вихідні вузли в підпроцесі
                    start_nodes = [n for n, d in subprocess_graph.in_degree() if d == 0]
                    end_nodes = [n for n, d in subprocess_graph.out_degree() if d == 0]

                    # Замінюємо тип стартових подій підпроцесу
                    for start_node in start_nodes:
                        if subprocess_graph.nodes[start_node].get('type') == 'startEvent':
                            subprocess_graph.nodes[start_node]['type'] = 'startEventsp'
                    for end_node in end_nodes:
                        if subprocess_graph.nodes[end_node].get('type') == 'endEvent':
                            subprocess_graph.nodes[end_node]['type'] = 'endEventsp'

                    # Отримуємо попередники та наступники вузла callActivity
                    predecessors = list(graph.predecessors(node_id))
                    successors = list(graph.successors(node_id))

                    # Видаляємо вузол callActivity
                    graph.remove_node(node_id)

                    # Додаємо вузли та ребра підпроцесу до головного графа
                    graph = nx.compose(graph, subprocess_graph)

                    # З'єднуємо вхідні вузли підпроцесу з попередниками callActivity
                    for pred in predecessors:
                        for start_node in start_nodes:
                            graph.add_edge(pred, start_node)

                    # З'єднуємо вихідні вузли підпроцесу з наступниками callActivity
                    for succ in successors:
                        for end_node in end_nodes:
                            graph.add_edge(end_node, succ)

                else:
                    logger.warning(f"Не вдалося побудувати граф для підпроцесу з ключем {subprocess_key}.")

        logger.info("Граф успішно побудований.")
        return graph
    except Exception as e:
        logger.error(f"Помилка побудови графа: {e}")
        return None