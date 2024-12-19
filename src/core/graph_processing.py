# Побудова графа для GNN
import xml.etree.ElementTree as ET
import networkx as nx


def parse_bpmn_and_find_subprocesses(bpmn_xml):
    """
    Розбір BPMN XML і пошук підпроцесів.
    :param bpmn_xml: XML BPMN схеми.
    :return: Список підпроцесів (KEY_).
    """
    tree = ET.ElementTree(ET.fromstring(bpmn_xml))
    root = tree.getroot()
    ns = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    subprocess_keys = []

    # Знаходимо всі callActivity
    for call_activity in root.findall(".//bpmn:callActivity", ns):
        subprocess_key = call_activity.attrib.get("calledElement")
        if subprocess_key:
            subprocess_keys.append(subprocess_key)

    return subprocess_keys
def build_process_graph(bpmn_df):
    """
    Побудова графа процесів на основі BPMN XML.
    :param bpmn_df: DataFrame із BPMN XML.
    :return: NetworkX граф.
    """
    graph = nx.DiGraph()

    for _, row in bpmn_df.iterrows():
        bpmn_xml = row['bpmn_model']
        subprocess_keys = parse_bpmn_and_find_subprocesses(bpmn_xml)

        # Додаємо вузли та зв’язки
        graph.add_node(row['process_key'], type="process", name=row['process_name'])
        for subprocess_key in subprocess_keys:
            graph.add_node(subprocess_key, type="subprocess")
            graph.add_edge(row['process_key'], subprocess_key)

    return graph