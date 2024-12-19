from src.pipelines.preprocess_pipeline import run_preprocess_pipeline
from src.core.graph_processing import build_process_graph
from src.utils.logger import get_logger
from src.utils.file_utils import load_raw_data, save_graph
import networkx as nx
import os

logger = get_logger(__name__)

GRAPH_SAVE_PATH = "data/processed/graphs"
os.makedirs(GRAPH_SAVE_PATH, exist_ok=True)  # Переконуємося, що папка існує

def save_graphs(process_graph):
    """
    Зберігає графи у вигляді файлів для кожного процесу.
    """
    for node, attrs in process_graph.nodes(data=True):
        if attrs.get("type") == "process":
            graph_file = os.path.join(GRAPH_SAVE_PATH, f"{node}.graphml")
            subgraph = nx.ego_graph(process_graph, node)
            nx.write_graphml(subgraph, graph_file)
            logger.info(f"Граф процесу {node} збережено у {graph_file}.")


def run_experimental_mode(reload):
    """
    Запуск експериментального режиму.
    """
    logger.info("Запуск експериментального режиму...")

    # Попередня обробка
    if reload:
        logger.info("Оновлення даних...")
        run_preprocess_pipeline()

    # Завантаження BPMN XML
    bpmn_df = load_raw_data("bpmn_definitions")
    if bpmn_df.empty:
        logger.error("BPMN XML не знайдено! Завершення роботи.")
        return

    # Побудова графа процесів
    process_graph = build_process_graph(bpmn_df)

    # Збереження графів
    save_graphs(process_graph)

    # Візуалізація графа (за потреби)
    logger.info("Відображення прикладу графа...")
    example_process = next(iter(process_graph.nodes(data=True)))
    nx.draw(
        process_graph,
        with_labels=True,
        labels=nx.get_node_attributes(process_graph, 'name'),
        node_color="lightblue"
    )
    logger.info("Експериментальний режим завершено.")
