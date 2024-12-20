from src.pipelines.preprocess_pipeline import run_preprocess_pipeline
from src.core.graph_processing import build_process_graph
from src.utils.logger import get_logger
from src.utils.file_utils import load_raw_data, save_graphs
from src.utils.visualizer import visualize_graph
from src.config.config import GRAPH_PATH
import networkx as nx

logger = get_logger(__name__)



def run_experimental_mode(reload):
    """
    Запуск експериментального режиму.
    """
    logger.info("Запуск експериментального режиму...")

    # Попередня обробка
    if reload:
        logger.info("Оновлення даних...")
        run_preprocess_pipeline()
############################################
######## перенести в run_preprocess_pipeline
############################################
    # Завантаження BPMN XML
    bpmn_df = load_raw_data("bpmn_definitions")
    #logger.debug(bpmn_df, variable_name="bpmn_df", depth=3)
    if bpmn_df.empty:
        logger.error("BPMN XML не знайдено! Завершення роботи.")
        return

    # Побудова графа процесів
    process_graph = build_process_graph(bpmn_df)

    # Збереження графів
    #save_graphs(process_graph, GRAPH_PATH)
############################################
######## перенести в run_preprocess_pipeline
############################################

    # Візуалізація графа (за потреби)
    logger.info("Відображення графа...")
    visualize_graph(process_graph, "graph")
    logger.info("Експериментальний режим завершено.")
