from src.pipelines.preprocess_pipeline import run_preprocess_pipeline

from src.utils.logger import get_logger
from src.utils.file_utils import read_from_parquet, save_graphs
from src.utils.visualizer import visualize_graph

import networkx as nx
from src.modules.document_analysis import analyze_documents

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

    #analyze_documents(caption_filter="001 Запит на закупівлю")
    analyze_documents(caption_filter="066 Network scheme")

    ############################################
############################################
    # Завантаження BPMN XML
    bpmn_df = read_from_parquet("bpmn_definitions")
    #logger.debug(bpmn_df, variable_name="bpmn_df", depth=3)
    if bpmn_df.empty:
        logger.error("BPMN XML не знайдено! Завершення роботи.")
        return

    # Побудова графа процесів
   # process_graph = build_process_graph(bpmn_df)

    # Збереження графів
    #save_graphs(process_graph, GRAPH_PATH)
############################################
############################################

    # Візуалізація графа (за потреби)
  #  logger.info("Відображення графа...")
  #  visualize_graph(process_graph, "graph")
    logger.info("Експериментальний режим завершено.")
