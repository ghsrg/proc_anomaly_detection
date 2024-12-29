from src.pipelines.preprocess_pipeline import run_preprocess_pipeline

from src.utils.logger import get_logger
from src.utils.file_utils import initialize_register

from src.pipelines.doc_analysis_pipeline import analyze_documents

logger = get_logger(__name__)



def run_experimental_mode(args):
    """
    Запуск експериментального режиму.
    """
    logger.info("Запуск експериментального режиму...")
    initialize_register("graph_register", ['doc_id', 'root_proc_id', 'graph_path', 'date', 'params'])

    # Завантаження даних з системи
    if args.raw_reload:
        logger.info("Перезавантаження даних у raw.")
        run_preprocess_pipeline()

    # Аналіз документів і побудова графів з RAW файлів
    if args.doc2graph:
        logger.info("Генерація первинних графів з документів.")
        analyze_documents(caption_filter="001 Запит на закупівлю")  # можемо вказати тип документів, який потрібно аналізувати
        #analyze_documents(caption_filter="066 Network scheme")

    # Герерація варіацій для навчання
    if args.graph_synthesis:
       logger.info("Генерація додаткових графів для навчання.")
     #   generate_variations()

    ############################################
############################################
    # Завантаження BPMN XML
    #bpmn_df = read_from_parquet("bpmn_definitions")
    #logger.debug(bpmn_df, variable_name="bpmn_df", depth=3)
    #if bpmn_df.empty:
    #    logger.error("BPMN XML не знайдено! Завершення роботи.")
    #    return

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
