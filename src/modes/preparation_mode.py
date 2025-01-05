from src.pipelines.preprocess_pipeline import run_preprocess_pipeline
from src.pipelines.doc_analysis_pipeline import analyze_documents
from src.pipelines.generate_variations_pipeline import generate_variations

from src.utils.logger import get_logger
from src.utils.file_utils import initialize_register

logger = get_logger(__name__)


def run_experimental_mode(args):
    """
    Запуск попереднього режиму.
    """
    logger.info("Запуск режиму підготовки даних ...")
    initialize_register("graph_register", ['doc_id', 'root_proc_id', 'graph_path', 'date', 'doc_info'])

    # Завантаження даних з системи
    if args.raw_reload:
        logger.info("Перезавантаження даних у raw.")
        run_preprocess_pipeline()

    # Аналіз документів і побудова графів з RAW файлів
    if args.doc2graph:
        logger.info("Генерація первинних графів з документів.")
        analyze_documents(
            caption_filter="001 Запит на закупівлю")  # можемо вказати тип документів, який потрібно аналізувати
        #analyze_documents(caption_filter="066 Network scheme")

    # Герерація варіацій для навчання
    if args.graph_synthesis:
        normal_var = args.normal_var or 0
        missing_steps = args.missing_steps or 0
        duplicate_steps = args.duplicate_steps or 0
        initialize_register("normal_graphs",
                            ['id', 'doc_id', 'root_proc_id', 'graph_path', 'date', 'params', 'doc_info'])
        initialize_register("anomalous_graphs",
                            ['id', 'doc_id', 'root_proc_id', 'graph_path', 'date', 'params', 'doc_info'])
        if normal_var > 0:
            logger.info("Генерація додаткових нормальних графів для навчання.")
            generate_variations(normal_var, anomaly_type=None)  # missing_steps, duplicate_steps, wrong_route, abnormal_duration, abnormal_frequency, attribute_anomaly, incomplete_graph,
        if missing_steps > 0:
            logger.info("Генерація аномальних missing_steps графів для навчання.")
            generate_variations(missing_steps, anomaly_type='missing_steps')  # missing_steps, duplicate_steps, wrong_route, abnormal_duration, abnormal_frequency, attribute_anomaly, incomplete_graph,
        if duplicate_steps > 0:
            logger.info("Генерація аномальних duplicate_steps графів для навчання.")
            generate_variations(duplicate_steps, anomaly_type='duplicate_steps')  # missing_steps, duplicate_steps, wrong_route, abnormal_duration, abnormal_frequency, attribute_anomaly, incomplete_graph,
    else:
        logger.error("Треба задати клькість генерації для normal_var та anomaly_var.")

    logger.info("Режим підготовки даних завершено.")
