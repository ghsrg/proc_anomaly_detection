import pandas as pd
from src.utils.logger import get_logger
from src.utils.file_utils import read_from_parquet

logger = get_logger(__name__)


def select_document_definition(caption_filter=None):
    """
    Вибір дефініції документа для аналізу на основі значення caption.
    :param caption_filter: Фільтр для вибору дефініції документа.
    :return: ID обраної дефініції документа (id).
    """
    try:
        #doc_def_path = "data/processed/bpm_doc_def.parquet"
        doc_definitions = read_from_parquet("bpm_doc_def")

        if caption_filter:
            filtered_defs = doc_definitions[doc_definitions['caption'].str.contains(caption_filter, na=False)]
        else:
            filtered_defs = doc_definitions

        if filtered_defs.empty:
            logger.warning("Не знайдено дефініцій документів, що відповідають фільтру.")
            return None

        logger.debug(filtered_defs[['ID', 'caption']],variable_name="filtered_defs")

        selected_doc_def_id = filtered_defs.iloc[0]
        logger.debug(selected_doc_def_id, variable_name="selected_doc_def_id")
        return selected_doc_def_id
    except Exception as e:
        logger.error(f"Помилка під час вибору дефініції документа: {e}")
        return None


def get_documents_for_definition(doc_def_id):
    """
    Отримання списку документів на основі обраної дефініції.
    :param doc_def_id: ID дефініції документа.
    :return: DataFrame зі списком документів.
    """
    try:
        if not doc_def_id:
            logger.warning("ID дефініції документа не задано.")
            return None

        documents = read_from_parquet("bpm_docs")

        selected_documents = documents[documents['doctype_id'] == doc_def_id]

        if selected_documents.empty:
            logger.warning(f"Не знайдено документів для дефініції ID: {doc_def_id}")
            return None

        logger.info(f"Знайдено {len(selected_documents)} документів для дефініції ID: {doc_def_id}")
        return selected_documents
    except Exception as e:
        logger.error(f"Помилка під час отримання документів для дефініції: {e}")
        return None

def get_process_instances(doc_id_list):
    """
    Отримання екземплярів процесів для кожного документа.
    :param doc_id_list: Список ID документів.
    :return: Словник, де ключ — ID документа, значення — DataFrame процесів для цього документа.
    """
    try:
        if not doc_id_list:
            logger.warning("Список ID документів порожній.")
            return None

        processes = read_from_parquet("bpm_process")
        result = {}

        for doc_id in doc_id_list:
            doc_processes = processes[processes['doc_id'] == doc_id]
            if doc_processes.empty:
                logger.warning(f"Не знайдено процесів для документа ID: {doc_id}")
            else:
                result[doc_id] = doc_processes

        logger.info(f"Знайдено процеси для {len(result)} документів.")
        logger.debug(result, variable_name="result")
        return result
    except Exception as e:
        logger.error(f"Помилка під час отримання екземплярів процесів: {e}")
        return None

def group_process_instances_by_root(process_instances):
    """
    Групування екземплярів процесів за ROOT_PROC_INST_ID_ для кожного документа.
    :param process_instances: Словник процесів для документів.
    :return: Словник, де ключ — ID документа, значення — словник груп процесів.
    """
    try:
        if not process_instances:
            logger.warning("Список процесів порожній.")
            return None

        camunda_instances = read_from_parquet("ACT_HI_PROCINST")
        result = {}

        for doc_id, processes in process_instances.items():
            doc_proc_external_ids = processes['proc_externalid'].tolist()
            selected_instances = camunda_instances[camunda_instances['ID_'].isin(doc_proc_external_ids)]

            if selected_instances.empty:
                logger.warning(f"Не знайдено екземплярів процесів у Camunda для документа ID: {doc_id}")
            else:
                grouped = selected_instances.groupby('ROOT_PROC_INST_ID_')
                result[doc_id] = {root_id: group for root_id, group in grouped}

        logger.info(f"Групування процесів завершено для {len(result)} документів.")
        return result
    except Exception as e:
        logger.error(f"Помилка під час групування екземплярів процесів: {e}")
        return None


def analyze_documents(caption_filter=None):
    """
    Основна функція для аналізу документів.
    :param caption_filter: Фільтр для вибору дефініції документа.
    """
    logger.info("Запуск аналізу документів...")

    # Вибір дефініції документа
    doc_def = select_document_definition(caption_filter)

    if doc_def is None:
        logger.warning("Аналіз перервано через відсутність обраної дефініції.")
        return

    # Отримання списку документів для аналізу
    documents = get_documents_for_definition(doc_def['ID'])

    if documents is None or documents.empty:
        logger.warning("Аналіз перервано через відсутність документів для обраної дефініції.")
        return

        # Отримання екземплярів процесів для документів
    process_instances = get_process_instances(documents['doc_id'].tolist())

    if process_instances is None or process_instances.empty:
        logger.warning("Аналіз перервано через відсутність процесів для обраних документів.")
        return

    # Групування екземплярів процесів за ROOT_PROC_INST_ID_
    grouped_instances = group_process_instances_by_root(process_instances['proc_externalid'].tolist())

    if grouped_instances is None:
        logger.warning("Аналіз перервано через відсутність груп екземплярів процесів.")
        return

    logger.info("Аналіз документів завершено.")

# Приклад використання:
if __name__ == "__main__":
    analyze_documents(caption_filter="Purchase")
