import pandas as pd
from src.utils.logger import get_logger
from src.utils.file_utils import read_from_parquet
from src.modules.graph_processing import build_graph_for_group
from src.config.config import GRAPH_PATH
from src.utils.file_utils import read_from_parquet, save_graph, save_graph_to_zip, save_graph_pic
from src.utils.visualizer import visualize_graph_with_dot

logger = get_logger(__name__)

def select_document_definition(doc_definitions, caption_filter=None):
    """
    Вибір дефініції документа для аналізу на основі значення caption.
    :param caption_filter: Фільтр для вибору дефініції документа.
    :return: ID обраної дефініції документа (id).
    """
    try:



        if caption_filter:
            filtered_defs = doc_definitions[doc_definitions['caption'].str.contains(caption_filter, na=False)]
        else:
            filtered_defs = doc_definitions

        if filtered_defs.empty:
            logger.warning("Не знайдено дефініцій документів, що відповідають фільтру.")
            return None

        #logger.debug(filtered_defs[['ID', 'caption']],variable_name="filtered_defs")

        selected_doc_def_id = filtered_defs.iloc[0]
        #logger.debug(selected_doc_def_id, variable_name="selected_doc_def_id")
        return selected_doc_def_id
    except Exception as e:
        logger.error(f"Помилка під час вибору дефініції документа: {e}")
        return None

def get_documents_for_definition(doc_def_id,  documents, filter_ids=None,):
    """
    Отримання списку документів на основі обраної дефініції.
    :param doc_def_id: ID дефініції документа.
    :return: DataFrame зі списком документів.
    """
    try:
        if not doc_def_id:
            logger.warning("ID дефініції документа не задано.")
            return None



        selected_documents = documents[(documents['doctype_id'] == doc_def_id) & (documents['docstate_code'] != 'new')]

        if filter_ids and isinstance(filter_ids, list):
            selected_documents = selected_documents[selected_documents['doc_id'].isin(filter_ids)]

        if selected_documents.empty:
            logger.warning(f"Не знайдено документів для дефініції ID: {doc_def_id}")
            return None

        logger.info(f"Знайдено {len(selected_documents)} документів для дефініції ID: {doc_def_id}")
        return selected_documents
    except Exception as e:
        logger.error(f"Помилка під час отримання документів для дефініції: {e}")
        return None

def get_process_instances(doc_id_list, processes):
    """
    Отримання екземплярів процесів для кожного документа.
    :param doc_id_list: Список ID документів.
    :return: Словник, де ключ — ID документа, значення — DataFrame процесів для цього документа.
    """
    try:
        if not doc_id_list:
            logger.warning("Список ID документів порожній.")
            return None


        result = {}

        for doc_id in doc_id_list:
            doc_processes = processes[processes['doc_id'] == doc_id]
            if doc_processes.empty:
                logger.warning(f"Не знайдено процесів для документа ID: {doc_id}")
            else:
                result[doc_id] = doc_processes

        logger.info(f"Знайдено процеси для {len(result)} документів.")
        #logger.debug(result, variable_name="result", max_lines=3)
        return result
    except Exception as e:
        logger.error(f"Помилка під час отримання екземплярів процесів: {e}")
        return None

def group_process_instances_by_root(process_instances,camunda_instances ):
    """
    Групування екземплярів процесів за ROOT_PROC_INST_ID_ для кожного документа.
    :param process_instances: Словник процесів для документів.
    :return: Словник, де ключ — ID документа, значення — словник груп процесів.
    """
    try:
        if not process_instances:
            logger.warning("Список процесів порожній.")
            return None

        result = {}

        for doc_id, processes in process_instances.items():
            doc_proc_ids = processes['proc_id'].tolist()
            doc_proc_external_ids = processes['proc_externalid'].tolist()
            #logger.debug(doc_proc_external_ids, variable_name="doc_proc_external_ids", max_lines=3)
            selected_instances = camunda_instances[camunda_instances['ID_'].isin(doc_proc_external_ids)]

            if selected_instances.empty:
                logger.warning(f"Не знайдено екземплярів процесів у Camunda для документа ID: {doc_id}, process ext_id:{doc_proc_external_ids[0]}, id:{doc_proc_ids[0]}")
            else:
                grouped = selected_instances.groupby('ROOT_PROC_INST_ID_')
                result[doc_id] = {root_id: group for root_id, group in grouped}
                logger.info(f"Групування процесів завершено для {len(result)} документів.")


        return result
    except Exception as e:
        logger.error(f"Помилка під час групування екземплярів процесів: {e}")
        return None

def enrich_grouped_instances_with_bpmn(grouped_instances, bpmn_df):
    """
    Доповнює групи екземплярів процесів інформацією про BPMN модель.
    :param grouped_instances: Словник {doc_id: {root_id: DataFrame процесів}}.
    :param bpmn_df: DataFrame із BPMN моделями.
    :return: Оновлений словник з доданими bpmn_model.
    """
    try:
        for doc_id, root_groups in grouped_instances.items():
            logger.debug(f"Обробка документа ID: {doc_id}")
            for root_id, group in root_groups.items():
                logger.debug(f"Обробка ROOT_PROC_INST_ID_: {root_id}")

                # Використовуємо merge для додавання bpmn_model
                group = group.merge(
                    bpmn_df[['process_definition_id', 'bpmn_model','KEY_']],
                    left_on='PROC_DEF_ID_',
                    right_on='process_definition_id',
                    how='left'
                )

                if 'bpmn_model' not in group.columns:
                    logger.warning(f"BPMN модель не знайдена для ROOT_PROC_INST_ID_: {root_id}")

                root_groups[root_id] = group  # Оновлюємо DataFrame у групі

        logger.info("Додано BPMN моделі до груп екземплярів процесів.")
        return grouped_instances

    except Exception as e:
        logger.error(f"Помилка під час доповнення груп екземплярів процесів BPMN моделями: {e}")
        return None



def analyze_documents(caption_filter=None):
    """
    Основна функція для аналізу документів.
    :param caption_filter: Фільтр для вибору дефініції документа.
    """
    logger.info("Запуск аналізу документів...")

    ###########################
    # Вибір дефініції документа
    doc_definitions = read_from_parquet("bpm_doc_def")
    doc_def = select_document_definition(doc_definitions, caption_filter)

    if doc_def is None:
        logger.warning("Аналіз перервано через відсутність обраної дефініції.")
        return

    ###########################
    # Отримання списку документів для аналізу
    docs = read_from_parquet("bpm_docs", columns=["doc_id", "doctype_id", "docstate_code"])
    documents = get_documents_for_definition(doc_def['ID'], docs, [3003643937678])

    if documents is None or documents.empty:
        logger.warning("Аналіз перервано через відсутність документів для обраної дефініції.")
        return

    ###########################
    # Отримання екземплярів процесів для документів
    processes = read_from_parquet("bpm_process")
    process_instances = get_process_instances(documents['doc_id'].tolist(), processes)

    if not process_instances:
        logger.warning("Аналіз перервано через відсутність процесів для обраних документів.")
        return

    ###########################
    # Групування екземплярів процесів за ROOT_PROC_INST_ID_ для кожного документа
    grouped_instances = {}
    camunda_instances = read_from_parquet("ACT_HI_PROCINST", columns=["ID_", "ROOT_PROC_INST_ID_", "PROC_DEF_ID_"]) #, columns=["ID_", "ROOT_PROC_INST_ID_"]
    # logger.debug(camunda_instances, variable_name="camunda_instances", max_lines=3)
    for doc_id, processes in process_instances.items():
        grouped = group_process_instances_by_root({doc_id: processes}, camunda_instances)
        if grouped and doc_id in grouped:
            grouped_instances[doc_id] = grouped[doc_id]

    ###########################
    # додавання до процесу XML BPMN
    bpmn_df = read_from_parquet("bpmn_definitions")
    grouped_instances_with_bpmn = enrich_grouped_instances_with_bpmn(grouped_instances, bpmn_df)
    #logger.debug(grouped_instances_with_bpmn, variable_name="grouped_instances_with_bpmn", max_lines=3)

    if not grouped_instances_with_bpmn:
        logger.warning("Не вдалося доповнити групи екземплярів процесів BPMN моделями.")
        return

    ###########################
    # будуємо граф для процесів, враховуючі деталі по задачам
    bpm_tasks = read_from_parquet("bpm_tasks")
    camunda_tasks = read_from_parquet("act_hi_taskinst", columns=["ID_", "TASK_DEF_KEY_"])
    camunda_actions = read_from_parquet("act_inst", columns=["ACT_ID_", "ACT_NAME_", "ACT_TYPE_", "SEQUENCE_COUNTER_", "DURATION_", "ROOT_PROC_INST_ID_", "PROC_INST_ID_"])
    #logger.debug(camunda_actions, variable_name="camunda_actions", max_lines=3)

    enriched_tasks = bpm_tasks.merge(
        camunda_tasks[['ID_', 'TASK_DEF_KEY_']],
        how='left',
        left_on='externalid',
        right_on='ID_'
    )
    grouped_graph = build_graph_for_group(grouped_instances_with_bpmn, enriched_tasks, camunda_actions)

    #logger.debug(grouped_graph, variable_name="grouped_graph", max_lines=3)

    ###########################
    # зберігаємо отримані графи
    for doc_id, root_graphs in grouped_graph.items():
        for root_proc_id, graph in root_graphs.items():
            file_name = f"{doc_id}_{root_proc_id}"
            try:
                #save_graph_to_zip(graph, file_name, GRAPH_PATH)
                save_graph(graph, file_name, GRAPH_PATH)
                save_graph_pic(graph, file_name, GRAPH_PATH, visualize_graph_with_dot)
                #logger.info(f"Граф збережено: {GRAPH_PATH}")
            except Exception as e:
                logger.error(f"Не вдалося зберегти граф {file_name}: {e}")

    if not grouped_instances:
        logger.warning("Аналіз перервано через відсутність груп екземплярів процесів.")
        return

    logger.info("Аналіз документів завершено.")



