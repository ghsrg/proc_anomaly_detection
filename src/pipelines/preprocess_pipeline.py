import os
from src.data_sources.mssql_connector import execute_query
from src.utils.logger import get_logger
from src.utils.file_utils import save_to_parquet
from src.config.secrets import BPMS_CONFIG, CAMUNDA_CONFIG

logger = get_logger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)  # Переконуємося, що папка існує


def load_execution_and_tasks():
    """
    Завантажує таблиці ACT_HI_ACTINST, ACT_HI_TASKINST для зв'язків між задачами та активностями з CAMUNDA.
    """
    logger.info("Завантаження даних з Camunda про виконання процесів та задач.")

    # Завантаження ACT_HI_ACTINST
    actinst_query = """
    SELECT 
        ID_ AS activity_id,
        PARENT_ACT_INST_ID_ AS parent_activity_id,
        PROC_DEF_ID_ AS process_definition_id,
        ROOT_PROC_INST_ID_ AS root_process_instance_id,
        PROC_INST_ID_ AS process_instance_id,
        ACT_ID_ AS activity_definition_id,
        ACT_NAME_ AS activity_name,
        ACT_TYPE_ AS activity_type,
        START_TIME_ AS start_time,
        END_TIME_ AS end_time,
        DURATION_ AS duration
    FROM 
        ACT_HI_ACTINST;
    """
    actinst_df = execute_query(actinst_query, CAMUNDA_CONFIG)
    save_to_parquet(actinst_df, "act_hi_actinst")

    # Завантаження ACT_HI_TASKINST
    taskinst_query = """
           SELECT 
            act.ID_ AS activity_id,
            act.ACT_NAME_ AS activity_name,
            act.ACT_TYPE_ AS activity_type,
            act.PARENT_ACT_INST_ID_ AS parent_activity_id,
            act.PROC_INST_ID_ AS process_instance_id,
            task.ID_ AS task_id,
            task.NAME_ AS task_name,
            task.ASSIGNEE_ AS task_assignee,
            act.SEQUENCE_COUNTER_ AS sequence_counter,
            act.START_TIME_ AS activity_start_time,
            act.END_TIME_ AS activity_end_time,
            task.START_TIME_ AS task_start_time,
            task.END_TIME_ AS task_end_time
        FROM 
            ACT_HI_ACTINST act
        LEFT JOIN 
            ACT_HI_TASKINST task
        ON 
            act.ID_ = task.ACT_INST_ID_;
    """
    taskinst_df = execute_query(taskinst_query, CAMUNDA_CONFIG)
    save_to_parquet(taskinst_df, "act_hi_taskinst")

    # Завантаження BPMN XML
    bpmn_query = """
       SELECT 
           proc_def.ID_ AS process_definition_id,
           proc_def.KEY_ AS process_key,
           proc_def.NAME_ AS process_name,
           proc_def.DEPLOYMENT_ID_ AS deployment_id,
           bytearray.BYTES_ AS bpmn_model
       FROM 
           ACT_RE_PROCDEF proc_def
       JOIN 
           ACT_GE_BYTEARRAY bytearray
       ON 
           proc_def.DEPLOYMENT_ID_ = bytearray.DEPLOYMENT_ID_;
       """
    bpmn_df = execute_query(bpmn_query)
    save_to_parquet(bpmn_df, "bpmn_definitions")

def load_camunda_data():
    """
    Завантажує BPMN XML і таблиці ks_dwh_bpm_proc_def для процесів з CAMUNDA та BPMS.
    """
    logger.info("Завантаження BPMN XML і таблиць ks_dwh_bpm.")

       # Завантаження BPMN XML з CAMUNDA
    bpmn_query = """
    SELECT 
        proc_def.ID_ AS process_definition_id,
        proc_def.KEY_ AS process_key,
        proc_def.NAME_ AS process_name,
        proc_def.DEPLOYMENT_ID_ AS deployment_id,
        bytearray.BYTES_ AS bpmn_model
    FROM 
        ACT_RE_PROCDEF proc_def
    JOIN 
        ACT_GE_BYTEARRAY bytearray
    ON 
        proc_def.DEPLOYMENT_ID_ = bytearray.DEPLOYMENT_ID_;
    """
    bpmn_df = execute_query(bpmn_query, CAMUNDA_CONFIG)
    save_to_parquet(bpmn_df, "bpmn_definitions")

def load_bpms_data():
    """
    Завантажує всі необхідні дані для побудови графів та подальшої аналітики.
    """
    logger.info("Початок завантаження та обробки BPMS даних.")

    # SQL-запити для таблиць ks_dwh_bpm
    queries = {
        "ks_dwh_bpm_docs": """
            SELECT 
                doc_id,
                doc_number,
                doctype_code,
                doctype_name,
                doctype_caption,
                doc_regnumber,
                CAST(doc_regdate AS DATETIME) AS doc_regdate,
                doc_subject,
                docstate_id,
                docstate_name,
                docstate_code,
                docfld_id,
                docfld_par_id,
                docfldr_code,
                CAST(doc_createdate AS DATETIME) AS doc_createdate,
                CAST(doc_modifyDate AS DATETIME) AS doc_modifyDate,
                user_create_login,
                staff_externalid,
                CAST(doc_deletedate AS DATETIME) AS doc_deletedate,
                doc_par_id,
                doc_labels
            FROM dbo.ks_dwh_bpm_docs;
        """,
        "bpm_tasks": "SELECT * FROM dbo.ks_dwh_bpm_tasks;",
        "purchase_doc": "SELECT * FROM dbo.ks_dwh_purchase_cxo_rep;",
        "proc_def": "SELECT * FROM dbo.ks_dwh_bpm_proc_def;"
    }

    # Виконання запитів і збереження у Parquet
    for filename, query in queries.items():
        logger.info(f"Зберігання файлу: {filename}")
        df = execute_query(query, BPMS_CONFIG)
        save_to_parquet(df, filename)

    # Завантаження виконання процесів і задач
    load_execution_and_tasks()

    logger.info("Завантаження та обробка даних завершені.")

def run_preprocess_pipeline():
    """
    Основний запуск конвеєра обробки.
    """
    logger.info("Запуск конвеєра попередньої обробки.")
    load_bpms_data()
    load_camunda_data()
    logger.info("Конвеєр попередньої обробки завершено.")
