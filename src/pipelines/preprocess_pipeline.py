#import os
from src.data_sources.mssql_connector import execute_query
from src.utils.logger import get_logger
from src.utils.file_utils import save_to_parquet
from src.config.secrets import BPMS_CONFIG, CAMUNDA_CONFIG

logger = get_logger(__name__)


def load_camunda_data():
    """
    Завантажує таблиці ACT_HI_ACTINST, ACT_HI_TASKINST для зв'язків між задачами та активностями з CAMUNDA.
    """
    logger.info("Завантаження даних з Camunda про виконання процесів та задач.")

    # Завантаження ACT_HI_PROCINST
    actinst_query = """
                SELECT 
                    *
                FROM 
                    ACT_HI_PROCINST;
                """
  #  procinst_df = execute_query(actinst_query, CAMUNDA_CONFIG)
  #  save_to_parquet(procinst_df, "act_hi_procinst")


    # Завантаження ACT_HI_TASKINST
    taskinst_query = """
           SELECT  ID_, TASK_DEF_KEY_, PROC_DEF_KEY_  FROM ACT_HI_TASKINST act;
    """
#    taskinst_df = execute_query(taskinst_query, CAMUNDA_CONFIG)
 #   save_to_parquet(taskinst_df, "act_hi_taskinst")

    # Завантаження ACT_HI_ACTINST
    taskinst_query = """
               SELECT *  FROM ACT_HI_ACTINST act;
        """
 #   act_inst = execute_query(taskinst_query, CAMUNDA_CONFIG)
 #   save_to_parquet(act_inst, "act_inst")


    # Завантаження BPMN XML
    bpmn_query = """
       SELECT 
           proc_def.ID_ AS process_definition_id,
            proc_def.KEY_ AS KEY_,
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
        "bpm_docs": """
            SELECT 
                doc_id,
                doc_number,
                doctype_id,
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
        "bpm_doc_purch": """
                SELECT 
                    doc_id,
                    doc_subject,
                    docstate_code,
                    KindPurchase,
                    TypePurchase,
                    ClassSSD,
                    FlowType,
                    CategoryL1,
                    CategoryL2,
                    CategoryL3,
                    Company_SO,
                    CAST(ExpectedDate AS DATETIME) AS ExpectedDate,
                    CAST(DateKTC AS DATETIME) AS DateKTC,
                    CAST(DateInWorkKaM AS DATETIME) AS DateInWorkKaM,
                    CAST(DateApprovalFD AS DATETIME) AS DateApprovalFD,
                    CAST(DateApprovalStartProcurement AS DATETIME) AS DateApprovalStartProcurement,
                    CAST(DateAppFunAss AS DATETIME) AS DateAppFunAss,
                    CAST(DateAppCommAss AS DATETIME) AS DateAppCommAss,
                    CAST(DateApprovalProcurementResults AS DATETIME) AS DateApprovalProcurementResults,
                    CAST(DateAppProcCom AS DATETIME) AS DateAppProcCom,
                    CAST(DateAppContract AS DATETIME) AS DateAppContract,
                    CAST(DateSentSO AS DATETIME) AS DateSentSO,
                    PurchasingBudget,
                    InitialPrice,
                    FinalPrice,
                    CAST(doc_createdate AS DATETIME) AS doc_createdate,
                    responsible_user_login,
                    CAM_user_login,
                    CEO2_user_login,
                    BudgetAnalyst_user_login,
                    ContractManager_user_login,
                    ManagerFunction_user_login
                FROM dbo.ks_dwh_purchase_cxo_rep;
            """,
        "bpm_tasks": "SELECT * FROM dbo.ks_dwh_bpm_tasks;",
        "bpm_doc_purchase": "SELECT * FROM dbo.ks_dwh_purchase_cxo_rep;",
        "bpm_process": "SELECT * FROM dbo.ks_dwh_bpm_process;",
        "bpm_doc_def": "SELECT * FROM dbo.ks_dwh_bpm_doc_def;",
        "bpm_proc_def": "SELECT * FROM dbo.ks_dwh_bpm_proc_def;"
    }

    # Виконання запитів і збереження у Parquet
    for filename, query in queries.items():
        logger.info(f"Зберігання файлу: {filename}")
        df = execute_query(query, BPMS_CONFIG)
        save_to_parquet(df, filename)

    # Завантаження виконання процесів і задач


    logger.info("Завантаження та обробка даних завершені.")

def run_preprocess_pipeline():
    """
    Основний запуск конвеєра обробки.
    """
    logger.info("Запуск конвеєра попередньої обробки.")
    load_bpms_data()
    #load_camunda_data()
    logger.info("Конвеєр попередньої обробки завершено.")
