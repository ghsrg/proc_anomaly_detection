import os
from src.data_sources.mssql_connector import execute_query
from src.utils.logger import get_logger
from src.config.config import RAW_DATA_PATH


logger = get_logger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)  # Переконуємося, що папка існує


def save_to_csv(df, filename):
    """
    Зберігає DataFrame у CSV-файл у папку RAW.
    :param df: pandas DataFrame.
    :param filename: Назва файлу для збереження.
    """
    if not df.empty:
        file_path = os.path.join(RAW_DATA_PATH, filename)
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info(f" Дані збережено у файл: {file_path}")
    else:
        logger.warning(f" Дані для файлу {filename} порожні. Запис не виконано.")

def load_and_preprocess_data():
    """
    Завантажує дані з MSSQL та зберігає їх у папку RAW.
    """
    logger.info("Початок завантаження та обробки даних.")

    # SQL-запити
    queries = {
        "ks_dwh_bpm_docs.csv": """
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
        "ks_dwh_bpm_tasks.csv": "SELECT * FROM dbo.ks_dwh_bpm_tasks;",
        "ks_dwh_purchase_cxo_rep.csv": "SELECT * FROM dbo.ks_dwh_purchase_cxo_rep;"
    }

    # Виконання запитів і збереження у CSV
    for filename, query in queries.items():
        logger.info(f"Виконується запит для файлу: {filename}")
        df = execute_query(query)
        save_to_csv(df, filename)

    logger.info("Завантаження та обробка даних завершені.")

def run_preprocess_pipeline():
    load_and_preprocess_data()