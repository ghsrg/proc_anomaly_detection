import pandas as pd
import pyodbc
from src.utils.logger import get_logger
from src.config.secrets import MSSQL_CONFIG

logger = get_logger(__name__)

def get_mssql_connection():
    """
    Створює підключення до MSSQL бази даних.
    :return: Об'єкт підключення.
    """
    try:
        conn = pyodbc.connect(
            f"DRIVER={MSSQL_CONFIG['driver']};"
            f"SERVER={MSSQL_CONFIG['server']};"
            f"DATABASE={MSSQL_CONFIG['database']};"
            f"UID={MSSQL_CONFIG['username']};"
            f"PWD={MSSQL_CONFIG['password']}"
        )
        logger.info("Підключення до MSSQL успішно встановлено.")
        return conn
    except Exception as e:
        logger.error(f"Помилка підключення до MSSQL: {e}")
        return None

def execute_query(query: str):
    """
    Виконує SQL-запит і повертає результат у вигляді DataFrame.
    :param query: SQL-запит.
    :return: DataFrame з результатами.
    """
    conn = get_mssql_connection()
    if not conn:
        logger.error("З'єднання не було встановлено. Запит не виконано.")
        return pd.DataFrame()  # Повертаємо пустий DataFrame як універсальну структуру

    try:
        df = pd.read_sql_query(query, conn)
        logger.debug(f"Запит виконано успішно: {query}")
        return df
    except Exception as e:
        logger.error(f"Помилка виконання запиту: {e}")
        logger.error(f"Запит, який викликав помилку: {query}")
        return pd.DataFrame()
    finally:
        conn.close()
        logger.info("Підключення до MSSQL закрито.")