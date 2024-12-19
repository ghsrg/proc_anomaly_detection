from sqlalchemy import create_engine
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def execute_query(query, connection_config):
    """
    Виконує SQL-запит та повертає результат як DataFrame.
    :param query: SQL-запит.
    :param connection_config: Конфігурація підключення (BPMS_CONFIG або CAMUNDA_CONFIG).
    :return: pandas DataFrame з результатами запиту.
    """
    try:
        # Створення URI підключення для SQLAlchemy
        db_uri = (f"mssql+pyodbc://{connection_config['username']}:{connection_config['password']}@"
                  f"{connection_config['server']}/{connection_config['database']}?driver=ODBC+Driver+17+for+SQL+Server")

        # Створення SQLAlchemy engine
        engine = create_engine(db_uri)
        logger.info(f"Підключення до MSSQL ({connection_config['database']}) через SQLAlchemy успішно встановлено.")

        # Виконання запиту
        df = pd.read_sql_query(query, engine)
        logger.info("Запит виконано успішно.")
        return df

    except Exception as e:
        logger.error(f"Помилка виконання запиту: {e}")
        return pd.DataFrame()
