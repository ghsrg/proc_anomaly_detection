import pandas as pd

# Завантаження та обробка даних

from src.data_sources.mssql_connector import connect_to_mssql

def preprocess_data(data):
    """
    Виконує базову обробку даних.
    :param data: DataFrame.
    :return: Оброблений DataFrame.
    """
    # Наприклад, заповнення пропусків
    data = data.fillna(0)

    print("Обробка даних завершена.")
    return data
