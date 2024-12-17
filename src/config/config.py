# Налаштування проєкту
import os
from datetime import datetime

# Параметри логування
LOG_LEVEL = "DEBUG"  # Загальний рівень логування
LOG_TO_SCREEN = {     # Налаштування для виводу на екран
    "DEBUG": 1,
    "INFO": 1,
    "WARNING": 1,
    "ERROR": 1,
    "CRITICAL": 1
}

# Шлях до логів
LOGS_DIR = os.path.join("logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Генерація файлів логів з датою і часом
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
APP_LOG_FILE = os.path.join(LOGS_DIR, f"app_{current_time}.log")
ERROR_LOG_FILE = os.path.join(LOGS_DIR, f"error_{current_time}.log")

# Шляхи до даних
BASE_DATA_PATH = "data"
RAW_DATA_PATH = os.path.join(BASE_DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed")
OUTPUTS_DATA_PATH = os.path.join(BASE_DATA_PATH, "outputs")

# Створення необхідних папок
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(OUTPUTS_DATA_PATH, exist_ok=True)