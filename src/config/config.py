# Налаштування проєкту
from src.utils.file_utils_l import make_dir, join_path
from datetime import datetime


# Параметри логування
LOG_LEVEL = "INFO"  # Загальний рівень логування
LOG_TO_SCREEN = {     # Налаштування для виводу на екран
    "DEBUG": 1,
    "INFO": 1,
    "WARNING": 1,
    "ERROR": 1,
    "CRITICAL": 1
}

# Шлях до логів
LOGS_PATH = join_path(['logs'])
make_dir(LOGS_PATH)

# Генерація назв файлів логів з датою і часом
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
APP_LOG_FILE = join_path([LOGS_PATH,f"app_{current_time}.log"])
ERROR_LOG_FILE = join_path([LOGS_PATH,f"error_{current_time}.log"])

# Шляхи до даних
BASE_DATA_PATH = join_path(['data'])
RAW_DATA_PATH = join_path([BASE_DATA_PATH,'raw'])
PROCESSED_DATA_PATH = join_path([BASE_DATA_PATH, 'processed'])
OUTPUTS_DATA_PATH = join_path([BASE_DATA_PATH, 'outputs'])

# Шлях до графів
GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'graphs'])
NORMAL_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'normal_graphs'])
ANOMALOUS_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'anomalous_graphs'])

RAW_PATH = join_path(['data', 'raw'])

REGISTER_PATH = join_path([BASE_DATA_PATH, 'registers'])

# Створення необхідних папок
make_dir(RAW_DATA_PATH)
make_dir(PROCESSED_DATA_PATH)
make_dir(OUTPUTS_DATA_PATH)
make_dir(GRAPH_PATH)
make_dir(REGISTER_PATH)