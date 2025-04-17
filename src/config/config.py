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

# Генерація назв файлів логів з датою і часом
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
APP_LOG_FILE = join_path([LOGS_PATH, f"app_{current_time}.log"])
ERROR_LOG_FILE = join_path([LOGS_PATH, f"error_{current_time}.log"])

# Шляхи до даних
BASE_DATA_PATH = join_path(['data'])
BASE_PR_DATA_PATH = join_path(['data'])
RAW_DATA_PATH = join_path([BASE_DATA_PATH,'raw'])
PROCESSED_DATA_PATH = join_path([BASE_DATA_PATH, 'processed'])
OUTPUTS_DATA_PATH = join_path([BASE_DATA_PATH, 'outputs'])

# Шлях до графів
GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'graphs'])
NORMAL_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'normal_graphs'])
ANOMALOUS_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'anomalous_graphs'])
NORMALIZED_NORMAL_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'normalized_normal_graphs'])
NORMALIZED_PR_NORMAL_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'normalized_pr_normal_graphs'])
NORMALIZED_ANOMALOUS_GRAPH_PATH = join_path([BASE_DATA_PATH, 'processed', 'normalized_anomalous_graphs'])

RAW_PATH = join_path(['data', 'raw'])
REGISTER_PATH = join_path([BASE_DATA_PATH, 'registers'])
REPORTS_PATH = join_path(['reports'])

BASE_OUTPUTS_PATH = join_path([BASE_DATA_PATH, 'outputs'])
LEARN_DIAGRAMS_PATH = join_path([BASE_OUTPUTS_PATH, 'learn_diagrams'])
TEST_DIAGRAMS_PATH = join_path([BASE_OUTPUTS_PATH, 'test_diagrams'])

BASE_PR_OUTPUTS_PATH = join_path([BASE_PR_DATA_PATH, 'outputs'])
LEARN_PR_DIAGRAMS_PATH = join_path([BASE_PR_OUTPUTS_PATH, 'learn_diagrams'])
TEST_PR_DIAGRAMS_PATH = join_path([BASE_PR_OUTPUTS_PATH, 'test_diagrams'])

BASE_NN_MODELS_PATH = join_path(['nn_models'])
NN_MODELS_CHECKPOINTS_PATH = join_path([BASE_NN_MODELS_PATH, 'checkpoints'])
NN_MODELS_TRAINED_PATH = join_path([BASE_NN_MODELS_PATH, 'trained'])
NN_MODELS_DATA_PATH = join_path([BASE_NN_MODELS_PATH, 'input_data'])

BASE_PR_NN_MODELS_PATH = join_path(['nn_pr_models'])
NN_PR_MODELS_CHECKPOINTS_PATH = join_path([BASE_PR_NN_MODELS_PATH, 'checkpoints'])
NN_PR_MODELS_TRAINED_PATH = join_path([BASE_PR_NN_MODELS_PATH, 'trained'])
NN_PR_MODELS_DATA_PATH = join_path([BASE_PR_NN_MODELS_PATH, 'input_data'])

# Створення необхідних тек, якщо їх немає
make_dir(LOGS_PATH)
make_dir(RAW_PATH)
make_dir(RAW_DATA_PATH)
make_dir(PROCESSED_DATA_PATH)
make_dir(OUTPUTS_DATA_PATH)
make_dir(GRAPH_PATH)
make_dir(NORMAL_GRAPH_PATH)
make_dir(ANOMALOUS_GRAPH_PATH)
make_dir(NORMALIZED_NORMAL_GRAPH_PATH)
make_dir(NORMALIZED_PR_NORMAL_GRAPH_PATH)
make_dir(NORMALIZED_ANOMALOUS_GRAPH_PATH)
make_dir(REGISTER_PATH)
make_dir(LEARN_DIAGRAMS_PATH)
make_dir(TEST_DIAGRAMS_PATH)
make_dir(NN_MODELS_CHECKPOINTS_PATH)
make_dir(NN_MODELS_TRAINED_PATH)
make_dir(NN_MODELS_DATA_PATH)
make_dir(NN_PR_MODELS_CHECKPOINTS_PATH)
make_dir(NN_PR_MODELS_TRAINED_PATH)
make_dir(NN_PR_MODELS_DATA_PATH)
make_dir(REPORTS_PATH)
