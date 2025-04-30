# Режим виконання
from src.utils.logger import get_logger
from src.utils.file_utils import aggregate_statistics, save_aggregated_statistics, save_checkpoint, load_checkpoint, load_register, save_prepared_data, load_prepared_data, load_global_statistics_from_json, save2csv
from src.utils.file_utils_l import is_file_exist, join_path
from src.config.config import LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH, PROCESSED_DATA_PATH

logger = get_logger(__name__)

def run_analitics_mode(args):
    """
    Аналітики резульатів.
    """

    logger.info("🚀 Запущено аналітичний режим.")
    print("⚙️ Виконується аналітичний режим ...")

    final_df = aggregate_statistics(LEARN_PR_DIAGRAMS_PATH)

    final_df_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_df_statistics.xlsx'])
    save_aggregated_statistics(final_df, final_df_file)
