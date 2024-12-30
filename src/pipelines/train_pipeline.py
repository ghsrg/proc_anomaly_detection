import json
from src.utils.logger import get_logger
from src.utils.file_utils import save_checkpoint, load_checkpoint, load_register
from src.utils.file_utils_l import is_file_exist
from src.utils.visualizer import save_training_diagram
from nn_models.architectures import cnn, gnn, rnn, transformer, autoencoder
from src.config.config import LEARN_DIAGRAMS_PATH
import src.core.gnn as gnn_core
import src.core.rnn as rnn_core
import src.core.cnn as cnn_core
import src.core.transformer as transformer_core
import src.core.autoencoder as autoencoder_core

logger = get_logger(__name__)

MODEL_MAP = {
    "GNN": (gnn, gnn_core),
    "RNN": (rnn, rnn_core),
    "CNN": (cnn, cnn_core),
    "Transformers": (transformer, transformer_core),
    "Autoencoder": (autoencoder, autoencoder_core)
}

def train_model(model_type, anomaly_type, resume=False, checkpoint=None, num_epochs=50):
    """
    Запускає процес навчання для вказаної моделі.

    :param model_type: Тип моделі (GNN, RNN, CNN, Transformers, Autoencoder).
    :param anomaly_type: Тип аномалії для навчання.
    :param resume: Продовжити навчання з контрольної точки.
    :param checkpoint: Шлях до файлу контрольної точки.
    :param num_epochs: Кількість епох для навчання.
    """
    try:
        logger.info(f"Запуск навчання для моделі: {model_type}, тип аномалії: {anomaly_type}")

        if model_type not in MODEL_MAP:
            raise ValueError(f"Невідомий тип моделі: {model_type}")

        model_architecture, core_module = MODEL_MAP[model_type]
        model = model_architecture.create_model()

        # Завантаження контрольної точки
        start_epoch = 0
        if resume and checkpoint:
            if not is_file_exist(checkpoint):
                raise FileNotFoundError(f"Файл контрольної точки не знайдено: {checkpoint}")
            start_epoch, _ = load_checkpoint(checkpoint, model)

        # Завантаження реєстрів
        normal_graphs = load_register('normal_graphs')
        anomalous_graphs = load_register('anomalous_graphs')

        if normal_graphs.empty or anomalous_graphs.empty:
            raise ValueError("Реєстри графів порожні. Перевірте дані!")

        # Підготовка даних
        data = core_module.prepare_data(normal_graphs, anomalous_graphs, anomaly_type)

        # Навчання
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Епоха {epoch + 1}/{num_epochs}")
            loss = core_module.train_epoch(model, data)

            # Збереження контрольної точки
            checkpoint_path = f"nn_models/checkpoints/{model_type}_{anomaly_type}_epoch_{epoch + 1}.pt"
            save_checkpoint(model, epoch, loss, checkpoint_path)

        # Збереження статистики та візуалізація
        stats = core_module.calculate_statistics(model, data)
        save_training_diagram(stats, f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_training.png")

        logger.info(f"Навчання завершено для моделі {model_type} з типом аномалії {anomaly_type}")

    except Exception as e:
        logger.error(f"Помилка під час навчання моделі: {e}")
        raise


def load_and_prepare_registers(normal_register_name: str, anomaly_register_name: str, anomaly_type: str):
    """
    Завантажує та перевіряє реєстри графів.

    :param normal_register_name: Назва реєстру нормальних графів.
    :param anomaly_register_name: Назва реєстру аномальних графів.
    :param anomaly_type: Тип аномалії для фільтрації.
    :return: Списки нормальних та аномальних графів.
    """
    try:
        normal_graphs = load_register(normal_register_name)
        anomalous_graphs = load_register(anomaly_register_name)

        if normal_graphs.empty:
            raise ValueError(f"Реєстр {normal_register_name} порожній.")

        if anomalous_graphs.empty:
            raise ValueError(f"Реєстр {anomaly_register_name} порожній.")

        # Фільтрація аномалій за типом
        filtered_anomalous_graphs = anomalous_graphs[anomalous_graphs['params'].apply(
            lambda x: json.loads(x)['anomaly_type'] == anomaly_type
        )]

        if filtered_anomalous_graphs.empty:
            raise ValueError(f"Немає графів з аномаліями типу {anomaly_type} у реєстрі {anomaly_register_name}.")

        return normal_graphs, filtered_anomalous_graphs

    except Exception as e:
        logger.error(f"Помилка при завантаженні реєстрів: {e}")
        raise

