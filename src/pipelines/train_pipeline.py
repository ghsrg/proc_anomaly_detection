
from src.utils.logger import get_logger
from src.utils.file_utils import save_checkpoint, load_checkpoint, load_register, save_register, load_graph, save_graph
from src.utils.file_utils_l import is_file_exist, join_path
from src.utils.visualizer import save_training_diagram
from src.config.config import LEARN_DIAGRAMS_PATH, NN_MODELS_CHECKPOINTS_PATH, NORMALIZED_NORMAL_GRAPH_PATH, NORMALIZED_ANOMALOUS_GRAPH_PATH
from src.core.split_data import split_data
import src.core.core_gnn as gnn_core
#import src.core.core_rnn as rnn_core
#import src.core.core_cnn as cnn_core
#import src.core.transformer as transformer_core
#import src.core.core_autoencoder as autoencoder_core


logger = get_logger(__name__)

MODEL_MAP = {
    "GNN": ( gnn_core)
 #   "RNN": ( rnn_core),
 #   "CNN": ( cnn_core),
 #   "Transformers": (transformer_core),
 #   "Autoencoder": (autoencoder_core)
}

def train_model(
    model_type,
    anomaly_type,
    resume=False,
    checkpoint=None,
    num_epochs=50,
    split_ratio=(0.7, 0.2, 0.1),
    learning_rate=0.001,
    batch_size=8
):
    """
    Запускає процес навчання для вказаної моделі.

    :param model_type: Тип моделі (GNN, RNN, CNN, Transformers, Autoencoder).
    :param anomaly_type: Тип аномалії для навчання.
    :param resume: Продовжити навчання з контрольної точки.
    :param checkpoint: Шлях до файлу контрольної точки.
    :param num_epochs: Кількість епох для навчання.
    :param split_ratio: Частки для розділення даних на train, val, test.
    :param learning_rate: Рівень навчання.
    :param batch_size: Розмір пакету для навчання.
    """
    try:
        logger.info(f"Запуск навчання для моделі: {model_type}, тип аномалії: {anomaly_type}")

        if model_type not in MODEL_MAP:
            raise ValueError(f"Невідомий тип моделі: {model_type}")

        core_module = MODEL_MAP[model_type]

        # Завантаження реєстрів
        normal_graphs = load_register('normalized_normal_graphs')
        anomalous_graphs = load_register('normalized_anomalous_graphs')

        if normal_graphs.empty or anomalous_graphs.empty:
            raise ValueError("Реєстри графів порожні. Перевірте дані!")

        # Підготовка даних і визначення структури
        data, input_dim = core_module.prepare_data(normal_graphs, anomalous_graphs, anomaly_type)
        logger.info(f"Визначено input_dim: {input_dim}")

        # Ініціалізація моделі з динамічним input_dim
        model = core_module.GNN(input_dim=input_dim, hidden_dim=92, output_dim=1)

        # Завантаження контрольної точки
        start_epoch = 0
        if resume and checkpoint:
            if not is_file_exist(checkpoint):
                raise FileNotFoundError(f"Файл контрольної точки не знайдено: {checkpoint}")
            start_epoch, _ = load_checkpoint(checkpoint, model)

        # Розділення даних
        train_data, val_data, test_data = split_data(data, split_ratio)

        # Оптимізатор
        optimizer = core_module.create_optimizer(model, learning_rate)

        stats = {"epochs": [], "train_loss": [], "val_precision": [], "val_recall": [], "val_roc_auc": [], "val_f1_score": []}
        test_stats = {}
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Епоха {epoch + 1}/{num_epochs}")
            print(f"Епоха {epoch + 1}/{num_epochs}")
            stats["epochs"].append(epoch + 1)

            # Навчання за епоху
            train_loss = core_module.train_epoch(model, train_data, optimizer, batch_size)
            stats["train_loss"].append(train_loss)
            logger.info(f"Втрати на навчанні: {train_loss}")

            # Валідація
            val_stats = core_module.calculate_statistics(model, val_data)
            print(train_loss, val_stats["precision"], val_stats["recall"], val_stats["recall"], val_stats.get("roc_auc", None))
            stats["val_precision"].append(val_stats["precision"])
            stats["val_recall"].append(val_stats["recall"])
            stats["val_roc_auc"].append(val_stats.get("roc_auc", None))
            stats["val_f1_score"].append(val_stats.get("f1_score", None))
            logger.info(f"Статистика валідації: {val_stats}")

            # Збереження контрольної точки
            checkpoint_path = f"{NN_MODELS_CHECKPOINTS_PATH}/{model_type}_{anomaly_type}_epoch_{epoch + 1}.pt"
            save_checkpoint(model=model, optimizer=None, epoch=epoch, loss=train_loss, file_path=checkpoint_path)

            # Тестування після кожної епохи
            #test_stats = core_module.calculate_statistics(model, test_data)
            #logger.info(f"Статистика тестування (епоха {epoch + 1}): {test_stats}")

             # Збереження статистики та візуалізація після кожної епохи
            save_training_diagram(stats,
                                  f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_training_epoch_{epoch + 1}.png",
                                  test_stats)

        test_stats = core_module.calculate_statistics(model, test_data)
        # Збереження фінальної статистики з тестуванням
        save_training_diagram(stats,
                              f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_training_epoch_{epoch + 1}_Final.png",
                              test_stats)

        logger.info(f"Навчання завершено для моделі {model_type} з типом аномалії {anomaly_type}")

    except Exception as e:
        logger.error(f"Помилка під час навчання моделі: {e}")
        raise


