import argparse
from src.pipelines.train_pipeline import train_model
from src.pipelines.retrain_pipeline import retrain_model
from src.utils.logger import get_logger
from src.core.normalize import normalize_all_graphs
from src.utils.file_utils import initialize_register


logger = get_logger(__name__)


def run_learning_mode(args):
    """
    Логіка запуску режиму навчання моделей.

    :param args: Аргументи командного рядка для визначення сценарію навчання.
    """
    initialize_register("normalized_normal_graphs",['id', 'doc_id', 'root_proc_id', 'graph_path', 'date', 'params'])
    initialize_register("normalized_anomalous_graphs",['id', 'doc_id', 'root_proc_id', 'graph_path', 'date', 'params'])

    if args.normalize:
        normalize_all_graphs()
    else:
        logger.info("Пропуск нормалізації. Використовуються готові нормалізовані графи.")

        # Виконати навчання
    try:

        # Завантаження конфігурації
        model_type = args.model_type
        anomaly_type = args.anomaly_type
        action = args.action
        checkpoint_path = args.checkpoint

        logger.info(f"Запуск режиму навчання для моделі {model_type} з аномалією {anomaly_type}.")

        if action == "start":
            # Почати навчання з початку
            logger.info(f"Розпочинається навчання з початку для моделі {model_type}.")
            train_model(model_type=model_type, anomaly_type=anomaly_type, resume=False, checkpoint=None, num_epochs=50)

        elif action == "resume":
            # Продовжити навчання з контрольної точки
            if not checkpoint_path:
                logger.error("Контрольна точка не вказана.")
                return

            logger.info(f"Продовження навчання з контрольної точки: {checkpoint_path}.")
            train_model(model_type=model_type, anomaly_type=anomaly_type, resume=True, checkpoint=checkpoint_path, num_epochs=50)

        elif action == "retrain":
            # Донавчання моделі
            if not checkpoint_path:
                logger.error("Контрольна точка не вказана.")
                return

            logger.info(
                f"Донавчання моделі {model_type} для аномалії {anomaly_type} з контрольної точки {checkpoint_path}.")
            retrain_model(model_type=model_type, anomaly_type=anomaly_type, checkpoint=checkpoint_path)

        else:
            logger.error(f"Невідомий тип дії: {action}. Доступні дії: start, resume, retrain.")

    except Exception as e:
        logger.error(f"Помилка у режимі навчання: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Режим навчання моделей.")
    parser.add_argument("--model_type", required=True,
                        help="Тип моделі для навчання (GNN, RNN, Autoencoder, CNN, Transformers тощо).")
    parser.add_argument("--anomaly_type", required=True,
                        help="Тип аномалії для навчання (missing_steps, duplicate_steps, тощо).")
    parser.add_argument("--action", required=True, choices=["start", "resume", "retrain"],
                        help="Тип дії: start (почати з початку), resume (продовжити), retrain (донавчати).")
    parser.add_argument("--checkpoint", help="Шлях до файлу контрольної точки для продовження або донавчання.")

    args = parser.parse_args()
    run_learning_mode(args)
