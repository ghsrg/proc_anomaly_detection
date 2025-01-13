
from src.utils.logger import get_logger
from src.utils.file_utils import save_checkpoint, load_checkpoint, load_register, save_prepared_data, load_prepared_data, save_statistics_to_json, save2csv
from src.utils.file_utils_l import is_file_exist, join_path
from src.utils.visualizer import save_training_diagram, visualize_distribution, plot_confusion_matrix, visualize_confusion_matrix
from src.config.config import LEARN_DIAGRAMS_PATH, NN_MODELS_CHECKPOINTS_PATH, NN_MODELS_DATA_PATH
from src.core.split_data import split_data, create_kfold_splits
from datetime import datetime
import torch  # Для роботи з GPU
from tqdm import tqdm
import src.core.core_gnn as gnn_core
import src.core.core_rnn as rnn_core
import src.core.core_cnn as cnn_core
import src.core.transformer as transformer
import src.core.autoencoder as autoencoder
from sklearn.metrics import classification_report


logger = get_logger(__name__)
torch.set_num_threads(12)

# Визначаємо пристрій (GPU або CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Використовується пристрій: {device}")
logger.info(f"Використовується пристрій: {device}")

MODEL_MAP = {
    "GNN": ( gnn_core),
    "RNN": ( rnn_core),
    "CNN": ( cnn_core),
    "Transformer": (transformer),
    "Autoencoder": (autoencoder)

#   "GAT": ( gat),
    # "VGAE": (vgae)
    #"TGN"
#"CapsNets"
}

def train_model(
    model_type,
    anomaly_type,
    resume=False,
    checkpoint=None,
    data_file=None,
    num_epochs=50,
    split_ratio=(0.7, 0.2, 0.1),
    learning_rate=0.001,
    batch_size=64,
    hidden_dim=64,
    patience=10,  # Кількість епох без покращення перед зупинкою
    delta=1e-4  # Мінімальне покращення, яке вважається значущим
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



        data = None
        input_dim = None
        if data_file:  # Спроба завантажити підготовлені дані
            #data_path = "/content/drive/MyDrive/prepared_data/data_Transformer_missing_steps.pt"
            data_path = join_path([NN_MODELS_DATA_PATH, f"{data_file}.pt"])
            data, input_dim, doc_dim = load_prepared_data(data_path)
        else:
            data_file = 'prepared_data'

        if data is None or input_dim is None or doc_dim is None:
            logger.info(f"data_list чи input_dim пусті, потрібна підготовка даних...")
            # Завантаження реєстрів
            normal_graphs = load_register('normalized_normal_graphs')
            anomalous_graphs = load_register('normalized_anomalous_graphs')

            if normal_graphs.empty or anomalous_graphs.empty:
                raise ValueError("Реєстри графів порожні. Перевірте дані!")
            # Підготовка даних і визначення структури
            data, input_dim, doc_dim = core_module.prepare_data(normal_graphs, anomalous_graphs, anomaly_type)
            #print(doc_dim)
            # Збереження підготовлених даних
            data_path = join_path([NN_MODELS_DATA_PATH, f"{data_file}.pt"])
            save_prepared_data(data, input_dim, doc_dim, data_path)

        # Переміщення даних на GPU/CPU
        for i in range(len(data)):
            for key, value in data[i].items():
                if isinstance(value, torch.Tensor):
                    data[i][key] = value.to(device)

        # Визначення edge_dim із першого елемента даних
        if "edge_features" in data[0] and data[0]["edge_features"] is not None:
            edge_dim = data[0]["edge_features"].size(1)  # Розмір другого виміру edge_features
        else:
            edge_dim = None  # Якщо зв'язків немає або вони не використовуються
        #print(f"Визначено input_dim: {input_dim}")
        #print(f"Визначено doc_dim: {doc_dim}")
        #print(f"Визначено edge_dim: {edge_dim}")
        #logger.info(f"Визначено input_dim: {input_dim}")
        #logger.info(f"Визначено doc_dim: {doc_dim}")
        #logger.info(f"Визначено edge_dim: {edge_dim}")

        # Ініціалізація моделі з динамічним input_dim
        model_class = getattr(core_module, model_type, None)
        if model_class is None:
            raise ValueError(f"Невідома модель: {model_type}")


        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, doc_dim=doc_dim, edge_dim=edge_dim)
        model = model.to(device)  # Переміщення моделі на GPU
        #model = core_module.GNN(input_dim=input_dim, hidden_dim=92, output_dim=1, doc_dim=doc_dim)

        #diagram = generate_model_diagram(model, model_name="Graph Neural Network")
        #diagram.render("gnn_model_diagram", view=True)  # Збереження і візуалізація

        # Оптимізатор
        optimizer = core_module.create_optimizer(model, learning_rate)

        stats = {"epochs": [], "train_loss": [], "val_precision": [], "val_recall": [], "val_roc_auc": [], "val_f1_score": [], "spend_time": [], "val_auprc":[], "val_adr":[], "val_far":[], "val_fpr":[], "val_fnr":[] }
        test_stats = {}

        # Завантаження контрольної точки
        start_epoch = 0
        if resume and checkpoint:
            checkpoint_path = join_path([NN_MODELS_CHECKPOINTS_PATH, f"{checkpoint}.pt"])
            if not is_file_exist(checkpoint_path):
                raise FileNotFoundError(f"Файл контрольної точки не знайдено: {checkpoint_path}")
            start_epoch, _, stats = load_checkpoint(checkpoint_path, model, optimizer, stats)
            start_epoch = start_epoch + 1

        # Розділення даних
        train_data, val_data, test_data = split_data(data, split_ratio)

        # Фіксація часу початку навчання
        start_time = datetime.now()
        print(f"Час початку навчання: {start_time}")
        logger.info(f"Час початку навчання: {start_time}")

        # Ініціалізація Early Stopping
        best_val_loss = float('inf')  # Найкраща валідаційна втрата
        epochs_no_improve = 0  # Лічильник епох без покращення

        #for epoch in range(start_epoch, num_epochs):
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Навчання", unit="епох", dynamic_ncols=True):
            logger.info(f"Епоха {epoch + 1}/{num_epochs}")
            #print(f"Епоха {epoch + 1}/{num_epochs}")
            stats["epochs"].append(epoch + 1)

            # Навчання за епоху
            train_loss = core_module.train_epoch(model, train_data, optimizer, batch_size)
            stats["train_loss"].append(train_loss)
            logger.info(f"Втрати на навчанні: {train_loss}")

            # Валідація
            val_stats = core_module.calculate_statistics(model, val_data)
            #print(val_stats)
            stats["val_precision"].append(val_stats["precision"])
            stats["val_recall"].append(val_stats["recall"])
            stats["val_roc_auc"].append(val_stats.get("roc_auc", 0))
            stats["val_f1_score"].append(val_stats.get("f1_score", 0))
            stats["val_auprc"].append(val_stats.get("auprc", 0))
            stats["val_adr"].append(val_stats.get("adr", 0))
            stats["val_far"].append(val_stats.get("far", 0))
            stats["val_fpr"].append(val_stats.get("fpr", 0))
            stats["val_fnr"].append(val_stats.get("fnr", 0))
            eph_end_time = datetime.now()
            eph_training_duration = eph_end_time - start_time
            stats["spend_time"].append(eph_training_duration.total_seconds())
            logger.info(f"Статистика валідації: {val_stats}")

            # Перевірка Early Stopping
            if train_loss < best_val_loss - delta:
                best_val_loss = train_loss
                epochs_no_improve = 0
                # Можна зберігати найкращу модель
                checkpoint_path = f"{NN_MODELS_CHECKPOINTS_PATH}/{model_type}_{anomaly_type}_best.pt"
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=train_loss,
                                file_path=checkpoint_path, stats=stats)
            else:
                epochs_no_improve += 1
                logger.info(f"Валідаційна втрата не покращилась: {epochs_no_improve}/{patience}")

            # Збереження контрольної точки
            checkpoint_path = f"{NN_MODELS_CHECKPOINTS_PATH}/{model_type}_{anomaly_type}_epoch_{epoch + 1}.pt"
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=train_loss, file_path=checkpoint_path, stats=stats)

            # Тестування після кожної епохи
            #test_stats = core_module.calculate_statistics(model, test_data)
            #logger.info(f"Статистика тестування (епоха {epoch + 1}): {test_stats}")

            # Збереження статистики та візуалізація після кожної епохи
            save_training_diagram(stats,
                                  f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_training_epoch_{epoch + 1}.png",
                                  test_stats, title=f"{model_type} Training and Validation Metrics {anomaly_type} anomaly")
            # Збереження матриці плутанини
            class_labels = ["Normal", "Anomalous"]
            confusion_matrix_path = f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_confusion_matrix.png"
            # Візуалізація матриці плутанини
            visualize_confusion_matrix(
                confusion_matrix_object=val_stats["confusion_matrix"],
                class_labels=class_labels,
                file_path=confusion_matrix_path
            )
            stat_path = join_path([LEARN_DIAGRAMS_PATH, f'{model_type}_{anomaly_type}_statistics'])
            save2csv(stats, stat_path)

            # Зупинка навчання
            if epochs_no_improve >= patience and epoch > 15:  # якщо оцінка не змінюється і більше Х епох тоді стоп
                logger.info(f"Early Stopping: зупинено на епосі {epoch + 1}")
                break

        # Обчислення та вивід часу навчання
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"Час завершення навчання: {end_time}")
        print(f"Тривалість навчання: {training_duration}")
        logger.info(f"Час завершення навчання: {end_time}")
        logger.info(f"Тривалість навчання: {training_duration}")
        test_stats = core_module.calculate_statistics(model, test_data)
        # Збереження фінальної статистики з тестуванням
        save_training_diagram(stats,
                              f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_training_epoch_{epoch + 1}_Final.png",
                              test_stats, title=f"{model_type} Training and Validation Metrics {anomaly_type} anomaly")

        stats["epochs"].append('Testing')
        stats["train_loss"].append('')
        stats["val_precision"].append(test_stats["precision"])
        stats["val_recall"].append(test_stats["recall"])
        stats["val_roc_auc"].append(test_stats["roc_auc"])
        stats["val_f1_score"].append(test_stats["f1_score"])
        stats["spend_time"].append('')

        stat_path = join_path([LEARN_DIAGRAMS_PATH, f'{model_type}_{anomaly_type}_statistics'])
        save2csv(stats, stat_path)
        logger.info(f"Навчання завершено для моделі {model_type} з типом аномалії {anomaly_type}")

        # Збереження матриці плутанини
        #class_labels = ["Normal", "Anomalous"]
        #confusion_matrix_path = f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_confusion_matrix.png"
        # Візуалізація матриці плутанини
        #visualize_confusion_matrix(
        #    confusion_matrix_object=test_stats["confusion_matrix_test"],
        #    class_labels=class_labels,
        #    file_path=confusion_matrix_path
        #)
    except Exception as e:
        logger.error(f"Помилка під час навчання моделі: {e}")
        raise
