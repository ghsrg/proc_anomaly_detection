
from src.utils.logger import get_logger
from src.utils.file_utils import save_checkpoint, load_checkpoint, load_register, save_prepared_data, load_prepared_data, save_statistics_to_json, save2csv
from src.utils.file_utils_l import is_file_exist, join_path
from src.utils.visualizer import save_training_diagram, visualize_distribution, plot_confusion_matrix, visualize_confusion_matrix
from src.config.config import LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH
from src.core.split_data import split_data, create_kfold_splits
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import src.core.core_GATConv_pr as GATConv_pr_core
import src.core.core_TGAT_pr as TGAT_pr_core
#import src.core.core_rnn as rnn_core
#import src.core.core_cnn as cnn_core
#import src.core.transformer as transformer
#import src.core.autoencoder as autoencoder
from sklearn.metrics import classification_report


logger = get_logger(__name__)
torch.set_num_threads(12)

# Визначаємо пристрій (GPU або CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Використовується пристрій: {device}")
logger.info(f"Використовується пристрій: {device}")

MODEL_MAP = {
    "GATConv_pr": (GATConv_pr_core),
    "TGAT_pr": (TGAT_pr_core)#,
  #  "RNN": ( rnn_core),
  #  "CNN": ( cnn_core),
  #  "Transformer": (transformer),
  #  "Autoencoder": (autoencoder)
#ADD new models
    # "GAT": ( gat),
    # "VGAE": (vgae)
    #"TGN"
    #"CapsNets"
}

def train_model_pr(
    model_type, # Тип моделі (GATConv_pr, TGAT_pr...)
    resume=False,
    checkpoint=None,    # Шлях до контрольної точки для відновлення навчання
    data_file=None,     # Шлях до файлу з підготовленими даними
    num_epochs=80,  # Кількість епох для навчання
    split_ratio=(0.7, 0.2, 0.1),    # Частки для розділення даних на train, val, test
    learning_rate=0.002,    # Початковий рівень навчання
    batch_size=72,  # Розмір пакету для навчання
    hidden_dim=64,  # Розмір прихованого шару
    patience=6,  # Кількість епох без покращення перед зупинкою
    delta=1e-4,  # Мінімальне покращення, яке вважається значущим
    args=None,  # Аргументи командного рядка
    output_dim=470, # Розмір виходу моделі (максимальна кількість вузлів в графі)
    fraction=0.1 # Частка даних для навчання (1 - всі дані, 0.5 - половина даних)
):
    """
    Запускає процес навчання для вказаної моделі.

    :param model_type: Тип моделі (GNN, RNN, CNN, Transformers, Autoencoder).
    :param resume: Продовжити навчання з контрольної точки.
    :param checkpoint: Шлях до файлу контрольної точки.
    :param num_epochs: Кількість епох для навчання.
    :param split_ratio: Частки для розділення даних на train, val, test.
    :param learning_rate: Рівень навчання.
    :param batch_size: Розмір пакету для навчання.
    """
    try:
        logger.info(f"Запуск навчання для моделі прогнозу: {model_type}")

        if model_type not in MODEL_MAP:
            raise ValueError(f"Невідомий тип моделі: {model_type}")

        core_module = MODEL_MAP[model_type]

        pr_mode = args.pr_mode
        data = None
        input_dim = None
        if data_file:  # Спроба завантажити підготовлені дані
            #data_path = "/content/drive/MyDrive/prepared_data/data_Transformer_missing_steps.pt"
            data_path = join_path([NN_PR_MODELS_DATA_PATH, f"{data_file}.pt"])
            data, input_dim, doc_dim = load_prepared_data(data_path)
        else:
            data_file = 'prepared_data'

        if data is None or input_dim is None or doc_dim is None:
            logger.info(f"data_list чи input_dim пусті, потрібна підготовка даних...")
            # Завантаження реєстрів
            normal_graphs = load_register('normalized_normal_graphs')

            if normal_graphs.empty:
                raise ValueError("Реєстри нормалізованих графів порожні. Перевірте дані!")
            # Підготовка даних і визначення структури
            if pr_mode == 'bpmn':
                data, input_dim, doc_dim = core_module.prepare_data(normal_graphs)
            elif    pr_mode == 'logs':
                data, input_dim, doc_dim = core_module.prepare_data_log_only(normal_graphs)
            else:
                raise ValueError(f"Невірний режим підготовки даних pr_mode: {pr_mode} (bpms/logs)")
            # Збереження підготовлених даних
            data_path = join_path([NN_PR_MODELS_DATA_PATH, f"{data_file}.pt"])
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

        # Ініціалізація моделі
        model_class = getattr(core_module, model_type, None)
        if model_class is None:
            raise ValueError(f"Невідома модель: {model_type}")

        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, doc_dim=doc_dim, edge_dim=edge_dim)
        model = model.to(device)  # Переміщення моделі на GPU

        # Оптимізатор
        optimizer = core_module.create_optimizer(model, learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-3, verbose=True)

        stats = {
            "epochs": [], "train_loss": [], "spend_time": [],
            "val_accuracy": [], "val_top_k_accuracy": [], #"val_precision": [], "val_recall": [], "val_f1_score": [],
             "val_mae": [], "val_rmse": [], "val_r2": []
            }
        test_stats = {}

        # Завантаження контрольної точки
        start_epoch = 0
        if resume and checkpoint:
            checkpoint_path = join_path([NN_PR_MODELS_CHECKPOINTS_PATH, f"{checkpoint}.pt"])
            if not is_file_exist(checkpoint_path):
                raise FileNotFoundError(f"Файл контрольної точки не знайдено: {checkpoint_path}")
            start_epoch, _, stats = load_checkpoint(checkpoint_path, model, optimizer, stats)
            start_epoch = start_epoch + 1

        # Розділення даних
        train_data, val_data, test_data = split_data(data, split_ratio, fraction=fraction)

        # Фіксація часу початку навчання
        start_time = datetime.now()
        print(f"Час початку навчання: {start_time}")
        logger.info(f"Час початку навчання: {start_time}")

        # Ініціалізація Early Stopping
        best_val_loss = float('inf')  # Найкраща валідаційна втрата
        epochs_no_improve = 0  # Лічильник епох без покращення

        #for epoch in range(start_epoch, num_epochs):
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Навчання", unit="епох",position=0, dynamic_ncols=False, leave=False ):
            #logger.info(f"Епоха {epoch + 1}/{num_epochs}")
            #print(f"Епоха {epoch + 1}/{num_epochs}")
            stats["epochs"].append(epoch + 1)

            # Навчання за епоху
            train_loss = core_module.train_epoch(model, train_data, optimizer, batch_size)
            stats["train_loss"].append(train_loss)
            #logger.info(f"Втрати на навчанні: {train_loss}")

            # Валідація
            val_stats = core_module.calculate_statistics(model, val_data)
            #print(val_stats)
            if "val_accuracy" in stats: stats["val_accuracy"].append(val_stats["accuracy"])
            if "val_top_k_accuracy" in stats: stats["val_top_k_accuracy"].append(val_stats["top_k_accuracy"])
            if "val_precision" in stats: stats["val_precision"].append(val_stats.get("precision", 0))
            if "val_recall" in stats: stats["val_recall"].append(val_stats.get("recall", 0))
            if "val_f1_score" in stats: stats["val_f1_score"].append(val_stats.get("f1_score", 0))
            if "val_mae" in stats: stats["val_mae"].append(val_stats.get("mae", 0))
            if "val_rmse" in stats: stats["val_rmse"].append(val_stats.get("rmse", 0))
            if "val_r2" in stats: stats["val_r2"].append(val_stats.get("r2", 0))

            eph_end_time = datetime.now()
            eph_training_duration = eph_end_time - start_time
            stats["spend_time"].append(eph_training_duration.total_seconds())
            #logger.info(f"Статистика валідації: {val_stats}")

            if "accuracy" in val_stats:
                #print(val_stats["accuracy"])
                scheduler.step(val_stats["accuracy"])
            #for param_group in optimizer.param_groups:
                #print("Current learning rate:", param_group['lr'])

            # Перевірка Early Stopping
            if train_loss < best_val_loss - delta:
                best_val_loss = train_loss
                epochs_no_improve = 0
                # Можна зберігати найкращу модель
                checkpoint_path = f"{NN_PR_MODELS_CHECKPOINTS_PATH}/{model_type}_{pr_mode}_best.pt"
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=train_loss,
                                file_path=checkpoint_path, stats=stats)
            else:
                epochs_no_improve += 1
                #logger.info(f"Валідаційна втрата не покращилась: {epochs_no_improve}/{patience}")

            # Збереження контрольної точки
            checkpoint_path = f"{NN_PR_MODELS_CHECKPOINTS_PATH}/{model_type}_{pr_mode}_epoch_{epoch + 1}.pt"
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=train_loss, file_path=checkpoint_path, stats=stats)

            # Тестування після кожної епохи
            #test_stats = core_module.calculate_statistics(model, test_data)
            #logger.info(f"Статистика тестування (епоха {epoch + 1}): {test_stats}")

            # Збереження статистики та візуалізація після кожної епохи
            #file_path = f"{LEARN_PR_DIAGRAMS_PATH}/{model_type}_epoch_{epoch + 1}.png"
            file_path = f"{LEARN_PR_DIAGRAMS_PATH}/{model_type}_{pr_mode}_dim-{hidden_dim}_bs-{batch_size}.png"
            save_training_diagram(stats,
                                  file_path,
                                  test_stats, title=f"{model_type} Training and Validation Metrics")
            # Збереження матриці плутанини
            confusion_matrix_path = f"{LEARN_PR_DIAGRAMS_PATH}/{model_type}_{pr_mode}_CM.png"
            # Візуалізація матриці плутанини

            visualize_confusion_matrix(
                confusion_matrix_object=val_stats["confusion_matrix"],
                #class_labels=[f"Task {i}" for i in range(val_stats["confusion_matrix"].shape[0])],
                file_path=confusion_matrix_path,
                top_k=('best', 50),
                true_node_ids=val_stats.get("true_node_ids")
            )
            confusion_matrix_path_w = f"{LEARN_PR_DIAGRAMS_PATH}/{model_type}_{pr_mode}_CM_worst.png"

            visualize_confusion_matrix(
                confusion_matrix_object=val_stats["confusion_matrix"],
                # class_labels=[f"Task {i}" for i in range(val_stats["confusion_matrix"].shape[0])],
                file_path=confusion_matrix_path_w,
                top_k=('worst', 50),
                true_node_ids=val_stats.get("true_node_ids")
            )

            stat_path = join_path([LEARN_PR_DIAGRAMS_PATH, f'{model_type}_{pr_mode}_statistics'])
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
        #save_training_diagram(stats,
        #                      f"{LEARN_PR_DIAGRAMS_PATH}/{model_type}_training_epoch_{epoch + 1}_Final.png",
        #                      test_stats, title=f"{model_type} Training and Validation Metrics")

        stats["epochs"].append('Testing')
        stats["train_loss"].append(train_loss)
        if "val_accuracy" in stats: stats["val_accuracy"].append(test_stats["accuracy"])
        if "val_top_k_accuracy" in stats: stats["val_top_k_accuracy"].append(test_stats["top_k_accuracy"])
        if "val_precision" in stats: stats["val_precision"].append(test_stats.get("precision", 0))
        if "val_recall" in stats: stats["val_recall"].append(test_stats.get("recall", 0))
        if "val_f1_score" in stats: stats["val_f1_score"].append(test_stats.get("f1_score", 0))
        stats["spend_time"].append('')

        stat_path = join_path([LEARN_PR_DIAGRAMS_PATH, f'{model_type}_statistics'])
        save2csv(stats, stat_path)
        logger.info(f"Навчання завершено для моделі {model_type}")

        # Збереження матриці плутанини
        #class_labels = ["Normal", "Anomalous"]
        #confusion_matrix_path = f"{LEARN_DIAGRAMS_PATH}/{model_type}_confusion_matrix.png"
        # Візуалізація матриці плутанини
        #visualize_confusion_matrix(
        #    confusion_matrix_object=test_stats["confusion_matrix_test"],
        #    class_labels=class_labels,
        #    file_path=confusion_matrix_path
        #)
    except Exception as e:
        logger.error(f"Помилка під час навчання моделі: {e}")
        raise