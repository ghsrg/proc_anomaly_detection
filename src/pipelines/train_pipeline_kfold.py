from src.utils.logger import get_logger
from src.utils.file_utils import save_checkpoint, load_checkpoint, load_register, save_prepared_data, load_prepared_data, save_statistics_to_json, save2csv
from src.utils.file_utils_l import is_file_exist, join_path
from src.utils.visualizer import save_training_diagram, generate_model_diagram, visualize_distribution
from src.config.config import LEARN_DIAGRAMS_PATH, NN_MODELS_CHECKPOINTS_PATH, NN_MODELS_DATA_PATH
from src.core.split_data import create_kfold_splits
from datetime import datetime
from tqdm import tqdm
import src.core.core_gnn as gnn_core
import src.core.core_rnn as rnn_core
import src.core.core_cnn as cnn_core
import src.core.transformer as transformer
import src.core.autoencoder as autoencoder

logger = get_logger(__name__)

MODEL_MAP = {
    "GNN": gnn_core,
    "RNN": rnn_core,
    "CNN": cnn_core,
    "Transformer": transformer,
    "Autoencoder": autoencoder
}

def train_model_kfold(
    model_type,
    anomaly_type,
    data_file=None,
    num_epochs=100,
    learning_rate=0.001,
    batch_size=64,
    hidden_dim=64,
    patience=10,
    delta=1e-4,
    k=5
):
    try:
        logger.info(f"Starting K-Fold training for model: {model_type}, anomaly type: {anomaly_type}")

        if model_type not in MODEL_MAP:
            raise ValueError(f"Unknown model type: {model_type}")

        core_module = MODEL_MAP[model_type]

        # Load data
        data, input_dim, doc_dim = load_prepared_data(
            join_path([NN_MODELS_DATA_PATH, f"{data_file}.pt"])
        )

        if data is None:
            raise ValueError("Prepared data not found!")

        # Initialize model
        model_class = getattr(core_module, model_type, None)
        if model_class is None:
            raise ValueError(f"Unknown model class: {model_type}")

        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, doc_dim=doc_dim)
        optimizer = core_module.create_optimizer(model, learning_rate)

        all_stats = []

        kfold = create_kfold_splits(data, k=k, shuffle=True)
        fold = 1

        for train_data, val_data in kfold:
            logger.info(f"Fold {fold}/{k}")

            stats = {"epochs": [], "train_loss": [], "val_precision": [], "val_recall": [], "val_roc_auc": [], "val_f1_score": []}

            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold}", unit="epoch"):
                stats["epochs"].append(epoch + 1)

                train_loss = core_module.train_epoch(model, train_data, optimizer, batch_size)
                stats["train_loss"].append(train_loss)

                val_stats = core_module.calculate_statistics(model, val_data)
                stats["val_precision"].append(val_stats["precision"])
                stats["val_recall"].append(val_stats["recall"])
                stats["val_roc_auc"].append(val_stats.get("roc_auc", None))
                stats["val_f1_score"].append(val_stats.get("f1_score", None))

                if train_loss < best_val_loss - delta:
                    best_val_loss = train_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} for fold {fold}")
                    break

            save_training_diagram(stats, f"{LEARN_DIAGRAMS_PATH}/fold_{fold}_{model_type}_{anomaly_type}.png")
            all_stats.append(stats)
            fold += 1

        # Collect stats for distribution visualization
        min_values = [min(fold_stats["train_loss"]) for fold_stats in all_stats]
        max_values = [max(fold_stats["train_loss"]) for fold_stats in all_stats]
        mean_values = [sum(fold_stats["train_loss"]) / len(fold_stats["train_loss"]) for fold_stats in all_stats]

        visualize_distribution(
            {
                "Min": min_values,
                "Max": max_values,
                "Mean": mean_values
            },
            f"{LEARN_DIAGRAMS_PATH}/{model_type}_{anomaly_type}_distribution.png"
        )

    except Exception as e:
        logger.error(f"Error during K-Fold training: {e}")
        raise
