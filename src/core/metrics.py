from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import numpy as np


def calculate_precision_recall(true_labels, predictions):
    """
    Розраховує Precision і Recall для моделі.

    :param true_labels: Справжні мітки (список або масив).
    :param predictions: Передбачені значення (список або масив).
    :return: Кортеж (precision, recall).
    """
    # Перетворюємо ймовірності на бінарні передбачення
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

    precision = precision_score(true_labels, binary_predictions, zero_division=0)
    recall = recall_score(true_labels, binary_predictions, zero_division=0)

    return precision, recall


def calculate_roc_auc(true_labels, predictions):
    """
    Розраховує ROC-AUC для моделі.

    :param true_labels: Справжні мітки (список або масив).
    :param predictions: Передбачені значення (список або масив).
    :return: Значення ROC-AUC.
    """
    try:
        roc_auc = roc_auc_score(true_labels, predictions)
    except ValueError as e:
        # Якщо дані незбалансовані або немає позитивних прикладів
        print(f"Помилка розрахунку ROC-AUC: {e}")
        roc_auc = np.nan

    return roc_auc


def calculate_f1_score(true_labels, predictions):
    """
    Розраховує F1-score для моделі.

    :param true_labels: Справжні мітки (список або масив).
    :param predictions: Передбачені значення (список або масив).
    :return: Значення F1-score.
    """
    # Перетворюємо ймовірності на бінарні передбачення
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

    f1 = f1_score(true_labels, binary_predictions, zero_division=0)
    return f1
