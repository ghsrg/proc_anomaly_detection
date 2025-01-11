from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, precision_recall_curve, auc
import numpy as np

def calculate_auprc(true_labels, predictions):
    """
    1 Розраховує Area Under Precision-Recall Curve (AUPRC) - Здатність моделі справлятися з дисбалансованими даними
    Для задач, де аномалії є рідкісними, враховує співвідношення точності та повноти що краще відображає баланс ніж ROC-AUC

    :param true_labels: Справжні мітки (0 для нормальних, 1 для аномалій).
    :param predictions: Передбачені значення (ймовірності).
    :return: Значення AUPRC.
    """

    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    auprc = auc(recall, precision)
    return auprc


def calculate_roc_auc(true_labels, predictions):
    """
    2 Розраховує ROC-AUC для моделі. Загальну здатність моделі розрізняти між аномаліями і нормальними даними
    Значення ROC-AUC ближче до 1 вказує на те, що модель має високу точність і повноту одночасно.
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
    3 Розраховує F1-score для моделі. Баланс між точністю і повнотою
    Баланс між Precision і Recall. Підходить для задач, де важливо одночасно мінімізувати хибні позитиви і негативи.
    :param true_labels: Справжні мітки (список або масив).
    :param predictions: Передбачені значення (список або масив).
    :return: Значення F1-score.
    """
    # Перетворюємо ймовірності на бінарні передбачення
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

    f1 = f1_score(true_labels, binary_predictions, zero_division=0)
    return f1

def calculate_precision_recall(true_labels, predictions):
    """
   4 і 5 Розраховує Precision і Recall для моделі.
   Точність (Precision) - Частка правильно виявлених аномалій серед усіх передбачених як аномалії
     Висока точність означає, що модель мінімізує хибні позитивні спрацьовування (зайві попередження).
   Повнота (Recall) - Частка аномалій, які були правильно виявлені.
   Висока повнота означає, що модель знаходить більшість справжніх аномалій, навіть якщо при цьому підвищуються хибні спрацьовування.

    :param true_labels: Справжні мітки (список або масив).
    :param predictions: Передбачені значення (список або масив).
    :return: Кортеж (precision, recall).
    """
    # Перетворюємо ймовірності на бінарні передбачення
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

    precision = precision_score(true_labels, binary_predictions, zero_division=0)
    recall = recall_score(true_labels, binary_predictions, zero_division=0)

    return precision, recall

def calculate_adr(true_labels, predictions):
    """
    6 Розраховує Anomaly Detection Rate (ADR) - Коефіцієнт виявлення аномалій. Частка аномалій, правильно ідентифікованих моделлю
    Дає пряме уявлення про ефективність моделі у виявленні всіх аномалій.

    :param true_labels: Справжні мітки (0 для нормальних, 1 для аномалій).
    :param predictions: Передбачені значення (ймовірності або бінарні передбачення).
    :return: Значення ADR.
    """
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    anomalies_detected = sum((pred == 1 and true == 1) for pred, true in zip(binary_predictions, true_labels))
    total_anomalies = sum(true_labels)
    adr = anomalies_detected / total_anomalies if total_anomalies > 0 else 0
    return adr


def calculate_far(true_labels, predictions):
    """
    8 Розраховує False Alarm Rate (FAR) - Коефіцієнт хибних тривог. Частка нормальних об'єктів, помилково визначених як аномалії
    FAR оцінює схильність моделі до "перестрахування", тобто неправильного сигналу про аномалії.

    :param true_labels: Справжні мітки (0 для нормальних, 1 для аномалій).
    :param predictions: Передбачені значення (ймовірності або бінарні передбачення).
    :return: Значення FAR.
    """
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    false_alarms = sum((pred == 1 and true == 0) for pred, true in zip(binary_predictions, true_labels))
    total_normals = len(true_labels) - sum(true_labels)
    far = false_alarms / total_normals if total_normals > 0 else 0
    return far


def calculate_fpr(true_labels, predictions):
    """
    7 Розраховує False Positive Rate (FPR).Частка нормальних об'єктів, які помилково визначені як аномалії
    Визначає, наскільки часто модель "хибно звинувачує" нормальні дані. Це відношення хибних позитивів до загальної кількості нормальних даних.

    Формула:
        FPR = FP / (FP + TN)

    :param true_labels: Справжні мітки (0 для нормальних, 1 для аномалій).
    :param predictions: Передбачені значення (ймовірності або бінарні передбачення).
    :return: Значення FPR.
    """
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    fp = sum((pred == 1 and true == 0) for pred, true in zip(binary_predictions, true_labels))
    tn = sum((pred == 0 and true == 0) for pred, true in zip(binary_predictions, true_labels))

    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = np.nan  # Якщо немає TN і FP
        print("Попередження: Немає TN або FP для обчислення FPR.")

    return fpr


def calculate_fnr(true_labels, predictions):
    """
    9 Розраховує False Negative Rate (FNR). Частка аномалій, які модель не змогла виявити.
    Визначає, наскільки часто модель пропускає справжні аномалії. Це відношення хибних негативів до загальної кількості аномалій.

    Формула:
        FNR = FN / (FN + TP)

    :param true_labels: Справжні мітки (0 для нормальних, 1 для аномалій).
    :param predictions: Передбачені значення (ймовірності або бінарні передбачення).
    :return: Значення FNR.
    """
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    fn = sum((pred == 0 and true == 1) for pred, true in zip(binary_predictions, true_labels))
    tp = sum((pred == 1 and true == 1) for pred, true in zip(binary_predictions, true_labels))

    try:
        fnr = fn / (fn + tp)
    except ZeroDivisionError:
        fnr = np.nan  # Якщо немає TP і FN
        print("Попередження: Немає TP або FN для обчислення FNR.")

    return fnr






