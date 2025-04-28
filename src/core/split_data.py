from sklearn.model_selection import KFold
import pandas as pd
import random
import numpy as np

def split_data(data, split_ratio=(0.7, 0.2, 0.1), shuffle=True, fraction=1.0, seed=None):
    """
    Розділяє дані на навчальну, валідаційну та тестову вибірки, з можливістю обмежити частку і зафіксувати seed.

    :param data: Список Data-об'єктів.
    :param split_ratio: Частки (train, val, test), сума має бути 1.0
    :param shuffle: Чи перемішувати перед розділенням
    :param fraction: Скільки відсотків даних повернути (0.0 – 1.0)
    :param seed: Випадкове зерно для забезпечення відтворюваності
    :return: train_data, val_data, test_data
    """
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("Сума split_ratio повинна дорівнювати 1.")

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)

    total_count = len(data)
    train_end = int(total_count * split_ratio[0])
    val_end = train_end + int(total_count * split_ratio[1])

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    if fraction < 1.0:
        def reduce(subset):
            count = max(1, int(len(subset) * fraction))
            return subset[:count]
        train_data = reduce(train_data)
        val_data = reduce(val_data)
        test_data = reduce(test_data)

    return train_data, val_data, test_data


def split_data_old(data, split_ratio=(0.7, 0.2, 0.1), shuffle=True):
    """
    Розділяє дані на навчальну, валідаційну та тестову вибірки.

    :param data: Список, де кожен елемент — кортеж (graph, label).
    :param split_ratio: Кортеж із частками для train, val, test (сума повинна дорівнювати 1).
    :param shuffle: Чи перемішувати дані перед розділенням.
    :return: train_data, val_data, test_data
    """
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("Сума split_ratio повинна дорівнювати 1.")

    if shuffle:
        random.shuffle(data)  # Перемішуємо дані, якщо зазначено

    total_count = len(data)
    train_end = int(total_count * split_ratio[0])
    val_end = train_end + int(total_count * split_ratio[1])

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def create_kfold_splits(data, k=5, shuffle=True, random_state=None):
    """
    Створює K-fold розділи даних для валідації.

    :param data: Повний набір даних (наприклад, DataFrame).
    :param k: Кількість фолдів.
    :param shuffle: Чи потрібно перемішувати дані перед розділенням.
    :param random_state: Початкове значення для генератора випадкових чисел.
    :return: Список (або генератор) з (train_index, val_index).
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        yield train_data, val_data

