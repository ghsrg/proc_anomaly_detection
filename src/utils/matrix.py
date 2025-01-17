import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Створення списку для anomalies_metrics
anomalies_metrics = [
    'Missing Steps - Normal as Normal',
    'Missing Steps - Normal as Anomaly',
    'Missing Steps - Anomaly as Anomaly',
    'Missing Steps - Anomaly as Normal',

    'Duplicate Steps - Normal as Normal',
    'Duplicate Steps - Normal as Anomaly',
    'Duplicate Steps - Anomaly as Anomaly',
    'Duplicate Steps - Anomaly as Normal',

    'Wrong Route - Normal as Normal',
    'Wrong Route - Normal as Anomaly',
    'Wrong Route - Anomaly as Anomaly',
    'Wrong Route - Anomaly as Normal',

    'Abnormal Duration - Normal as Normal',
    'Abnormal Duration - Normal as Anomaly',
    'Abnormal Duration - Anomaly as Anomaly',
    'Abnormal Duration - Anomaly as Normal'
]

# Створення датасету з нулями вручну 3340 3336
dataset = pd.DataFrame([
    [2852, 2864, 2880, 2865, 2873],  # Missing Steps - True Normal*Predicted Normal
    [92, 97, 92, 95, 71],  # Missing Steps - True Normal*Predicted Anomaly
    [402, 391, 380, 392, 317],  # Missing Steps - True Anomaly*Predicted Anomaly
    [0, 5, 10, 0, 91],  # Missing Steps - True Anomaly*Predicted Normal

    [2874, 2861, 2852, 2878, 2884],  # Duplicate Steps - True Normal*Predicted Normal
    [76, 95, 109, 98, 70],  # Duplicate Steps - True Normal*Predicted Anomaly
    [345, 393, 388, 373, 339],  # Duplicate Steps - True Anomaly*Predicted Anomaly
    [0, 54, 0, 0, 56],  # Duplicate Steps - True Anomaly*Predicted Normal

    [2856, 2866, 2879, 2862, 2879],  # Wrong Route - True Normal*Predicted Normal
    [99, 92, 95, 76, 63],  # Wrong Route - True Normal*Predicted Anomaly
    [397, 394, 378, 414, 268],  # Wrong Route - True Anomaly*Predicted Anomaly
    [0, 0, 0, 0, 142],  # Wrong Route - True Anomaly*Predicted Normal

    [2867, 2844, 2859, 2852, 2918],  # Abnormal Duration - True Normal*Predicted Normal
    [89, 110, 108, 92, 77],  # Abnormal Duration - True Normal*Predicted Anomaly
    [389, 398, 385, 408, 284],  # Abnormal Duration - True Anomaly*Predicted Anomaly
    [0, 19, 0, 0, 73]  # Abnormal Duration - True Anomaly*Predicted Normal

], columns=['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder'])

# Створення комбінованої теплової карти
fig, ax = plt.subplots(figsize=(12, 8))

# Лейбли для архітектур і поєднаних аномалій і метрик
architectures = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
anomalies_metrics = anomalies_metrics

# Палітри кольорів для різних категорій
# Зелена палітра для True Normal * Predicted Normal і True Anomaly * Predicted Anomaly
cmap_normal = sns.light_palette("green", as_cmap=True)
# Червона палітра для True Normal * Predicted Anomaly і True Anomaly * Predicted Normal
cmap_anomaly = sns.light_palette("red", as_cmap=True)

# Створюємо маску для кожної категорії: чергування зеленого та червоного
mask_combined = np.zeros_like(dataset.values)

# Маскуємо зелений і червоний
for i in range(len(anomalies_metrics)):
    if i % 2 == 0:  # Для парних рядків (0, 2, 4...) - зелений
        mask_combined[i, :] = 1  # Зелений
    else:  # Для непарних рядків (1, 3, 5...) - червоний
        mask_combined[i, :] = 0  # Червоний

# Побудова комбінованої теплової карти
sns.heatmap(dataset, annot=True, fmt='.0f', cmap=cmap_normal, cbar=True,
            xticklabels=architectures, yticklabels=anomalies_metrics, ax=ax, mask=mask_combined==0)

sns.heatmap(dataset, annot=True, fmt='.0f', cmap=cmap_anomaly, cbar=True,
            xticklabels=architectures, yticklabels=anomalies_metrics, ax=ax, mask=mask_combined==1)

# Заголовки і мітки
ax.set_title('Combined Confusion Matrix for Architectures and Anomalies', fontsize=14)
ax.set_xlabel('Architecture', fontsize=12)
ax.set_ylabel('Anomalies and Metrics (True / Predicted)', fontsize=12)

# Показати графік
plt.tight_layout()
plt.show()