# Проєкт: Аналіз аномалій у бізнес-процесах

## Опис проєкту
Цей проєкт спрямований на виявлення аномалій у бізнес-процесах, таких як закупівлі, з використанням підходів машинного навчання, зокрема **Unsupervised Learning** (кластеризація та Autoencoder) і **Reinforcement Learning** (на майбутніх етапах).


### Основні функції:
1. Завантаження даних із MSSQL та REST API.
2. Попередня обробка даних та їх нормалізація.
3. Виявлення аномалій через кластеризацію (DBSCAN, HDBSCAN).
4. Використання Autoencoder для аналізу атрибутів.
5. Побудова графів і аналіз зв'язків через GNN.
6. Інтеграція Reinforcement Learning для навчання моделі на знайдених аномаліях (планується).

---

## Архітектура проєкту

### Загальна структура:

```plaintext
project_root/
├── data/                     # Дані
│   ├── raw/                  # Сирі дані
│   ├── processed/            # Оброблені дані
│   |   ├── graphs               # Реальні графи
│   |   ├── normal_graphs        # Згенеровані варіації графів 
│   |   ├── anomalous_graphs     # Згенеровані варіації аномальних графіва
│   ├── outputs/              # Результати моделей
│   |   ├── learn_diagrams       # діаграми процесу навчання
│   |   ├── test_diagrams        # діаграми процесу тестування
│   ├── registers/            # Реєстри як результати обробки, переліки, тощо
├── nn_models/                # Файли моделей нейромереж
│   ├── checkpoints/             # Проміжні збереження моделей
│   ├── trained/                 # Навчені моделі
│   ├── architectures/           # Опис архітектур моделей
├── reports/                  # Звіти для наукової роботи
├── logs
│   ├── app.log               # Логування роботи системи
│   ├── error.log             # Логування помилок
├── src/                      # Основний код
│   ├── core/                 # Центральна логіка
│   │   ├── core_autoencoder.py       # Реалізація Autoencoder
│   │   ├── core_cnn.py               # Реалізація CNN
│   │   ├── core_gnn.py               # Реалізація GNN
│   │   ├── core_rnn.py               # Реалізація RNN
│   │   ├── core_transformers.py      # Реалізація Transformers
│   │   ├── data_processing.py   # Загальна обробка даних для моделей
│   │   ├── clustering.py        # Кластеризація реальних даних для пошуку аномалій
│   │   ├── metrics.py           # Метрики продуктивності
│   │   ├── retraining.py        # Донавчання моделей
│   ├── pipelines/            # Конвеєри
│   │   ├── preprocess_pipeline.py     # Конвеєр попередньої обробки
│   │   ├── doc_analysis_pipeline.py   # Конвеєр обробки документів в графи
│   │   ├── generate_variations_pipeline.py # Конвеєр генерації графів для навчання
│   │   ├── clustering_pipeline.py     # Конвеєр кластеризації
│   │   ├── train_pipeline.py          # Конвеєр навчання
│   │   ├── retrain_pipeline.py        # Конвеєр донавчання
│   │   ├── reinforcement_pipeline.py  # Конвеєр RL навчання
│   │   ├── evaluate_pipeline.py       # Конвеєр оцінки моделей
│   ├── modes/                   # Режими запуску проекту
│   │   ├── preparation_mode.py  # Режим підготовки даних для навчання
│   │   ├── learning_mode.py     # Режим навчання моделей
│   │   ├── analityc_mode.py     # Режим аналізу та тестування моделей
│   ├── data_sources/         # Завантаження даних
│   │   ├── mssql_connector.py     # Підключення до MSSQL
│   │   ├── rest_api_connector.py  # Підключення до REST API
│   ├── utils/                # Допоміжні функції
│   │   ├── file_utils.py         # Робота з файлами верхнерівневі
│   │   ├── file_utils_l.py       # Деякі функції низькорівневі
│   │   ├── graph_creator.py      # Створення графа з BPMN XML та BPMS докумнтів 
│   │   ├── graph_utils.py        # Базові функції роботи з графом 
│   │   ├── graph_variations.py   # Функції зміни графа як норм варіації, так і аномальні
│   │   ├── logger.py             # Логування
│   │   ├── string_utils.py       # строкові функції
│   │   ├── visualizer.py         # Функції для візуалізації результатів
│   ├── config/               # Конфігураційні файли
│   │   ├── config.py             # Загальна конфігурація
│   │   ├── secrets.py            # Секрети
├── tests/                    # Тести
│   ├── test_data_processing.py   # Тести обробки даних
│   ├── test_models.py            # Тести моделей
├── requirements.txt          # Залежності
├── README.md                 # Документація
└── main.py                   # Головна точка запуску```

---

### Діаграма архітектури
![Архітектура проєкту](diagram.png)


---

## Основні залежності

Проєкт потребує таких бібліотек:

```plaintext
pandas
numpy
matplotlib
tensorflow
torch
networkx
scikit-learn
pyodbc
requests
gym
hdbscan
```
Для візуалізації графів використовується graphviz
Також потрібен Microsoft Visual C++ 14.0 or greater with "Microsoft C++ Build Tools

---

## Як розпочати

1. **Клонувати репозиторій:**
   ```bash
   git clone <repo_url>
   ```

2. **Встановити залежності:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Запустити головний файл:**
   ```bash
   python main.py
   ```

---

## Можливості для розвитку
1. Інтеграція Reinforcement Learning для навчання моделі на основі знайдених аномалій.
2. Розширення роботи з даними через REST API.
3. Додавання нових графових алгоритмів для GNN.
4. Покращення метрик оцінки результатів.

