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
│   ├── outputs/              # Результати моделей
│   ├── anomalies/            # Виділені аномалії
├── reports/                  # Звіти для наукової роботи
├── logs
│   ├── app.log               # Логування роботи системи
│   ├── error.log             # Логування помилок
├── src/                      # Основний код
│   ├── core/                 # Центральна логіка
│   │   ├── data_processing.py    # 🟢 Завантаження та обробка даних
│   │   ├── clustering.py         # 🟢 Кластеризація
│   │   ├── graph_processing.py   # Побудова графів
│   │   ├── autoencoder.py        # Реалізація автоенкодера
│   │   ├── gnn.py                # Реалізація GNN
│   │   ├── metrics.py            # Метрики продуктивності
│   │   ├── visualizer.py         # Візуалізація результатів
│   │   ├── retraining.py         # Перенавчання моделей
│   ├── pipelines/            # Конвеєри
│   │   ├── preprocess_pipeline.py # Конвеєр попередньої обробки
│   │   ├── clustering_pipeline.py # Конвеєр кластеризації
│   │   ├── retrain_pipeline.py    # Конвеєр перенавчання
│   │   ├── reinforcement_pipeline.py # Конвеєр RL навчання
│   │   ├── evaluate_pipeline.py   # Конвеєр оцінки моделей
│   ├── modes/
│   │   ├── experimental_mode.py   # Експериментальний режим
│   │   ├── analytical_mode.py     # Аналітичний режим
│   │   ├── production_mode.py     # Режим виконання
│   ├── data_sources/         # Завантаження даних
│   │   ├── mssql_connector.py     # 🟢 Підключення до MSSQL
│   │   ├── rest_api_connector.py  # Підключення до REST API
│   ├── utils/                # Допоміжні функції
│   │   ├── file_utils.py         # Робота з файлами
│   │   ├── logger.py             # Логування
│   │   ├── visualizer.py         # Функції для візуалізації результатів
│   │   ├── metrics.py            # Реалізація метрик
│   │   ├── config_utils.py       # Конфігурація
│   │   ├── string_utils.py       # строкові функції
│   ├── config/               # Конфігураційні файли
│   │   ├── config.py             # Загальна конфігурація
│   │   ├── secrets.py
│   │   ├── params_experimental.py # Параметри експериментального режиму
│   │   ├── params_analytical.py   # Параметри аналітичного режиму
├── tests/                    # Тести
│   ├── test_data_processing.py   # Тести обробки даних
│   ├── test_models.py            # Тести моделей
├── scripts/                  # Скрипти запуску режимів
│   ├── run_experimental.py       # Запуск експериментального режиму
│   ├── run_analytical.py         # Запуск аналітичного режиму
│   ├── run_production.py         # Запуск продакшн режиму
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

