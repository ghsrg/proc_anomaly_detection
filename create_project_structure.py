import os

def create_project_structure(base_path):
    # Список каталогів для створення
    directories = [
        "data/raw",
        "data/processed",
        "data/outputs",
        "data/anomalies",
        "notebooks",
        "src/core",
        "src/pipelines",
        "src/data_sources",
        "tests"
    ]

    # Список файлів для створення
    files = {
        "README.md": "# Проєкт: Аналіз аномалій у бізнес-процесах\n",
        "requirements.txt": "pandas\nnumpy\nmatplotlib\ntensorflow\ntorch\nnetworkx\nscikit-learn\npyodbc\nrequests\ngym\n",
        "main.py": "if __name__ == '__main__':\n    print('Проєкт успішно запущено!')\n",
        "src/core/data_processing.py": "import pandas as pd\n\n# Завантаження та обробка даних\n",
        "src/core/graph_processing.py": "# Побудова графа для GNN\n",
        "src/core/autoencoder.py": "# Реалізація автоенкодера\n",
        "src/core/gnn.py": "# Реалізація GNN\n",
        "src/core/clustering.py": "# Кластеризація\n",
        "src/core/retraining.py": "# Перенавчання моделей\n",
        "src/core/reinforcement.py": "# Реалізація Reinforcement Learning для навчання\n",
        "src/pipelines/preprocess_pipeline.py": "# Конвеєр для попередньої обробки даних\n",
        "src/pipelines/clustering_pipeline.py": "# Конвеєр для кластеризації\n",
        "src/pipelines/retrain_pipeline.py": "# Конвеєр для перенавчання моделей\n",
        "src/pipelines/reinforcement_pipeline.py": "# Конвеєр для RL навчання\n",
        "src/pipelines/evaluate_pipeline.py": "# Конвеєр для оцінки моделей\n",
        "src/data_sources/mssql_connector.py": "# Завантаження даних із MSSQL\n",
        "src/data_sources/rest_api_connector.py": "# Завантаження даних через REST API\n",
        "tests/test_data_processing.py": "# Тести для обробки даних\n",
        "tests/test_models.py": "# Тести для моделей\n",
        "tests/test_reinforcement.py": "# Тести RL компонентів\n"
    }

    # Створення каталогів
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Створено каталог: {dir_path}")

    # Створення файлів
    for file, content in files.items():
        file_path = os.path.join(base_path, file)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Створено файл: {file_path}")

# Вкажіть базовий шлях до проєкту
project_base_path = "./proc_anomaly_detection"

# Запустити функцію створення структури
create_project_structure(project_base_path)
