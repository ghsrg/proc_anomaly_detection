# Експериментальний режим
from src.pipelines.preprocess_pipeline import run_preprocess_pipeline
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.evaluate_pipeline import evaluate_clusters


def run_experimental_mode():
    print("Запуск експериментального режиму...")

    # Попередня обробка
    data = run_preprocess_pipeline()

    # Кластеризація з різними параметрами
    for eps in [0.2, 0.5, 1.0]:
        clusters = run_clustering_pipeline(data, eps=eps, min_samples=10)
        evaluate_clusters(clusters)
