# Експериментальний режим

from src.pipelines.preprocess_pipeline import run_preprocess_pipeline
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.evaluate_pipeline import evaluate_clusters
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_experimental_mode(reload):
    print("Запуск експериментального режиму...")

    # Попередня обробка
    if reload:
        logger.info("Завантаження нових даних.")
        data = run_preprocess_pipeline()

    # Кластеризація з різними параметрами
   # for eps in [0.2, 0.5, 1.0]:
    #    clusters = run_clustering_pipeline(data, eps=eps, min_samples=10)
   #     evaluate_clusters(clusters)
