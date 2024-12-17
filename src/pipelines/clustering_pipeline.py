# Конвеєр для кластеризації
def run_clustering_pipeline(data, eps, min_samples):
    from src.core.clustering import clusterize
    from src.utils.visualizer import visualize_clusters

    clusters = clusterize(data, eps=eps, min_samples=min_samples)
    visualize_clusters(data, clusters)
    return clusters