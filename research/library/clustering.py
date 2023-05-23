import warnings
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .correlations import plot_correlations, get_distance_matrix


def plot_correlation_matrix_clusters(Sigma: pd.DataFrame, labels: np.array, print_clusters: bool = False):
    columns = Sigma.columns
    assert len(columns) == len(labels)
    unique_labels = set(labels)
    assert min(unique_labels) == 0 and max(unique_labels) == len(unique_labels) - 1
    clusters = []
    for label in unique_labels:
        clusters.append(columns[labels == label])
    sorted_labels = sum(map(list, clusters), start=[])
    cumulative_clusters_size = 0
    if print_clusters:
        print('Clusters:')
    for cluster in clusters:
        plt.axvline(cumulative_clusters_size)
        plt.axhline(cumulative_clusters_size)
        if print_clusters:
            print(list(cluster))
        cumulative_clusters_size += len(cluster)
    plt.axvline(cumulative_clusters_size)
    plt.axhline(cumulative_clusters_size)
    return plot_correlations(Sigma, sorted_labels)


@dataclass
class ClusteringResult:
    silhouette_train_score: float
    silhouette_test_score: float
    clusters_labels: np.array


@dataclass
class OptimalClusteringResult:
    silhouette_train_scores: list[float]
    silhouette_test_scores: list[float]

    n_clusters_list: list[int]
    best_n_clusters: int
    best_clusters_labels: np.array


def find_optimal_clusters_number(Sigma_train: pd.DataFrame, Sigma_test: pd.DataFrame, n_clusters_list: list[int], model: str) -> OptimalClusteringResult:
    dist_matrix_train = get_distance_matrix(Sigma_train)
    dist_matrix_test = get_distance_matrix(Sigma_test)

    scores_train = []
    scores_test = []
    clusters_labels = []

    # Iterate over number of clusters and compute scores
    for n_clusters in n_clusters_list:
        results = _get_clustering_results(dist_matrix_train, dist_matrix_test, n_clusters, model)
        scores_train.append(results.silhouette_train_score)
        scores_test.append(results.silhouette_test_score)
        clusters_labels.append(results.clusters_labels)

    # Choose the best number of clusters on test
    best_n_clusters_ind = np.argmax(scores_test)
    return OptimalClusteringResult(
        silhouette_train_scores=scores_train,
        silhouette_test_scores=scores_test,
        n_clusters_list=n_clusters_list,
        best_n_clusters=n_clusters_list[best_n_clusters_ind],
        best_clusters_labels=clusters_labels[best_n_clusters_ind]
    )


def _get_clustering_results(dist_matrix_train: pd.DataFrame, dist_matrix_test: pd.DataFrame, n_clusters: int, model: str) -> ClusteringResult:
    assert model in ['kmeans', 'spectral']
    if model == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif model == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(dist_matrix_train)
    score_train = silhouette_score(dist_matrix_train, labels, metric='precomputed')
    score_test = silhouette_score(dist_matrix_test, labels, metric='precomputed')
    return ClusteringResult(score_train, score_test, labels)
