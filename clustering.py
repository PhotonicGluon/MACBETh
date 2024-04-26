# IMPORTS
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import trange

# CONSTANTS
DIMENSIONALITY_REDUCTION_METHOD = "tsne"


# HELPER FUNCTIONS
def _scale_coordinates(coordinates: np.ndarray[float]) -> np.ndarray[float]:
    """
    Scales the coordinates.

    :param coordinates: coordinates to scale.
    :returns: scaled coordinates using the `StandardScaler`.
    """

    scaler = StandardScaler()
    return scaler.fit_transform(coordinates)


def _reduce_dimensionality(coordinates: np.ndarray[float], ndim: int = 3, method: str = DIMENSIONALITY_REDUCTION_METHOD) -> np.ndarray[float]:
    """
    Reduces the dimensionality of the coordinates.

    :param coordinates: coordinates to scale.
    :param ndim: number of dimensions for the resulting coordinates.
    :param method: method for dimensionality reduction. Valid methods are 'pca', 'tsne', 'mds', and 'isomap'.
    :returns: reduced coordinates.
    """

    valid_methods = {"pca", "tsne", "mds", "isomap"}

    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}' for dimensionality reduction")
    
    if method == "pca":
        return PCA(n_components=ndim).fit_transform(coordinates)
    if method == "tsne":
        return TSNE(n_components=ndim).fit_transform(coordinates)
    if method == "mds":
        return MDS(n_components=ndim).fit_transform(coordinates)
    if method == "isomap":
        return Isomap(n_components=ndim).fit_transform(coordinates)


# MAIN FUNCTIONS
def get_labels_using_kmeans(coordinates: np.ndarray[float], k_max: int = 10, num_clusters_override: int = -1) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using K-means clustering.
    
    :param coordinates: coordinates to label.
    :param k_max: Maximum `k` value for use in the K-means model.
    :param num_cluters_override: override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. If -1 then the number of clusters will be determined automatically.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """
        
    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(coordinates)
    k_max = min(k_max, len(scaled_coordinates))

    if num_clusters_override != -1:
        k_model = KMeans(
            n_clusters=num_clusters_override,
            init="random",
            n_init=25,  # Try initialising the model 25 times and get the best result
            max_iter=500,
            random_state=42
        )
        best_labels = k_model.fit_predict(scaled_coordinates)
    else:
        best_labels = None
        best_score = 0
        for k in trange(2, k_max+1, desc="Finding best number of clusters"):
            k_model = KMeans(
                n_clusters=k,
                init="random",
                n_init=25,  # Try initialising the model 25 times and get the best result
                max_iter=500,
                random_state=42
            )
            labels = k_model.fit_predict(scaled_coordinates)
            score = silhouette_score(scaled_coordinates, labels, metric="euclidean")

            if score > best_score:
                best_labels = labels
                best_score = score
    
    unique_labels = np.unique(best_labels)

    return reduced_coordinates, best_labels, unique_labels


def get_labels_using_dbscan(coordinates: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    
    :param coordinates: coordinates to label.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """
    
    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(coordinates)

    dbscan = DBSCAN(
        eps=20,
        min_samples=5,
        n_jobs=None  # Equivalent to 1 job, but can increase for parallel processing
    )
    labels = dbscan.fit_predict(scaled_coordinates)
    unique_labels = np.unique(labels)

    return reduced_coordinates, labels, unique_labels


def get_labels_using_optics(coordinates: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using Ordering Points To Identify the Clustering Structure (OPTICS).
    
    :param coordinates: coordinates to label.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """

    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(coordinates)

    optics = OPTICS(
        min_samples=10,
        xi=0.0125,
        min_cluster_size=25,
        n_jobs=None  # Equivalent to 1 job, but can increase for parallel processing
    )
    labels = optics.fit_predict(scaled_coordinates)
    unique_labels = np.unique(labels)

    return reduced_coordinates, labels, unique_labels


def get_labels_using_gmm(coordinates: np.ndarray[float], max_clusters=10, num_clusters_override: int = -1) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using a Gaussian Mixture Model (GMM).
    
    :param coordinates: coordinates to label.
    :param max_clusters: maximum number of clusters to consider. Must be at least 2.
    :param num_cluters_override: override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. If -1 then the number of clusters will be determined automatically.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """
    
    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(coordinates)
    max_clusters = min(max_clusters, len(scaled_coordinates))

    if num_clusters_override != -1:
        gmm = GaussianMixture(
            n_components=num_clusters_override,
            init_params="kmeans",
            max_iter=100,
            random_state=42
        )
        best_labels = gmm.fit_predict(scaled_coordinates)
    else:
        best_labels = None
        best_score = 0
        for n in trange(2, max_clusters+1, desc="Finding best number of clusters"):
            gmm = GaussianMixture(
                n_components=n,
                init_params="kmeans",
                n_init=10,
                max_iter=100,
                random_state=42
            )
            labels = gmm.fit_predict(scaled_coordinates)
            score = silhouette_score(scaled_coordinates, labels, metric="euclidean")

            if score > best_score:
                best_labels = labels
                best_score = score
    
    unique_labels = np.unique(best_labels)

    return reduced_coordinates, best_labels, unique_labels


def get_labels(coordinates: np.ndarray[float], method: str = "kmeans", num_cluters_override: int = -1) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates.
    
    :param coordinates: coordinates to label.
    :param method: method to generate the labels. Valid methods are 'kmeans', 'dbscan', 'optics', and 'gmm'.
    :param num_cluters_override: override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. If -1 then the number of clusters will be determined automatically.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """
    
    valid_methods = {"kmeans", "dbscan", "optics", "gmm"}

    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}' for clustering")

    if method == "kmeans":
        return get_labels_using_kmeans(coordinates, num_clusters_override=num_cluters_override)
    if method == "dbscan":
        return get_labels_using_dbscan(coordinates)
    if method == "optics":
        return get_labels_using_optics(coordinates)
    if method == "gmm":
        return get_labels_using_gmm(coordinates, num_clusters_override=num_cluters_override)


# MAIN CODE
if __name__ == "__main__":
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from clustering import get_labels
    from misc import load_data
    from preprocessing import get_unigrams_from_report, get_top_unigrams, unigram_list_to_coordinates

    sns.set_theme()

    data = load_data()

    # Process the data as coordinates
    unigram_counts = defaultdict(int)
    unigram_lists = []
    for report in data:
        unigrams = get_unigrams_from_report(report)
        unigram_lists.append(unigrams)
        for unigram in unigrams:
            unigram_counts[unigram] += 1

    top_unigrams = get_top_unigrams(unigram_counts, num=1000, seed=42)
    coordinates = np.array([unigram_list_to_coordinates(unigram_list, top_unigrams) for unigram_list in unigram_lists])

    # Get the labels
    reduced_coordinates, labels, unique_labels = get_labels(coordinates, method="gmm", num_cluters_override=7)

    # Plot the coordinates
    for unique_label in unique_labels:
        plt.scatter(reduced_coordinates[labels == unique_label, 0], reduced_coordinates[labels == unique_label, 1], label=unique_label)
    plt.legend()
    plt.show()
