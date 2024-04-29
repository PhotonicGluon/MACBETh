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
VALID_CLUSTERING_METHODS = {"kmeans", "dbscan", "optics", "gmm"}
VALID_DIMENSIONALITY_REDUCING_METHODS = {"pca", "tsne", "mds", "isomap"}


# HELPER FUNCTIONS
def _scale_coordinates(coordinates: np.ndarray[float]) -> np.ndarray[float]:
    """
    Scales the coordinates.

    :param coordinates: coordinates to scale
    :return: scaled coordinates using the `StandardScaler`
    """

    scaler = StandardScaler()
    return scaler.fit_transform(coordinates)


def _reduce_dimensionality(
    coordinates: np.ndarray[float], ndim: int = 2, method: str = "pca"
) -> np.ndarray[float]:
    """
    Reduces the dimensionality of the coordinates.

    :param coordinates: coordinates to scale
    :param ndim: number of dimensions for the resulting coordinates, defaults to 2
    :param method: method for dimensionality reduction, valid methods are "pca", "tsne", "mds", and
    "isomap", defaults to "pca"
    :raises ValueError: if the method is not recongised
    :return: reduced coordinates
    """

    method = method.lower()
    if method not in VALID_DIMENSIONALITY_REDUCING_METHODS:
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
def get_labels_using_kmeans(
    coordinates: np.ndarray[float],
    dim_reduction_method: str = "pca",
    k_max: int = 10,
    k_override: int = -1,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using K-means clustering.

    :param coordinates: coordinates to label
    :param dim_reduction_method: dimensionality reduction method, defaults to "pca"
    :param k_max: Maximum `k` value for use in the K-means model
    :param k_override: override value for the number of clusters to use. Applicable only for
    "kmeans" and "gmm". If -1 then the number of clusters will be determined automatically
    :return: the reduced coordinates for displaying
    :return: the labels for each of the reduced coordinates
    :return: the unique labels
    """ 

    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(
        coordinates, method=dim_reduction_method
    )
    k_max = min(k_max, len(scaled_coordinates))

    if k_override != -1:
        k_model = KMeans(
            n_clusters=k_override,
            init="random",
            n_init=25,  # Try initialising the model 25 times and get the best result
            max_iter=500,
            random_state=42,
        )
        best_labels = k_model.fit_predict(scaled_coordinates)
    else:
        best_labels = None
        best_score = 0
        for k in trange(2, k_max + 1, desc="Finding best number of clusters"):
            k_model = KMeans(
                n_clusters=k,
                init="random",
                n_init=25,  # Try initialising the model 25 times and get the best result
                max_iter=500,
                random_state=42,
            )
            labels = k_model.fit_predict(scaled_coordinates)
            score = silhouette_score(scaled_coordinates, labels, metric="euclidean")

            if score > best_score:
                best_labels = labels
                best_score = score

    unique_labels = np.unique(best_labels)

    return reduced_coordinates, best_labels, unique_labels


def get_labels_using_dbscan(
    coordinates: np.ndarray[float], dim_reduction_method: str = "pca"
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using Density-Based Spatial Clustering of
    Applications with Noise (DBSCAN).

    :param coordinates: coordinates to label
    :param dim_reduction_method: dimensionality reduction method, defaults to "pca"
    :return: the reduced coordinates for displaying
    :return: the labels for each of the reduced coordinates
    :return: the unique labels
    """

    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(
        coordinates, method=dim_reduction_method
    )

    dbscan = DBSCAN(
        eps=20,
        min_samples=5,
        n_jobs=None,  # Equivalent to 1 job, but can increase for parallel processing
    )
    labels = dbscan.fit_predict(scaled_coordinates)
    unique_labels = np.unique(labels)

    return reduced_coordinates, labels, unique_labels


def get_labels_using_optics(
    coordinates: np.ndarray[float], dim_reduction_method: str = "pca"
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using Ordering Points To Identify the Clustering
    Structure (OPTICS).

    :param coordinates: coordinates to label
    :param dim_reduction_method: dimensionality reduction method, defaults to "pca"
    :return: the reduced coordinates for displaying
    :return: the labels for each of the reduced coordinates
    :return: the unique labels
    """

    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(
        coordinates, method=dim_reduction_method
    )

    optics = OPTICS(
        min_samples=10,
        xi=0.0125,
        min_cluster_size=25,
        n_jobs=None,  # Equivalent to 1 job, but can increase for parallel processing
    )
    labels = optics.fit_predict(scaled_coordinates)
    unique_labels = np.unique(labels)

    return reduced_coordinates, labels, unique_labels


def get_labels_using_gmm(
    coordinates: np.ndarray[float],
    dim_reduction_method: str = "pca",
    max_clusters=10,
    num_clusters_override: int = -1,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using a Gaussian Mixture Model (GMM).

    :param coordinates: coordinates to label
    :param dim_reduction_method: dimensionality reduction method, defaults to "pca"
    :param max_clusters: maximum number of clusters to consider; must be at least 2
    :param num_cluters_override: override value for the number of clusters to use; aplicable only
    for "kmeans" and "gmm"; if -1 then the number of clusters will be determined automatically,
    defaults to -1
    :return: the reduced coordinates for displaying
    :return: the labels for each of the reduced coordinates
    :return: the unique labels
    """

    scaled_coordinates = _scale_coordinates(coordinates)
    reduced_coordinates = _reduce_dimensionality(
        coordinates, method=dim_reduction_method
    )
    max_clusters = min(max_clusters, len(scaled_coordinates))

    if num_clusters_override != -1:
        gmm = GaussianMixture(
            n_components=num_clusters_override,
            init_params="kmeans",
            max_iter=100,
            random_state=42,
        )
        best_labels = gmm.fit_predict(scaled_coordinates)
    else:
        best_labels = None
        best_score = 0
        for n in trange(2, max_clusters + 1, desc="Finding best number of clusters"):
            gmm = GaussianMixture(
                n_components=n,
                init_params="kmeans",
                n_init=10,
                max_iter=100,
                random_state=42,
            )
            labels = gmm.fit_predict(scaled_coordinates)
            score = silhouette_score(scaled_coordinates, labels, metric="euclidean")

            if score > best_score:
                best_labels = labels
                best_score = score

    unique_labels = np.unique(best_labels)

    return reduced_coordinates, best_labels, unique_labels


def get_labels(
    coordinates: np.ndarray[float],
    cluster_method: str = "kmeans",
    dim_reduction_method: str = "pca",
    num_cluters_override: int = -1,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates

    :param coordinates: coordinates to label
    :param cluster_method: method to generate the labels. Valid methods are "kmeans", "dbscan",
    "optics", and "gmm", defaults to "kmeans"
    :param dim_reduction_method: method for dimensionality reduction. Valid methods are "pca",
    "tsne", "mds", and "isomap", defaults to "pca"
    :param num_cluters_override: override value for the number of clusters to use; applicable only
    for "kmeans" and "gmm"; if -1 then the number of clusters will be determined automatically;
    defaults to -1
    :return: the reduced coordinates for displaying
    :return: the labels for each of the reduced coordinates
    :return: the unique labels
    """

    cluster_method = cluster_method.lower()
    if cluster_method not in VALID_CLUSTERING_METHODS:
        raise ValueError(f"Invalid method '{cluster_method}' for clustering")

    if cluster_method == "kmeans":
        return get_labels_using_kmeans(
            coordinates,
            dim_reduction_method=dim_reduction_method,
            k_override=num_cluters_override,
        )
    if cluster_method == "dbscan":
        return get_labels_using_dbscan(
            coordinates, dim_reduction_method=dim_reduction_method
        )
    if cluster_method == "optics":
        return get_labels_using_optics(
            coordinates, dim_reduction_method=dim_reduction_method
        )
    if cluster_method == "gmm":
        return get_labels_using_gmm(
            coordinates,
            dim_reduction_method=dim_reduction_method,
            num_clusters_override=num_cluters_override,
        )
