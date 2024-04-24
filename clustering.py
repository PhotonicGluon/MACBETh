# IMPORTS
from typing import List, Tuple

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# HELPER FUNCTIONS
def get_best_kmeans(data: List[int], k_max: int = 10) -> Tuple[int, KMeans]:
    """
    Finds the best `k` value for the K-means algorithm.

    :param data: the data to use for K-means clustering.
    :param k_max: maximum value of `k` to consider.
    :returns: the best `k` value for K-means clustering.
    :returns: the best model that uses the `k` value for clustering.
    """

    # First scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    k_max = min(k_max, len(scaled_features))

    # Try clustering with different values of k
    kmodels = []
    for k in range(1, k_max+1):
        kmodel = KMeans(
            n_clusters=k,
            init="random",
            n_init=10,  # Try initialising the model 10 times and get the best result
            max_iter=500,
            random_state=42
        )
        kmodel.fit(scaled_features)
        kmodels.append(kmodel)
    sse = [x.inertia_ for x in kmodels]

    # Use the Elbow method to find the best number if clusters to use
    kneelocator = KneeLocator(
        range(1, k_max+1), sse, curve="convex", direction="decreasing"
    )
    k_best = kneelocator.elbow
    if k_best is None:  # Can't find a good cluster number
        k_best = k_max
    
    best_model = kmodels[k_best-1]

    return k_best, best_model


def get_labels_using_kmeans(coordinates: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates using K-means clustering.
    
    :param coordinates: coordinates to label.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """
        
    # Use K-Means clustering
    k_best, k_model = get_best_kmeans(coordinates, k_max=10)
    
    # Reduce dimensionality
    pca = PCA(2)  # 2D for plotting
    reduced_coordinates = pca.fit_transform(coordinates)

    # Cluster in the reduced dimension space
    reduced_dim_k_model = KMeans(n_clusters=k_best, max_iter=1000)
    reduced_dim_k_model.fit(reduced_coordinates)
    labels = reduced_dim_k_model.predict(reduced_coordinates)
    unique_labels = np.unique(k_model.labels_)

    return reduced_coordinates, labels, unique_labels


# MAIN FUNCTIONS
def get_labels(coordinates: np.ndarray[float], method: str = "kmeans") -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Gets the labels for each of the coordinates.
    
    :param coordinates: coordinates to label.
    :param method: method to generate the labels. Valid methods are 'kmeans'.
    :return: the reduced coordinates for displaying.
    :return: the labels for each of the reduced coordinates.
    :return: the unique labels.
    """

    method = method.lower()
    if method not in {"kmeans"}:
        raise ValueError(f"Invalid method '{method}'")

    if method == "kmeans":
        return get_labels_using_kmeans(coordinates)
