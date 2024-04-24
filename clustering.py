# IMPORTS
from typing import List, Tuple

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# FUNCTIONS
def get_best_kmeans(data: List[int], k_max=10) -> Tuple[int, KMeans]:
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
