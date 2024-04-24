# IMPORTS
import glob
import json
from typing import List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from clustering import get_best_kmeans
from preprocessing import get_unigrams_from_report, get_top_unigrams, unigram_list_to_coordinates

# FUNCTIONS
def load_data(data_folder: str = "data/json") -> List[dict]:
    """
    Loads all the JSON files from the data folder.

    :param data_folder: folder that contains the JSON files.
    :returns: a list of the JSON data.
    """

    all_json_files = list(glob.glob("data/json/*.json"))
    data = []
    for file in all_json_files:
        with open(file, "r") as f:
            data.append(json.load(f))
    return data


def main():
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

    # Use K-Means clustering
    k_best, k_model = get_best_kmeans(coordinates, k_max=10)
    
    # Reduce dimensionality
    pca = PCA(2)  # 2D for plotting
    reduced_coordinates = pca.fit_transform(coordinates)

    # Cluster in the reduced dimension space
    reduced_dim_k_model = KMeans(n_clusters=k_best, max_iter=1000)
    reduced_dim_k_model.fit(reduced_coordinates)
    labels = reduced_dim_k_model.predict(reduced_coordinates)

    # Now visualise
    unique_labels = np.unique(k_model.labels_)
    for unique_label in unique_labels:
        plt.scatter(reduced_coordinates[labels == unique_label, 0], reduced_coordinates[labels == unique_label, 1], label=unique_label)
    plt.legend()
    plt.show()


# MAIN
if __name__ == "__main__":
    main()
