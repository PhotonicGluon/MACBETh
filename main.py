# IMPORTS
from argparse import ArgumentParser

from clustering import VALID_CLUSTERING_METHODS, VALID_DIMENSIONALITY_REDUCING_METHODS


# ARGUMENT VALIDATION FUNCTIONS
def number_of_clusters(n) -> int:
    """
    Checks and validates the value passed for the number of clusters.

    :param n: valid provided.
    :returns: the same value if it is valid.
    :raises ValueError: if the value is not valid.
    """

    try:
        n = int(n)
    except ValueError:
        raise ValueError
    
    if n == -1 or n > 0:
        return n

    raise ValueError


# MAIN FUNCTIONS
def clustering(cluster_method: str, dim_reduction_method: str, num_cluters_override: int):
    """
    Performs clustering.

    :param cluster_method: clustering method to use.
    :param dim_reduction_method: dimensionality reduction method to use.
    :param num_clusters_override: override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. If -1 then the number of clusters will be determined automatically.
    """

    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from clustering import get_labels
    from misc import load_data, get_unigrams_from_report, get_top_unigrams, unigram_list_to_coordinates

    sns.set_theme()
    data = load_data()

    if len(data) == 0:
        print("No data found in data folder")
        return

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
    reduced_coordinates, labels, unique_labels = get_labels(coordinates, dim_reduction_method=dim_reduction_method, cluster_method=cluster_method, num_cluters_override=num_cluters_override)

    # Plot the coordinates
    for unique_label in unique_labels:
        plt.scatter(reduced_coordinates[labels == unique_label, 0], reduced_coordinates[labels == unique_label, 1], label=unique_label)
    plt.legend()
    plt.show()


# PARSERS
parser = ArgumentParser(description="MACBETH: Malware Analysis and Classification Based on Entries in Threat Hunting")
subparsers = parser.add_subparsers(help="perform clustering or classification", dest="command", required=True)

parser_clustering = subparsers.add_parser("cluster", help="clustering submenu")
parser_clustering.add_argument("--cluster-method", "-m", help="clustering method", choices=VALID_CLUSTERING_METHODS, default="kmeans", type=str.lower)
parser_clustering.add_argument("--reduction-method", "-r", help="dimensionality reduction method", choices=VALID_DIMENSIONALITY_REDUCING_METHODS, default="pca", type=str.lower)
parser_clustering.add_argument("--number-of-clusters", "-n", help="override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. "
                               "If -1 then the number of clusters will be determined automatically.", default=-1, type=number_of_clusters)

args = parser.parse_args()

if args.command == "cluster":
    clustering(args.cluster_method, args.reduction_method, args.number_of_clusters)
