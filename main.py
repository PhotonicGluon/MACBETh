# IMPORTS
from argparse import ArgumentParser
from typing import Any

from clustering import VALID_CLUSTERING_METHODS, VALID_DIMENSIONALITY_REDUCING_METHODS


# ARGUMENT VALIDATION FUNCTIONS
def number_of_clusters(n: Any) -> int:
    """
    Checks and validates the value passed for the number of clusters.

    :param n: value provided.
    :raises ValueError: if the value is not an integer
    :raises ValueError: if the value is neither -1 nor greater than 0
    :return: the same value if it is valid
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

    :param cluster_method: clustering method to use
    :param dim_reduction_method: dimensionality reduction method to use
    :param num_cluters_override: override value for the number of clusters to use. Applicable only
    for 'kmeans' and 'gmm'. If -1 then the number of clusters will be determined automatically.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    from clustering import get_labels
    from misc import load_data, reports_to_coordinates

    sns.set_theme()

    data, _ = load_data()
    if len(data) == 0:
        print("No data found in data folder")
        return

    coordinates, _ = reports_to_coordinates(data, num_unigrams=1000, seed=42)

    # Get the labels
    reduced_coordinates, labels, unique_labels = get_labels(
        coordinates,
        dim_reduction_method=dim_reduction_method,
        cluster_method=cluster_method,
        num_cluters_override=num_cluters_override,
    )

    # Plot the coordinates
    for unique_label in unique_labels:
        plt.scatter(
            reduced_coordinates[labels == unique_label, 0],
            reduced_coordinates[labels == unique_label, 1],
            label=unique_label,
        )
    plt.legend()
    plt.show()


# PARSERS
parser = ArgumentParser(description="MACBETH: Malware Analysis and Classification Based on Entries in Threat Hunting")
subparsers = parser.add_subparsers(help="perform clustering or classification", dest="command", required=True)

parser_clustering = subparsers.add_parser("cluster", help="clustering submenu")
parser_clustering.add_argument(
    "--cluster-method",
    "-m",
    help="clustering method",
    choices=VALID_CLUSTERING_METHODS,
    default="kmeans",
    type=str.lower,
)
parser_clustering.add_argument(
    "--reduction-method",
    "-r",
    help="dimensionality reduction method",
    choices=VALID_DIMENSIONALITY_REDUCING_METHODS,
    default="pca",
    type=str.lower,
)
parser_clustering.add_argument(
    "--number-of-clusters",
    "-n",
    help="override value for the number of clusters to use. Applicable only for 'kmeans' and 'gmm'. "
    "If -1 then the number of clusters will be determined automatically.",
    default=-1,
    type=number_of_clusters,
)

args = parser.parse_args()

if args.command == "cluster":
    clustering(args.cluster_method, args.reduction_method, args.number_of_clusters)
