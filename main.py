# IMPORTS
import os
from argparse import ArgumentParser, ArgumentTypeError
from typing import Any

import json

from clustering import VALID_CLUSTERING_METHODS, VALID_DIMENSIONALITY_REDUCING_METHODS


# ARGUMENT VALIDATION FUNCTIONS
def number_of_clusters(n: Any) -> int:
    """
    Checks and validates the value passed for the number of clusters.

    :param n: value provided.
    :raises ArgumentTypeError: if the value is not an integer
    :raises ArgumentTypeError: if the value is neither -1 nor greater than 0
    :return: the same value if it is valid
    """

    try:
        n = int(n)
    except ValueError:
        raise ArgumentTypeError("number of clusters is not an integer")

    if n == -1 or n > 0:
        return n

    raise ArgumentTypeError("number of clusters must be -1 or a postive integer")


def report_file(path: Any) -> str:
    """
    Checks and validates the value passed for the report file.

    :param path: supposed path to the file
    :raises ArgumentTypeError: if no file was found at the specified path
    :raises ArgumentTypeError: if the file at the specified path does not appear to be a JSON file
    :raises ArgumentTypeError: if the JSON file at the specified path does not have the
    `additional_info` field
    :return: the same valid if it is valid
    """

    if not os.path.isfile(path):
        raise ArgumentTypeError("no file found at specified path")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if "additional_info" not in data:
                raise ArgumentTypeError("JSON file must have the 'additional_info' field")
    except json.JSONDecodeError:
        raise ArgumentTypeError("file at specified path does not appear to be a JSON file")

    return path


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
    print("Getting labels...")
    reduced_coordinates, labels, unique_labels = get_labels(
        coordinates,
        dim_reduction_method=dim_reduction_method,
        cluster_method=cluster_method,
        num_cluters_override=num_cluters_override,
    )

    # Plot the coordinates
    print("Plotting coordinates...")
    for unique_label in unique_labels:
        plt.scatter(
            reduced_coordinates[labels == unique_label, 0],
            reduced_coordinates[labels == unique_label, 1],
            label=unique_label,
        )
    plt.legend()
    plt.show()


def classifier(path: str):
    """
    Performs classification.

    :param file: path to the VirusTotal report to classify
    """

    import pickle

    import numpy as np
    import keras

    from misc import get_unigrams_from_report, unigram_list_to_coordinates

    with open(path, "r") as f:
        report = json.load(f)

    # Convert the report into data to feed into the model
    try:
        with open("data/top_unigrams.txt", "r") as f:
            top_unigrams = [x.strip() for x in f.readlines()]
    except FileNotFoundError:
        print("The 'top_unigrams.txt' file cannot be found in the 'data' directory.")
        exit(1)

    unigrams = get_unigrams_from_report(report)
    coordinates = unigram_list_to_coordinates(unigrams, top_unigrams)

    # Load the model
    model_path = "models/combined/combined.keras"
    if not os.path.isfile(model_path):
        print("The 'combined.keras' file cannot be found in the 'models/combined' directory.")
        exit(1)

    model = keras.models.load_model(model_path)

    # Make a prediction based off the data
    prediction = np.argmax(model.predict(np.array([coordinates]))[0])
    print(prediction)

    # Convert this index back into a label using the label encoder
    try:
        with open("models/classifier/label-encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("The 'label-encoder.pkl' file cannot be found in the 'models/classifier' directory.")
        exit(1)

    pred_label = label_encoder.inverse_transform([prediction])[0]
    print(f"\nPredicted label for '{path}' is {pred_label}\n")


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

parser_classifier = subparsers.add_parser("classify", help="classifier submenu")
parser_classifier.add_argument("file", help="report file to classify", type=report_file)

args = parser.parse_args()

if args.command == "cluster":
    clustering(args.cluster_method, args.reduction_method, args.number_of_clusters)
else:
    classifier(args.file)
