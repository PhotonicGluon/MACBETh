# IMPORTS
from collections import defaultdict
import json
import re
import random
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm


# FUNCTIONS
def get_unigrams_from_report(report: dict) -> List[str]:
    """
    Gets the unigrams from the report.

    :param report: the dictionary to process. Expects to have the 'additional_info' key.
    :raises AttributeError: if `additional_info` is not found in the report.
    :returns: a list of the unigrams from the report.
    """

    if "additional_info" not in report:
        raise AttributeError("`additional_info` not found in provided report")

    behaviours_data = json.dumps(report["additional_info"])

    # Remove any newlines and excess whitespaces
    behaviours_data = behaviours_data.replace("\n", " ")
    behaviours_data = re.sub(r"\s+", " ", behaviours_data)

    # Remove any hashes and versioning information
    # TODO: Should we keep the versioning though?
    behaviours_data = re.sub(r"[A-Fa-f\d]{32,128}", "", behaviours_data)
    behaviours_data = re.sub(r"\d+\.\d+", "", behaviours_data)

    # Now we can split by whitespace; each one of these things will be called a unigram
    unigrams = list(set(filter(None, behaviours_data.split())))  # Also remove any strings that are equivalent to empty

    # We only want the things within strings
    pattern = re.compile(r"^\"|\",?:?$|,$")
    new_unigrams = []
    for unigram in unigrams:
        if pattern.search(unigram):
            new_unigrams.append(pattern.sub("", unigram))
    
    new_unigrams = [unigram.strip() for unigram in new_unigrams]
    final_unigrams = [unigram for unigram in new_unigrams if len(unigram) > 3]

    return final_unigrams


def get_top_unigrams(unigram_counts: Dict[str, int], num=1000, seed=-1) -> List[str]:
    """
    Gets the top `num` unigrams.

    :param unigram_counts: dictionay of the counts of the unigrams.
    :param num: number of unigrams to find.
    :param seed: the seed for randomising the output list. If `-1` then no randomisation will be performed.
    :returns: a list of the top `num` unigrams, possibly randomised.
    """

    unigrams = [(key, val) for key, val in unigram_counts.items()]

    # To fix issues with reproducibility, we first sort by the unigram and then by the count
    unigrams.sort(reverse=True, key=lambda elem: elem[0])  # Sort based on unigram
    unigrams.sort(reverse=True, key=lambda elem: elem[1])  # Sort based on count

    unigrams = unigrams[:num]
    top_unigrams = [unigram for unigram, _ in unigrams]
    top_unigrams = top_unigrams.copy()

    if seed != -1:
        random.Random(seed).shuffle(top_unigrams)
    
    return top_unigrams


def unigram_list_to_coordinates(unigram_list: List[str], top_unigrams: List[str]) -> List[int]:
    """
    Converts a list of unigrams into a coordinate.
    The dimensionality of the coordinate is the same as the length of the `top_unigrams` list.
    More accurately, this is a bit string where the coordinates can only be 0 (which means that the top unigram is not
    present) or 1 (which means that the top unigram is present).

    :param unigram_list: list of unigrams to convert into coordinates.
    :param top_unigrams: list of the top unigrams.
    :returns: coordinate.
    """

    unigram_set = set(unigram_list)  # For faster lookup
    coordinates = [int(top_unigram in unigram_set) for top_unigram in top_unigrams]
    return coordinates


def reports_to_coordinates(data: List[dict], num_unigrams=1000, seed=-1) -> Tuple[np.ndarray, List[str]]:
    """
    Converts all the reports into coordinates.

    :param data: data array containing the reports.
    :param num_unigrams: number of unigrams to include in the coordinates.
    :param seed: the seed for randomising the output list. If `-1` then no randomisation will be performed.
    :return: an array of coordinates. The size of the array will be `(len(data), num_unigrams)`.
    :return: the list of the top `num_unigrams` unigrams, in the same order as the array of coordinates,
    """

    unigram_counts = defaultdict(int)
    unigram_lists = []
    for report in tqdm(data, desc="Converting reports to coordinates"):
        unigrams = get_unigrams_from_report(report)
        unigram_lists.append(unigrams)
        for unigram in unigrams:
            unigram_counts[unigram] += 1

    top_unigrams = get_top_unigrams(unigram_counts, num=num_unigrams, seed=seed)
    coordinates = np.array([unigram_list_to_coordinates(unigram_list, top_unigrams) for unigram_list in unigram_lists])

    return coordinates, top_unigrams
