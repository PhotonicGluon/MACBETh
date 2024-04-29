# IMPORTS
import glob
import json
from typing import List, Tuple

from tqdm import tqdm


# FUNCTIONS
def load_data(data_folder: str = "data/json") -> Tuple[List[dict], List[str]]:
    """
    Loads all the JSON files from the data folder.

    :param data_folder: folder that contains the JSON files, defaults to "data/json"
    :return: a list of the JSON data
    :return: a list of the file paths of the JSON files that the files come from
    """

    all_json_files = list(glob.glob(f"{data_folder}/*.json"))
    data = []
    for file in tqdm(all_json_files, desc="Loading data"):
        with open(file, "r") as f:
            data.append(json.load(f))
    return data, all_json_files
