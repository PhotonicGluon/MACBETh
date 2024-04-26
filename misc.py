# IMPORTS
import glob
import json
from typing import List


# FUNCTIONS
def load_data(data_folder: str = "data/json") -> List[dict]:
    """
    Loads all the JSON files from the data folder.

    :param data_folder: folder that contains the JSON files.
    :returns: a list of the JSON data.
    """

    all_json_files = list(glob.glob(f"{data_folder}/*.json"))
    data = []
    for file in all_json_files:
        with open(file, "r") as f:
            data.append(json.load(f))
    return data
