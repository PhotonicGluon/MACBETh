# IMPORTS
import hashlib
import os
import json
import random
import re
import subprocess
import shutil
from typing import List, Optional


# HELPER FUNCTIONS
def _random_hash() -> str:
    """
    Generates a random hash.

    :return: random hash
    """

    return hashlib.sha256(str(random.random()).encode()).hexdigest()


def _run_avclass(*args: List[str]) -> str:
    """
    Runs the `avclass` command.

    :param args: arguments to pass into the `avclass` command
    :return: string output of the command
    """

    pipe = subprocess.Popen(["avclass"] + list(args), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output, _ = pipe.communicate()
    return output.decode("UTF-8").strip()


def _majority_rules_labelling(temp_file: str) -> Optional[str]:
    """
    Use majority rules for labelling the sample located in the temporary file.

    :param temp_file: temporary file to run family tagging on
    :returns: supposed label
    """

    # Use family tagging by considering the other AVs reports
    output = _run_avclass("-f", temp_file, "-t")  # -t means tag families
    splitted = output.split()

    if len(splitted) < 2 or not splitted[1].isdigit():
        label = "unknown"
    elif int(splitted[1]) <= 5:  # At most 5 vendors flagged it
        label = "benign"
    elif len(splitted) == 2:  # More than 5 vendors flagged it, but we only have the count
        label = "unknown"
    else:
        _, _, details = splitted

        # Find the first family (i.e., most popular family) that was tagged
        family_pattern = r"FAM:(?P<family>.+?)\|"
        match = re.search(family_pattern, details)

        if match is not None:
            label = match.group("family")
        else:
            # No families matched
            label = "unknown"

    return label


# FUNCTIONS
def label_sample(report_json: dict) -> str:
    """
    Tries to guess the label of the sample based off the VirusTotal report.

    :param report_json: the VirusTotal report
    :return: the guessed label. A label of "UNKNOWN" means that the malware sample's true label
    cannot be understood, while "BENIGN" means that the sample is (likely) harmless
    """

    temp_file = f"temp-{_random_hash()}.json"

    # We need to place the report in a JSON file for use by `avclass`
    with open(temp_file, "w") as f:
        json.dump(report_json, f)

    # Try to get the label using default methods
    output = _run_avclass("-f", temp_file)
    splitted = output.split()

    if len(splitted) == 2:
        _, label = splitted

        # If the label starts with "SINGLETON", we can't attribute this sample using the default way
        if label.startswith("SINGLETON"):
            label = _majority_rules_labelling(temp_file)
    else:
        label = _majority_rules_labelling(temp_file)

    os.remove(temp_file)

    # As a last resort, if it is a known distributor, it should be benign
    if label == "unknown" and "known-distributor" in report_json.get("tags", []):
        label = "benign"

    return label.upper()


# MAIN CODE
if __name__ == "__main__":
    from tqdm import tqdm

    os.makedirs("data/labelled-data", exist_ok=True)

    data_files = os.listdir("../data/json")

    for file in tqdm(data_files, desc="Labelling samples"):
        filepath = f"../data/json/{file}"

        with open(filepath, "r") as f:
            report = json.load(f)

        label = label_sample(report)

        if len(file.split("_")) == 2:
            hash_ = file.split("_")[1]
        else:
            hash_ = file
        shutil.copyfile(filepath, f"data/labelled-data/{label}_{hash_}")
