# IMPORTS
import os
import time
import requests
import json
import urllib3

# SETUP
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# FUNCTIONS
def update_master_list(all_hashes):
    """
    Updates the master list of hashses.
    """

    all_hashes = sorted(list(all_hashes))
    
    with open("../data/hashes.txt", "w") as f:
        for hash_ in all_hashes:
            f.write(f"{hash_}\n")


def update_done_hashes():
    """
    Updates the list of processed hashes.
    """
    # Get all the JSON data from the files we downloaded
    data_files = os.listdir("../data/vt-data")
    all_json_data = []
    for file in data_files:
        with open(f"../data/vt-data/{file}", "r") as f:
            all_json_data.append(json.load(f))

    # Get all the hashes from the JSON data
    hashes = []
    for data in all_json_data:
        hashes.append(data["md5"])
        hashes.append(data["sha1"])
        hashes.append(data["sha256"])

    # Write to file
    with open("../data/done-hashes.txt", "w") as f:
        for hash_ in hashes:
            f.write(f"{hash_}\n")



def get_report(resource_hash, api_key, retry=5, delay=1):
    """
    Tries to get the VirusTotal report.
    Returns a tuple. The first value is a boolean on whether the request was successful or not. The second value is the
    report. May be `None`.
    """
    count = 1
    while count <= retry:
        print(f"\33[2K\rRetrieving '{resource_hash}' (Try {count}/{retry})", end="")
        try:
            r = requests.get(
                f"https://www.virustotal.com/vtapi/v2/file/report?apikey={api_key}&resource={resource_hash}&allinfo=1"
            )
        except requests.exceptions.SSLError:
            r = requests.get(
                f"https://www.virustotal.com/vtapi/v2/file/report?apikey={api_key}&resource={resource_hash}&allinfo=1",
                verify=False
            )

        if r.status_code != 200:
            time.sleep(delay)
            count += 1
        else:
            break
    print()

    if r.status_code != 200:
        return False, None
    
    data = json.loads(r.text)
    if data["response_code"] != 1:
        return True, None
    return True, data


# MAIN RUNNER
if __name__ == "__main__":
    resource_hash = input("Enter the resource hash: ").lower().strip()

    success, data = get_report(resource_hash)
    if not success:
        print("Request failed")
        exit(1)
    if data is None:
        print("No data available")
        exit(0)

    with open(f"{data['sha256']}.json", "w") as f:
        json.dump(data, f, indent=4)
