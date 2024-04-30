# IMPORTS
import os

from labelling import label_sample
from misc import pretty_sleep
from virustotal import *

# CONSTANTS
CONSECUTIVE_FAILURE_COUNT = 5  # Don't continue processing if there are this many failures in a row
FAILURE_SLEEP_DURATION = 15
API_KEY = os.getenv("API_KEY", None)

# PRE-CHECKS
# Ensure that an API key was provided
if API_KEY is None:
    print("Please set the `API_KEY` environment variable first.")
    exit(1)

print(f"Using API key '{API_KEY}'")

# Check that the folders exist
os.makedirs("../data/vt-data", exist_ok=True)

# INPUT
# Get all the hashes that we have
with open("../data/hashes.txt", "r") as f:
    all_hashes = set([x.strip() for x in f.readlines()])

# Get all of the hashes that we have already processed
try:
    with open("../data/done-hashes.txt", "r") as f:
        done_hashes = set([x.strip() for x in f.readlines()])
except FileNotFoundError:
    done_hashes = set()

# Remove all the done hashes from the master list
hashes = all_hashes.difference(done_hashes)

# MAIN
print("Press Ctrl + C to stop early.")

failure_count = 0
while failure_count < CONSECUTIVE_FAILURE_COUNT:
    try:
        # Get the next hash to process
        next_hash = hashes.pop()

        # Try and get the report
        success, report = get_report(next_hash, API_KEY)
        if not success:
            failure_count += 1
            print(f"Failed to get '{next_hash}' (Consecutive failure {failure_count}/{CONSECUTIVE_FAILURE_COUNT})")
            if failure_count < CONSECUTIVE_FAILURE_COUNT:
                print(f"Sleeping for {FAILURE_SLEEP_DURATION} seconds before continuing.")
                pretty_sleep(FAILURE_SLEEP_DURATION, dot_length=10, interval=0.1)

            continue

        failure_count = 0

        if report is None:
            print(f"VirusTotal has not yet processed '{next_hash}'. Moving on.")
            all_hashes.remove(next_hash)
            continue

        # Get the hashes from the report
        md5 = report["md5"]
        sha1 = report["sha1"]
        sha256 = report["sha256"]

        # Remove these hashes from the set
        hashes = hashes.difference({md5, sha1, sha256})

        # Try and guess the label of the sample
        label = label_sample(report)

        # Write the JSON report to disk
        with open(f"../data/vt-data/{label}_{sha256}.json", "w") as f:
            json.dump(report, f, indent=4)

        time.sleep(0.5)  # Wait a while for the request to cool-off

    except KeyboardInterrupt:
        print("\nRequest to stop received.")
        break
    except Exception as e:
        print(f"\nAn error occured: {e}")
        print("Stopping.")
        break

if failure_count >= CONSECUTIVE_FAILURE_COUNT:
    print("Max failures hit, stopping.")

# OUTPUT
print("Updating the completed hashes...")
update_master_list(all_hashes)
update_done_hashes()
print("All done.")
