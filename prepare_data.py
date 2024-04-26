# IMPORTS
import os

from tqdm import trange, tqdm

from misc import load_data, reports_to_coordinates

# MAIN CODE
data, filepaths = load_data()
if len(data) == 0:
    print("No data found in data folder.")

coordinates, top_unigrams = reports_to_coordinates(data, num_unigrams=10000, seed=42)

with open("data/top_unigrams.txt", "w") as f:
    for unigram in tqdm(top_unigrams, desc="Writing top unigrams"):
        f.write(f"{unigram}\n")

with open("data/dataset.csv", "w") as f:
    # We need to write this incredibly stupid header
    num_elem_in_coordinates = len(coordinates[0])
    num_elem_in_row = num_elem_in_coordinates + 2  # +2 for label and hash

    f.write("label,hash,")
    for i in range(num_elem_in_coordinates):
        f.write(f"dim-{i:03d}")
        if i != num_elem_in_coordinates - 1:
            f.write(",")
        else:
            f.write("\n")

    # Now we can write the actual data
    for i in trange(len(coordinates), desc="Generating dataset"):
        filepath = filepaths[i]
        coordinate = coordinates[i]

        splitted = os.path.basename(filepath).split("_")
        if len(splitted) != 2:
            print(f"'{filepath}' does not have the correct file naming convention; skipping")
            continue

        label, hash_ = splitted
        hash_ = hash_.split(".")[0]  # Remove the trailing ".json"
        row_list = [label] + [hash_] + list(coordinate)

        for i, elem in enumerate(row_list):
            f.write(str(elem))
            if i != num_elem_in_row - 1:
                f.write(",")
            else:
                f.write("\n")

print("Done")
