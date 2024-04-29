import os
import json

data_files = os.listdir("../data/vt-data")

for file in data_files:
    with open(f"../data/vt-data/{file}", "r") as f:
        data = json.load(f)
    with open(f"../data/vt-data/{file}", "w") as f:
        json.dump(data, f, indent=4)
