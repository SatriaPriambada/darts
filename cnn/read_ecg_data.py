import os
import numpy as np
from collections import Counter
import pickle

if __name__ == "__main__":
    # Path
    path = "~/data/challenge2017.pkl"
    full_path = os.path.expanduser(path)
    with open(full_path, "rb") as fin:
        res = pickle.load(fin)

    all_data = res["data"]
    all_label = res["label"]
    print(all_label)
    print(Counter(all_label))
