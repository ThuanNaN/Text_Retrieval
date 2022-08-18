from utils.ultis import AP
from utils.data_preprocessing import preprocess_res
import json
import os
import numpy as np


if __name__ == "__main__":
    result_files = os.listdir("./Result")
    files_path = sorted(result_files, key=lambda path:(int(path.split(".")[0])))

    rs  = []

    for file in files_path:
        with open("./Result/"+ file, "r") as f:
            results = [ rs.strip().split(" ")[-1] for rs in f.readlines()]
        rs.append(results)
    rs = np.array(rs)

    res = preprocess_res("./TEST/RES")
    ap = AP(rs, res).mean()
    print(ap)