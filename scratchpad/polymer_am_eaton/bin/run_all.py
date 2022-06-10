import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/data01/AMPolyCalc/code')
from rw_utils import read_raw_data_1X, save_path
from void_mapping import void_map_gpu
from tomo_encoders.misc import viewer

import cupy as cp
from params import pixel_size_1X, voids_path
import os
import pandas as pd

def run(sample_tag, layer_tag):

    projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer_tag)
    b = 4

    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size_1X)
    voids_4.calculate_ellipse_radius_ratio()

    voids_fname = f"voids_sample{sample_tag}_layer{layer_tag}_b{b}"
    voids_4.write_to_disk(os.path.join(voids_path, voids_fname))
    return

if __name__ == "__main__":

    df = pd.read_csv(save_path)
    sample_num = 1
    for sample_num in [1,2]:
        layers = list(df[df["sample_num"] == sample_num]["layer"])
        for layer in layers:
            print(f"\n\nPROCESSING sample {sample_num}; layer {layer}")
            run(sample_num, layer)
    






