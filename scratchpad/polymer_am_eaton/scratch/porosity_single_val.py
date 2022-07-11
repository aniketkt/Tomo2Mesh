from cmath import nan
from random import sample
from scratchpad.polymer_am_eaton.scratch.plot_eaton_graphs import number_density_z
from tomo2mesh.structures.voids import VoidLayers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
import os

from tomo2mesh.projects.eaton.rw_utils import read_raw_data_1X, save_path
from tomo2mesh.projects.eaton.void_mapping import void_map_gpu, void_map_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size

from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from tomo2mesh.projects.eaton.recon import recon_binned
import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage
from tomo2mesh.fbp.recon import recon_all

plots_dir = '/home/yash/eaton_plots2/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')

df1 = pd.read_csv(save_path)
sample_tag = df1["sample_num"]
layer =  df1["layer"]
dust_thresh = 4 

raw_pixel_size = pixel_size
number_density_radius = 50
b=1
start = [20,0,0,0,194]+[0]*(4+5+12*4+3)
stop = [1151]*(4+10+12*4+3)

porosity = []
sample = []
lyr = []
for i in range(len(sample_tag)):
    projs, theta, center, dark, flat = read_raw_data_1X(str(sample_tag[i]), str(layer[i]))
    voids = void_map_all(projs, theta, center, dark, flat, b, raw_pixel_size, start[i], stop[i])
    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear() 
    count = 0
    vol = (stop[i]-start[i]+1)*2448*2448*(np.pi/4)
    for j in range(len(voids['x_voids'])):
        if np.all(np.clip(voids['x_voids'][j].shape-np.array((dust_thresh,dust_thresh,dust_thresh)), 0, None)):
            count+=voids['sizes'][j]
    porosity.append(count/vol)
    sample.append(sample_tag[i])
    lyr.append(layer[i])

    info = {"sample_tag": sample, "layer": lyr, "porosity": porosity}
    df2 = pd.DataFrame(info)
    path = f"/data01/Eaton_Polymer_AM/csv_files/porosity_vals_dt{dust_thresh}.csv"
    df2.to_csv(path)

