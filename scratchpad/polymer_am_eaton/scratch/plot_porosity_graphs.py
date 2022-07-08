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

df1 = pd.read_csv("/data01/Eaton_Polymer_AM/csv_files/porosity_vals_dt4.csv")
sample_tag = df1["sample_tag"]
layer =  df1["layer"]
porosity = df1["porosity"]

samp1 = np.mean([porosity[i] for i in range(0,4)])
samp2 = np.mean([porosity[i] for i in range(4,8)]) #4,9
samp3 = np.mean([porosity[i] for i in range(9,14)])
samp4 = np.mean([porosity[i] for i in range(14,19)]) #14,20
samp5 = np.mean([porosity[i] for i in range(20,25)]) #20,26
samp6 = np.mean([porosity[i] for i in range(26,32)])
samp7 = np.mean([porosity[i] for i in range(32,38)])
samp8 = np.mean([porosity[i] for i in range(38,43)]) #38,44
samp9 = np.mean([porosity[i] for i in range(44,48)]) #44,50
samp10 = np.mean([porosity[i] for i in range(50,55)]) #50,56
samp11 = np.mean([porosity[i] for i in range(57,61)]) #56,62
samp12 = np.mean([porosity[i] for i in range(62,64)]) #62,65

list_porosity = np.array([samp1, samp2, samp3, samp4, samp5, samp6, samp7, samp8, samp9, samp10, samp11, samp12])
samples = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

fig, ax = plt.subplots(1,1, figsize = (16,8))
ax.bar(samples, list_porosity)
ax.set_title("Porosity of Each Sample")
ax.set_ylabel("Porosity")
ax.set_xlabel("Sample Number")
#ax.set(xlim=(20, 1130))
plt.savefig(plots_dir + f'porosity_bar_graph.png', format='png')
plt.close()

# for i in range(len(sample_tag)):
#     fig, ax = plt.subplots(1,1, figsize = (16,8))
#     ax.bar(str(sample_tag[i]) + str(layer[i]), porosity)
#     ax.set_title(f"Porosity of at Each Layer (Sample {sample_tag[i]})")
#     ax.set_ylabel("Porosity")
#     ax.set_xlabel("Layer Number Number")
#     #ax.set(xlim=(20, 1130))
#     plt.savefig(plots_dir + f'porosity_bar_graph_layers_sample{sample_tag[i]}.png', format='png')
#     plt.close()