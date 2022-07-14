from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from tomo2mesh.projects.eaton.recon import recon_binned, recon_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size
from tomo2mesh.projects.eaton.rw_utils_ae import read_raw_data_1X, save_path
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.structures.grid import Grid
from tomo2mesh.structures.patches import Patches
from tomo2mesh.projects.eaton.void_mapping import void_map_gpu, void_map_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size

import sys
import os
import pandas as pd
import cc3d

import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from cupyx.scipy import ndimage
from scipy import ndimage as ndimage_cpu
from scipy import ndimage as ndimage_cpu
import matplotlib.pyplot as plt

plots_dir = '/home/yash/eaton_plots2/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')


df1 = pd.read_csv("/data01/Eaton_Polymer_AM/csv_files/porosity_vals_adj2.csv")
sample_tag = list(df1["sample_tag"])
scan_tag =  list(df1["scan_tag"])
porosity = list(df1["porosity"])


samp1 = np.mean([porosity[i] for i in sample_tag if i=="1"])
samp2 = np.mean([porosity[i] for i in sample_tag if i=="2"])
samp3 = np.mean([porosity[i] for i in sample_tag if i=="3"])
samp4 = np.mean([porosity[i] for i in sample_tag if i=="4"])
samp5 = np.mean([porosity[i] for i in sample_tag if i=="5"])
samp6 = np.mean([porosity[i] for i in sample_tag if i=="6"])
samp7 = np.mean([porosity[i] for i in sample_tag if i=="7"])
samp8 = np.mean([porosity[i] for i in sample_tag if i=="8"])
samp9 = np.mean([porosity[i] for i in sample_tag if i=="9"])
samp10 = np.mean([porosity[i] for i in sample_tag if i=="10"])
samp11 = np.mean([porosity[i] for i in sample_tag if i=="11"])
samp12 = np.mean([porosity[i] for i in sample_tag if i=="12"])
sampleinj = np.mean([porosity[i] for i in sample_tag if i=="injmold"])


list_porosity = np.array([samp1, samp2, samp3, samp4, samp5, samp6, samp7, samp8, samp9, samp10, samp11, samp12, sampleinj])
samples = np.unique(sample_tag)

fig, ax = plt.subplots(1,1, figsize = (16,8))
ax.bar(samples, list_porosity)
ax.set_title("Porosity of Each Sample")
ax.set_ylabel("Porosity")
ax.set_xlabel("Sample Number")
#ax.set(xlim=(20, 1130))
plt.savefig(plots_dir + f'porosity_bar_graph.png', format='png')
plt.close()
