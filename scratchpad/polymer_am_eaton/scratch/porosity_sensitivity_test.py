from cmath import nan
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


sample_tag = str(sys.argv[1])
layer = str(sys.argv[2])
raw_pixel_size = pixel_size
number_density_radius = 50
b=1
vol = 1152*2448*2448*(np.pi/4)

projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
voids = void_map_all(projs, theta, center, dark, flat, b, raw_pixel_size)

dust_thresh = [2,4,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47] #11 is b=4
porosity = []
for i in range(len(dust_thresh)):
    count = 0
    for j in range(len(voids['x_voids'])):
        if np.all(np.clip(voids['x_voids'][j].shape-np.array((dust_thresh[i],dust_thresh[i],dust_thresh[i])), 0, None)):
            count+=voids['sizes'][j]
    #porosity.append(count/vol)
    porosity.append(count/np.sum(voids['x_boundary']==0))


sns.set(font_scale=1.3)
sns.set_style(style = "white")
fig, ax = plt.subplots(1,1, figsize = (16,8))
ax.scatter(dust_thresh, porosity)
ax.set_title("Porosity vs. Dust Threshold")
ax.set_ylabel("Porosity")
ax.set_xlabel("Dust Threshold")
#ax.set(xlim=(20, 1130))
plt.savefig(plots_dir + f'sensitivity_test_porosity_sample{sample_tag}_layer{str(layer)}.png', format='png')
plt.close()