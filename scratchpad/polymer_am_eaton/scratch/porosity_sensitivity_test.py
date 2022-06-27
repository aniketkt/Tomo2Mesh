from cmath import nan
from Tomo2Mesh.scratchpad.polymer_am_eaton.scratch.plot_eaton_graphs import number_density_z
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


sample_tag = '1'
layer = 1
raw_pixel_size = pixel_size
number_density_radius = 50
b=1
vol = 288*612*612*(np.pi/4)

projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
#voids = void_map_gpu(projs, theta, center, dark, flat, b, raw_pixel_size)
voids = void_map_all(projs, theta, center, dark, flat, b, raw_pixel_size)
voids.calc_max_feret_dm()
voids.calc_number_density(number_density_radius)

thresh = [1,2,3,4,5,6]
porosity = []
for i in range(len(thresh)):
    count = 0
    for j in (len(voids['sizes'])):
        if np.mult(voids['sizes'][j].shape) > thresh[i]**3:
            count+=np.sum(voids['sizes'][j])
    porosity.append(count/vol)

sns.set(font_scale=1.3)
sns.set_style(style = "white")
fig, ax = plt.subplots(1,1, figsize = (16,8))
ax.scatter(thresh, porosity)
ax.set_title("Porosity vs. Binning Factor")
ax.set_ylabel("Porosity")
ax.set_xlabel("Binning Factor")
#ax.set(xlim=(20, 1130))
plt.savefig(plots_dir + f'sensitivity_test_porosity.png', format='png')
plt.close()