# %%
import matplotlib.pyplot as plt
import sys
sys.path.append('/data01/AMPolyCalc/bin/')
import numpy as np
import os
from skimage import io
from void_params import _edge_map, fit_ellipsoid
from EllipsoidFitting import *

# %%
files = os.listdir("/data01/Eaton_Polymer_AM/raw_data/test_data/")

im_arrs = []
for i in range(len(files)):
    im_arrs.append(io.imread("/data01/Eaton_Polymer_AM/raw_data/test_data/"+files[i]))

im_arrs_bound = []
vol = []
for i in range(len(im_arrs)):
    vol.append(np.sum(np.array(im_arrs[i])))
    im_arrs_bound.append(_edge_map(im_arrs[i]))

cen, rad, rot, loc_exists, ellipticity_E, ellipticity_P = fit_ellipsoid(im_arrs_bound,vol)
print(cen)
print(rad)
print(rot)
print(loc_exists)
print(ellipticity_E)
print(ellipticity_P)


# %%
