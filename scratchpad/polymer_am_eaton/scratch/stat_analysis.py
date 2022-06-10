import time
from tomo_encoders import DataFile
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yash/TomoEncoders/scratchpad/voids_paper/configs/')
from params import model_path, get_model_params
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import label

from tomo_encoders.misc.voxel_processing import modified_autocontrast
from tomo_encoders.mesh_processing.vox2mesh import *
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders import Grid, Patches
from tomo_encoders.labeling.detect_voids import export_voids
from tomo_encoders.mesh_processing.void_params import _edge_map, calc_params, fit_ellipsoid

# #%%
# #Plot Slice of Segmented Data
# num = 1000
# seg_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel2_sam1/segmented/'
# im_arr = io.imread(seg_path+"segmented"+"0912"+".tif")
# plt.imshow(im_arr)
# plt.show()


if __name__ == "__main__":

    N_MAX_DETECT = 1e12

    w_num = 3
    sam_num = 5

    for j in range(w_num):
        for k in range(sam_num):
            por_larg = []
            por_all = []
            w = str(j+1)
            s = str(k+1)
            seg_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel'+w+'_sam'+s+'/segmented'
            print("")
            print("Currently Running Sample:"+" w"+w+"_s"+s)
    
            ds = DataFile(seg_path, tiff = True)
            Vx_full = ds.read_full()

            x_voids, p_voids= export_voids(Vx_full, N_MAX_DETECT, TIMEIT = True, invert = False)

            #Calculate Porosity
            count1=0
            for i in range(1,len(x_voids)):
                count1+=np.sum(x_voids[i]==1)
            #print(count1)

            Vx_voids = Vx_full-x_voids[0]
            count2 = np.sum(Vx_voids)
            #print(count2)

            print("Porosity (largest selected voids):" ,count1/np.sum(x_voids[0]==0))
            print("Porosity (all voids):", count2/np.sum(x_voids[0]==0)) 

            por_larg.append(count1/np.sum(x_voids[0]==0))
            por_all.append(count2/np.sum(x_voids[0]==0))

            #Statistical Analysis of Voids
            x_voids_adj = x_voids[1:len(x_voids)]

            #Retreive Boundary 3darray of all voids
            x_voids_adj2 = []
            for i in range(len(x_voids_adj)):
                x_voids_adj2.append(_edge_map(x_voids_adj[i]))

            #Calculate Feret Diameter
            feret_dm = calc_params(x_voids_adj2)
            feret_dm=np.array(feret_dm)
            feret_dm = feret_dm*1.172

            #Ellipsoid Fitting of all voids
            vol1 = p_voids["void_size"]**3
            vol1 = vol1[1:len(vol1)]
            cen, rad, rot, loc_exists, ellipticity_E, ellipticity_P = fit_ellipsoid(x_voids_adj2,vol1)

            #Calculate Diameter of Equivalent Volume Spheres
            vol1 = vol1*(1.172**3)
            r = (vol1*(3/4)*(1/np.pi))**(1/3)
            diam = r*2

            #Save in .csv files
            info = {"Volume": vol1, "Porosity": por_larg*len(feret_dm), "Feret_Diameter": feret_dm, "Equatorial_Ellipticity": ellipticity_E, "Polar_Ellipticity":ellipticity_P, "Axis_Lengths": rad, "Equivalent_Sphere_Volume_Diameter": diam, "Orientation": rot}
            df = pd.DataFrame(info)
            save_path = "/data01/csv_files/stats_w"+w+"_s"+s+".csv"
            df.to_csv(save_path)

            




    