import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from os.path import exists as file_exists
import os
from skimage import io
from skimage import measure as ms
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from tomo2mesh import DataFile
from tomo_encoders.mesh_processing.EllipsoidFitting import EllipsoidTool
from tomo_encoders.mesh_processing.vox2mesh import *
from tomo2mesh import Grid, Patches
from tomo_encoders.labeling.detect_voids import export_voids
from tomo_encoders.mesh_processing.void_params import _edge_map

def fit_ellipsoid_adj(x_voids_adj, bin_fact):
    start = time.time()
    loc_exists = []
    cen = []
    rad = []
    rot = []
    ellipticity_E = []
    ellipticity_P = []
    list_time = []
    ET = EllipsoidTool()
    for l in range(len(x_voids_adj)):
        t0 = time.time()
        #Find all the points (x,y,z) where the void exists
        P = []
        P = np.asarray(np.where(x_voids_adj[l])).T 
        loc_exists.append(np.array(P))
        P = np.array(P[::bin_fact])
            
        #Calculate the center location, radii, and orientation matrix of the fitted ellipsoid
        contain = ET.getMinVolEllipse(P, 0.01) 
        cen.append(contain[0])
        rad.append(contain[1])
        rot.append(contain[2])
        
        #Calculate the ellipticity of the ellipsoid
        a,b,c = sorted(contain[0], reverse=True)
        elp_E = np.sqrt((a**2-b**2)/a**2) #Equatorial ellipticity of tri-axial ellipsoid
        elp_P = np.sqrt((a**2-c**2)/a**2) #Polar ellipticity of tri-axial ellipsoid
        ellipticity_E.append(elp_E)
        ellipticity_P.append(elp_P)
            
        t1 = time.time()
        print("Time for Iteration:",t1-t0)
        print("Iteration:"+str(l+1)+"/"+str(len(x_voids_adj)))
    end = time.time()
    list_time.append(end-start)
    print("Time:",end-start)

    return (cen, rad, rot, loc_exists, ellipticity_E, ellipticity_P, list_time)

N_MAX_DETECT = 1e12
seg_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel1_sam1/segmented'
ds = DataFile(seg_path, tiff = True)
Vx_full = ds.read_full()

x_voids, p_voids= export_voids(Vx_full, N_MAX_DETECT, TIMEIT = True, invert = False)

vol = p_voids["void_size"]**3
vol = vol[1:len(vol)]
vol = vol*(1.172**3)

#Retreive Boundary 3darray of all voids
x_voids_adj = x_voids[1:len(x_voids)]
x_voids_adj2 = []
for i in range(len(x_voids_adj)):
    x_voids_adj2.append(_edge_map(x_voids_adj[i]))

#Collect sample data
samp_idx = np.random.randint(0, high=len(x_voids_adj2)/3, size = 500)
vol_samp = []
x_voids_samp = []
for i in range(len(samp_idx)):
    vol_samp.append(vol[samp_idx[i]])
    x_voids_samp.append(x_voids_adj2[samp_idx[i]])

#Calculate and save params from ellipsoid fitting
bin_fact = [1,2,4,8,16,32]
for i in range(len(bin_fact)):
    print("Trim Factor:", bin_fact[i])
    cen, rad, rot, loc_exists, ellipticity_E, ellipticity_P, list_time = fit_ellipsoid_adj(x_voids_samp, bin_fact[i])
    info = {"Volume": vol_samp, "Equatorial_Ellipticity": ellipticity_E, "Polar_Ellipticity": ellipticity_P, "Axis_Lengths": rad, "Time": list_time*len(rad)}
    df = pd.DataFrame(info)
    save_path = "/data01/csv_files/trim_sensitivity_f"+str(bin_fact[i])+".csv"
    df.to_csv(save_path)
    print("")
