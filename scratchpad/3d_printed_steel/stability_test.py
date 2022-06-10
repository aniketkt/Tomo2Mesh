import time

from pyrsistent import ny
from tomo_encoders import DataFile
import os
import numpy as np
import sys
sys.path.append('/home/yash/TomoEncoders/scratchpad/voids_paper/configs/')
from params import model_path, get_model_params
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tomo_encoders.misc.voxel_processing import modified_autocontrast
from tomo_encoders.mesh_processing.vox2mesh import *
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders import Grid, Patches
from tomo_encoders.labeling.detect_voids import export_voids
from tomo_encoders.mesh_processing.void_params import *


######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

# FILE I/O
dir_path = '/data01/AM_steel_project/xzhang_feb22_rec/data/wheel1_sam1'
save_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel1_sam1'
# dir_path = '/data01/AM_steel_project/mli_L206_HT_650/data'
# save_path = '/data01/AM_steel_project/mli_L206_HT_650/seg_data'

if not os.path.exists(save_path): os.makedirs(save_path)


# STITCHING PARAMETERS
id_start = [0,75,75]
id_end = [924,924,924]
# id_start = [130,219,221,225,223,225] 
# id_end = [878,879,879,878,877,1027] 
bin_fact = 1


# SEGMENTATION PARAMETERS
model_tag = "M_a02"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)
# patch size
wd = 32

# VOID DETECTION PARAMETERS
N_MAX_DETECT = 1e12

def make_stitched(dir_path, id_start, id_end, bin_fact, fe, wd, s_val):
    n_layers = len(id_start)
    #n_layers = 2
    #ind_adj = [0,2]
    Vx_full = []
    for il in range(n_layers):
        #a = ind_adj[il]
        #ds = DataFile(os.path.join(dir_path, f'layer{a+1}'), tiff=True)
        ds = DataFile(os.path.join(dir_path, f'layer{il+1}'), tiff=True)  
        #ds = DataFile(os.path.join(dir_path, f'mli_L206_HT_650_L{il+1}_rec_1x1_uint16_tiff'), tiff=True) 
        #import pdb; pdb.set_trace() 
        V_temp = ds.read_chunk(axis=0, slice_start=id_start[il]//bin_fact, slice_end=id_end[il]//bin_fact, return_slice=False).astype(np.float32)
        V_temp = V_temp[::bin_fact,::bin_fact,::bin_fact]
        h = modified_autocontrast(V_temp, s=s_val, normalize_sampling_factor=4)
        V_temp = np.clip(V_temp,*h)
        nz, ny, nx = V_temp.shape
        pad_x = int(np.ceil(nx/wd)*wd - nx)
        pad_y = int(np.ceil(ny/wd)*wd - ny)
        pad_z = int(np.ceil(nz/wd)*wd - nz)
        V_temp = np.pad(V_temp,((0,pad_z),(0,pad_y),(0,pad_x)), mode = "constant", constant_values = ((0,h[1]),(0,h[0]),(0,h[0])))
        print(f"shape of Vx_full was {V_temp.shape}")
        V_temp = segment_volume(V_temp, fe, wd)
        V_temp = V_temp[:-(pad_z), :-(pad_y), :-(pad_x)].copy()
        Vx_full.append(V_temp)
    Vx_full = np.concatenate(Vx_full, axis=0)
    print(Vx_full.shape)
    return Vx_full


def segment_volume(Vx_full, fe, wd):
    p_grid = Grid(Vx_full.shape, width = wd)
    min_max = Vx_full[::4,::4,::4].min(), Vx_full[::4,::4,::4].max()
    x = p_grid.extract(Vx_full)
    x = fe.predict_patches("segmenter", x[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    print(f"shape of x array is {x.shape}")
    p_grid.fill_patches_in_volume(x, Vx_full) # values in Vx_full are converted to binary (0 and 1) in-place
    Vx_full = Vx_full.astype(np.uint8)
    return Vx_full

if __name__ == "__main__":
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128, n_reps = 5, input_size = (wd,wd,wd))    

    por_vals = []
    s_list = [0.001,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]
    for i in range(len(s_list)):
        s_val = s_list[i]
        Vx_full = make_stitched(dir_path, id_start, id_end, bin_fact, fe, wd, s_val)
        x_voids, p_voids = export_voids(Vx_full, N_MAX_DETECT, TIMEIT = True, invert = False)

        #Calculate Porosity
        count1=0
        for j in range(1,len(x_voids)):
            count1+=np.sum(x_voids[j]==1)
        #print(count1)
        print("Porosity (largest selected voids):" ,count1/np.sum(x_voids[0]==0))
        por_vals.append(count1/np.sum(x_voids[0]==0))

        print("Iteration:"+str(i+1)+"/"+str(len(s_list)))

    info = {"Porosity": por_vals, "s": s_list}
    df = pd.DataFrame(info)
    save_path = "/data01/csv_files/stability_test.csv"
    df.to_csv(save_path)

