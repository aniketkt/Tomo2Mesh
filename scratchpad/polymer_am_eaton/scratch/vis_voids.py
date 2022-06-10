#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
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
dir_path = '/data01/AM_steel_project/xzhang_feb22_rec/data/wheel2_sam4'
save_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel2_sam4'
# dir_path = '/data01/AM_steel_project/mli_L206_HT_650/data'
# save_path = '/data01/AM_steel_project/mli_L206_HT_650/seg_data'

if not os.path.exists(save_path): os.makedirs(save_path)


# STITCHING PARAMETERS
id_start = [0,0]
id_end = [915,915]
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



def make_stitched(dir_path, id_start, id_end, bin_fact, fe, wd):
    #n_layers = len(id_start)
    n_layers = 2
    ind_adj = [0,2]
    Vx_full = []
    for il in range(n_layers):
        a = ind_adj[il]
        ds = DataFile(os.path.join(dir_path, f'layer{a+1}'), tiff=True)
        #ds = DataFile(os.path.join(dir_path, f'layer{il+1}'), tiff=True)  
        #ds = DataFile(os.path.join(dir_path, f'mli_L206_HT_650_L{il+1}_rec_1x1_uint16_tiff'), tiff=True) 
        #import pdb; pdb.set_trace() 
        V_temp = ds.read_chunk(axis=0, slice_start=id_start[il]//bin_fact, slice_end=id_end[il]//bin_fact, return_slice=False).astype(np.float32)
        V_temp = V_temp[::bin_fact,::bin_fact,::bin_fact]
        h = modified_autocontrast(V_temp, s=0.10, normalize_sampling_factor=4)
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
    # STEP 1
    # make a big volume that stitches together all layers in one volume; Vx_full.shape will be (tot_ht, ny, nx)
    # STEP 2
    # Process Vx_full into Vy_full where Vy_full contains only ones (inside void) and zeros (inside metal)
    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

    t_start = time.time()
    Vx_full = make_stitched(dir_path, id_start, id_end, bin_fact, fe, wd)

    ds_save = DataFile(os.path.join(save_path, "segmented"), tiff = True, d_shape = Vx_full.shape, d_type = Vx_full.dtype)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vx_full)

    t0 = time.time()
    print(f"TIME segmentation: {time.time()-t_start:.2f} seconds")


    # STEP 3
    # Process Vy_full into void_vols where void_vols is a list of many ndarrays with different shapes (pz, py, px) representing each void
    # Also output cz, cy, cx for each void_vol in void_vols giving the center of the void volume w.r.t. the coordinates in Vy_full
    x_voids, p_voids = export_voids(Vx_full, N_MAX_DETECT, TIMEIT = True, invert = False)

    # STEP 4
    # Process all void_vols into void_surfs in the form of a single .ply file and save
    
    #Export data into .ply files for visualization
    cor = p_voids["points"]
    cor_surf = [cor[0]]
    surf_vol = [x_voids[0]]

    cor_adj = cor[1:len(cor)]
    x_voids_adj = x_voids[1:len(x_voids)]
    
    start = time.time()
    voids2ply(x_voids_adj,cor_adj,"3DVisualization_Voids_w2_s4")
    voids2ply(surf_vol,cor_surf,"3DVisualization_Surface_w2_s4")
    end = time.time()
    print("Time:",(end-start)/60,"minutes")
    

    
    
    
 
