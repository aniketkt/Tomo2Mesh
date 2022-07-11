#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import time
import seaborn as sns
import pandas as pd
import cupy as cp
import tensorflow

from tomo2mesh.projects.steel_am.coarse2fine import coarse_map, process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU
from tomo2mesh.structures.voids import Voids
from tomo2mesh.projects.steel_am.rw_utils import *
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh import Grid, DataFile
from scipy.ndimage import label as label_np

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############



n_iter = 5
# logging measurements
t_gpu = TimerGPU("secs")
keys = ["b", "sparsity", "r_fac_z", "voids", \
        "reconstruct-coarse", "label-coarse", \
        "reconstruct-subset", "label-subset"]
df = pd.DataFrame(columns = keys)

# load the U-net model
model_params = get_model_params(model_tag)
segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                        model_names = model_names, \
                        model_path = model_path)    
# segmenter.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    


if __name__ == "__main__":
    
    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    projs, theta, center = read_raw_data_b2()
    ##### BEGIN ALGORITHM ########
    # coarse mapping
    cp._default_memory_pool.free_all_blocks()    

    # just to get rec_min_max
    voids_b = coarse_map(projs, theta, center, 4, 4, 2)
    rec_min_max = voids_b["rec_min_max"]
    del voids_b
    

    # just to get reconstructed volume
    nz, ntheta, n = projs.shape
    p_full = Grid((nz,n,n), width=wd)
    x_rec, p_full = process_subset(projs, theta, center, None, p_full, rec_min_max)
    V = np.empty((nz,n,n), dtype = np.float32)
    p_full.fill_patches_in_volume(x_rec, V)
    ds = DataFile(os.path.join(rec_dir, "2k_rec"), tiff = True, d_shape = V.shape, d_type = V.dtype)
    ds.create_new(overwrite=True)
    ds.write_full(V)

    t_rec_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        x_seg, p_full = process_subset(projs, theta, center, segmenter, p_full, rec_min_max)
        t_rec_subset.append(t_gpu.toc("full object reconstruction"))

    V = np.empty((nz,n,n), dtype = np.uint8)
    p_full.fill_patches_in_volume(x_seg, V)
    ds = DataFile(os.path.join(rec_dir, "2k_seg"), tiff = True, d_shape = V.shape, d_type = V.dtype)
    ds.create_new(overwrite=True)
    ds.write_full(V)

    
    t_label_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        Vl, n_det = label_np(V, structure = np.ones((3,3,3),dtype=np.uint8))
        voids = Voids().count_voids(Vl, 1, 2)
        t_label_subset.append(t_gpu.toc("subset relabeling"))

    print("this are the time measurements")
    df["reconstruct-subset"] = t_rec_subset
    df["label-subset"] = t_label_subset
    df["voids"] = [len(voids)]*n_iter
    print(df)
    
    print("saving voids data now...")
    t_gpu.tic()
    voids.write_to_disk(os.path.join(voids_dir,f"voids_b_1"))
    surf = voids.export_void_mesh_with_texture("sizes", edge_thresh = 1.0)
    surf.write_ply(os.path.join(ply_dir, f"voids_b_{1}.ply"))
    t_gpu.toc("saving data")
    

        
    
    
