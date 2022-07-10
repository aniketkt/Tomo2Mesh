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

from tomo2mesh.porosity.mapping import coarse_map, process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU
from tomo2mesh.structures.voids import Voids
from tomo2mesh.projects.steel_part_vis.rw_utils import *
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter


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


b_val = int(sys.argv[1])
b, b_K = b_val, b_val
n_iter = 3
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
    projs, theta, center = read_raw_data_b1()
    ##### BEGIN ALGORITHM ########
    # coarse mapping
    
    print(f"EXPERIMENT with b, b_K {(b,b_K)}")
     

    t_rec_coarse = []
    t_label_coarse = []
    for ib in range(n_iter):
        voids_b = coarse_map(projs, theta, center, b, b_K, 2)
        t_gpu.tic()
        
        p_voids, r_fac = voids_b.export_grid(wd)    
        t_export = t_gpu.toc()
        t_rec_coarse.append(voids_b.compute_time["reconstruction"])
        t_label_coarse.append(voids_b.compute_time["labeling"] + t_export)
        cp._default_memory_pool.free_all_blocks()   
        cp.fft.config.get_plan_cache().clear()


    # voids selection criteria goes here
    

    t_rec_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"])
        t_rec_subset.append(t_gpu.toc("subset reconstruction"))
        z_pts = np.unique(p_voids.points[:,0])
        cp._default_memory_pool.free_all_blocks()   
        cp.fft.config.get_plan_cache().clear()


    t_label_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
        t_label_subset.append(t_gpu.toc("subset relabeling"))

    print("this are the time measurements")
    df["reconstruct-coarse"] = t_rec_coarse
    df["label-coarse"] = t_label_coarse
    df["reconstruct-subset"] = t_rec_subset
    df["label-subset"] = t_label_subset
    df["sparsity"] = [1/r_fac]*n_iter
    df["voids"] = [voids_b.n_voids]*n_iter
    df["b"] = [b_val]*n_iter
    df["r_fac_z"] = [len(z_pts)/(p_voids.vol_shape[0]//wd)]*n_iter
    print(df)
    df.to_csv(os.path.join(time_logs, f"coarse2fine_4k_{b}.csv"), index = False, mode = 'w')
    
    print("saving voids data now...")
    t_gpu.tic()
    voids_b.write_to_disk(os.path.join(voids_dir,f"voids_4k_b_{b}"))
    surf_b = voids_b.export_void_mesh_with_texture("sizes", edge_thresh = 1.0)
    surf_b.write_ply(os.path.join(ply_dir, f"voids_4k_b_{b}.ply"))
    voids.write_to_disk(os.path.join(voids_dir,f"voids_4k_b_{b}_subset"))
    surf = voids.export_void_mesh_with_texture("sizes", edge_thresh = 1.0)
    surf.write_ply(os.path.join(ply_dir, f"voids_4k_{b}_subset.ply"))
    t_gpu.toc("saving data")
    

        
    
    
