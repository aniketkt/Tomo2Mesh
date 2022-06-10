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

from tomo_encoders.tasks.digital_zoom import coarse_map, process_subset
from tomo_encoders.misc.voxel_processing import TimerGPU
from tomo_encoders.structures.voids import Voids
from utils import *
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter


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

if __name__ == "__main__":
    
    # logging measurements
    n_iter = 1
    keys = ["b", "sparsity", "voids", "reconstruct-coarse", "label-coarse", "reconstruct-subset", "label-subset"]
    df = pd.DataFrame(columns = keys)

    t_gpu = TimerGPU("secs")

    # load the U-net model
    model_params = get_model_params(model_tag)
    segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    # segmenter.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    


    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    projs, theta, center = read_raw_data_b2()
    ##### BEGIN ALGORITHM ########
    # coarse mapping
    
    b, b_K = b_val, b_val
    print(f"EXPERIMENT with b, b_K {(b,b_K)}")
    cp.fft.config.clear_plan_cache(); cp._default_memory_pool.free_all_blocks()    

    t_rec_coarse = []
    t_label_coarse = []
    for ib in range(n_iter):
        voids_b = coarse_map(projs, theta, center, b, b_K, 2)
        t_rec_coarse.append(voids_b.compute_time["reconstruction"])
        t_label_coarse.append(voids_b.compute_time["labeling"])
    df["reconstruct-coarse"] = t_rec_coarse
    df["label-coarse"] = t_label_coarse


    t_rec_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        p_voids, r_fac = voids_b.export_grid(wd)    
        z_pts = np.unique(p_voids.points[:,0])
        x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"], seg_batch = True)
        t_rec_subset.append(t_gpu.toc("subset reconstruction"))
    df["reconstruct-subset"] = t_rec_subset

    t_label_subset = []
    for ib in range(n_iter):
        t_gpu.tic()
        voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
        t_label_subset.append(t_gpu.toc("subset relabeling"))
    df["label-subset"] = t_label_subset

    
    print("this are the time measurements")
    df["sparsity"] = [1/r_fac]*n_iter
    df["voids"] = [voids_b.n_voids]*n_iter
    df["b"] = [b_val]*n_iter
    df["r_fac_z"] = [len(z_pts)/(p_voids.vol_shape[0]//wd)]*n_iter
    print(df)
    df.to_csv(os.path.join(time_logs, f"coarse_mapping_{b}_only.csv"), index = False, mode = 'w')
    
    
    print("saving voids data now...")
    t_gpu.tic()
    voids_b.write_to_disk(os.path.join(voids_dir,f"voids_b_{b}"))
    surf_b = voids_b.export_void_mesh_with_texture("sizes")
    surf_b.write_ply(os.path.join(ply_dir, f"voids_b_{b}.ply"))
    voids.write_to_disk(os.path.join(voids_dir,f"voids_b_{b}_subset"))
    surf = voids.export_void_mesh_with_texture("sizes")
    surf.write_ply(os.path.join(ply_dir, f"voids_b_{b}_subset.ply"))
    t_gpu.toc("saving data")
    

        
    
    
