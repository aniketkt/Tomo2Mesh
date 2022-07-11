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

# SELECTION CRITERIA

criteria_list = ["spherical_neighborhood", "cylindrical_neighborhood", "none"]



if __name__ == "__main__":
    
    vol_name = str(sys.argv[1])

    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")

    vol_name = "2k"
    b = 4
    b_K = 4
    projs, theta, center = read_raw_data_b2()
    
    print(f"EXPERIMENT with b, b_K {(b,b_K)}")
    print(f"SHAPE OF PROJECTIONS DATA: {projs.shape}")
    t_gpu.tic()
    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()
    voids_b = coarse_map(projs, theta, center, b, b_K, 2)
    # voids selection criteria goes here

    # export subset coordinates here
    p_voids, r_fac = voids_b.export_grid(wd)    
    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()        
    # process subset reconstruction
    x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"])
    
    # import voids data from subset reconstruction
    voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
    t_mapping = t_gpu.toc(f"COARSE2FINE for {vol_name}, b = {b}")

    t_gpu.tic()
    surf_b = voids_b.export_void_mesh_mproc("sizes", edge_thresh = 0.0)
    surf = voids.export_void_mesh_mproc("sizes", edge_thresh = 1.0)        
    t_mesh = t_gpu.toc(f"MESHING TIME for {vol_name}, b = {b}")
    

        
    
    
