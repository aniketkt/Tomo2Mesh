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
    
    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    if b == 2:
        pixel_size = 2.34
        projs, theta, center = read_raw_data_b2()
        kval = '2k'
    elif b == 4:
        pixel_size = 1.17
        projs, theta, center = read_raw_data_b1()
        kval = '4k'
    else:
        raise NotImplementedError("unacceptable value")
    
    print(f"EXPERIMENT with b, b_K {(b,b_K)}")
    print(f"SHAPE OF PROJECTIONS DATA: {projs.shape}")
    for criteria in criteria_list:
        t_gpu.tic()
        cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()
        voids_b = coarse_map(projs, theta, center, b, b_K, 2)
        # voids selection criteria goes here
        void_id = np.argmax(voids_b["sizes"])
        if criteria == "cylindrical_neighborhood":
            print("select cylindrical neighborhood")
            z_min = voids_b["cents"][void_id,0] - 800.0/(b*pixel_size)
            z_max = voids_b["cents"][void_id,0] + 800.0/(b*pixel_size)
            voids_b.select_z_slab(z_min, z_max)
        elif criteria == "spherical_neighborhood":
            print("select spherical neighborhood")
            voids_b.select_around_void(void_id, 800.0, pixel_size_um=pixel_size)

        # export subset coordinates here
        p_voids, r_fac = voids_b.export_grid(wd)    
        cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()        
        # process subset reconstruction
        x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"])
        
        # import voids data from subset reconstruction
        voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
        t_mapping = t_gpu.toc(f"CRITERIA: {criteria}")
    
        t_gpu.tic()
        surf_b = voids_b.export_void_mesh_mproc("sizes", edge_thresh = 0.0)
        surf = voids.export_void_mesh_mproc("sizes", edge_thresh = 1.0)        
        t_mesh = t_gpu.toc()
        
        print("saving voids data now...")
        surf_b.write_ply(os.path.join(ply_dir, f"voids_{kval}_b_{b}_{criteria}.ply"))
        surf.write_ply(os.path.join(ply_dir, f"voids_{kval}_{b}_subset_{criteria}.ply"))
        print(os.path.join(ply_dir, f"voids_{kval}_{b}_subset_{criteria}.ply"))
        print(f"CRITERIA: {criteria}; t_mapping: {t_mapping:.2f} secs; t_mesh: {t_mesh:.2f} secs; 1/r value: {1/r_fac:.4g}")


    
    

        
    
    
