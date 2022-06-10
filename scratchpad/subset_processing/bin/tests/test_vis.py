#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

from utils import *
from tomo_encoders.tasks.digital_zoom import coarse_map, process_subset
from tomo_encoders.misc.voxel_processing import TimerGPU
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import cupy as cp
from tomo_encoders.structures.voids import Voids
from tomo_encoders.misc import viewer
import matplotlib.pyplot as plt
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


if __name__ == "__main__":

    t_gpu = TimerGPU()
    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    projs, theta, center = read_raw_data(raw_fname, wd*b)
    
    # load the U-net model
    model_params = get_model_params(model_tag)
    segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    segmenter.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    
    
    cp.fft.config.clear_plan_cache()
    cp._default_memory_pool.free_all_blocks()    
    
    # coarse mapping
    t_gpu.tic()
    voids_b = coarse_map(projs, theta, center, b, b_K)
    t_gpu.toc(f'coarse mapping')

    # selection criteria
    # none

    # export to grid
    t_gpu.tic()
    p_voids, r_fac = voids_b.export_grid(wd)    
    t_gpu.toc(f'export to grid')
        
    # process subset
    t_gpu.tic()
    x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"], seg_batch = False)
    t_gpu.toc(f'subset processing time')

    voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
    voids_b.export_void_mesh_with_texture("sizes").write_ply(ply_coarse)
    voids.export_void_mesh_with_texture("sizes").write_ply(ply_subset)
    
    
    # fig, ax = plt.subplots(1,2)
    # _ = ax[0].hist(vals, bins = 500)
    # _ = ax[1].hist(x_voids.reshape(-1), bins = 500)
    # plt.show()
    

