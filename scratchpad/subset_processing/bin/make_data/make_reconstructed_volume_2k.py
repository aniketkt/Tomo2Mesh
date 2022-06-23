#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import os
from utils import *
from tomo_encoders.tasks.digital_zoom import coarse_map, process_subset, crop_projs
from tomo_encoders.misc.voxel_processing import TimerGPU
from tomo_encoders.reconstruction.recon import recon_patches_3d
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import cupy as cp
import gc
from tomo_encoders.structures.grid import Grid
from tomo_encoders import DataFile

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

full_vis = False

if __name__ == "__main__":

    t_gpu = TimerGPU()
    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    projs, theta, center = read_raw_data(raw_fname, wd*b)
    
    
    if full_vis:
        # load the U-net model
        model_params = get_model_params(model_tag)
        segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                            model_names = model_names, \
                            model_path = model_path)    
        segmenter.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

        
        cp._default_memory_pool.free_all_blocks()    
        
        # coarse mapping
        t_gpu.tic()
        voids_b = coarse_map(projs, theta, center, b, b_K)
        t_gpu.toc(f'coarse mapping')

        p_voids = Grid((projs.shape[1], projs.shape[-1], projs.shape[-1]), width = wd)        
        # process subset
        t_gpu.tic()
        x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"], seg_batch = True)
        t_gpu.toc(f'subset processing time')
        
        seg_fpath = os.path.join(data_output, "binarized")
        V = np.empty(p_voids.vol_shape, dtype = np.uint8)
        p_voids.fill_patches_in_volume(x_voids, V)
        ds = DataFile(seg_fpath, tiff = True, d_type = np.uint8, d_shape = V.shape)
        ds.write_full(V)
    
    
    # cp._default_memory_pool.free_all_blocks()    
    
    # p_voids = Grid((projs.shape[1], projs.shape[-1], projs.shape[-1]), width = wd)        
    # # process subset
    # t_gpu.tic()
    # x_voids, p_voids = recon_patches_3d(projs, theta, center, p_voids, apply_fbp =True)
    # t_gpu.toc(f'subset processing time')
    
    # "saving reconstruction"
    # min_val = x_voids[:,::4,::4,::4].min()
    # max_val = x_voids[:,::4,::4,::4].max()
    # x_voids = 255.0*(x_voids - min_val)/(max_val - min_val)

    # ct_fpath = os.path.join(data_output, "reconstructed")
    # V = np.empty(p_voids.vol_shape, dtype = np.uint8)
    # p_voids.fill_patches_in_volume(x_voids, V)
    # ds = DataFile(ct_fpath, tiff = True, d_type = np.uint8, d_shape = V.shape)
    # ds.write_full(V)

