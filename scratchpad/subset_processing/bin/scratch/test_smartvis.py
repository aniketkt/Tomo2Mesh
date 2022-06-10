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

sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import model_path, get_model_params
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper')
from tomo_encoders.tasks.digital_zoom import process_patches
from tomo_encoders.structures.voids import Voids
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders.tasks.digital_zoom import coarse_segmentation


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

model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)
# patch size
wd = 32
# guess surface parameters
b = 4
b_K = 4
sparse_flag = True
pixel_res = 1.17
size_um = -1 # um
void_rank = 1
radius_around_void_um = 800.0 # um
blur_size = 0.5
# handy code for timing stuff
# st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
# end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
# print(f"time checkpoint {t_chkpt/1000.0:.2f} secs")

## Output for vis
ply_lowres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/lowres_full.ply'
ply_highres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/highres_around_void_%i.ply'%void_rank
voids_highres = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_highres'
voids_lowres = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_lowres'

if __name__ == "__main__":

    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

    # read data and initialize output arrays
    ## to-do: ensure reconstructed object has dimensions that are a multiple of the (wd,wd,wd) !!    
    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'r')
    projs = np.asarray(hf["data"][:])
    theta = np.asarray(hf['theta'][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    # make sure projection shapes are divisible by the patch width (both binning and full steps)
    print("BEGIN: Read projection data from disk")
    print(f'\tSTAT: shape of raw projection data: {projs.shape}')
    
    ##### BEGIN ALGORITHM ########
    # guess surface
    print(f"\nSTEP: visualize all voids with size greater than {size_um:.2f} um")
    V_bin, rec_min_max = coarse_segmentation(projs, theta, center, b_K, b, blur_size)
    voids_b = Voids().guess_voids(V_bin, b)    
    voids_b.select_by_size(size_um, pixel_size_um = pixel_res)
    voids_b.sort_by_size(reverse = True)
    surf = voids_b.export_void_mesh_with_texture("sizes")
    surf.write_ply(ply_lowres)

    # guess roi around a void
    void_id = np.argsort(voids_b["sizes"])[-void_rank]
    voids_b.select_around_void(void_id, radius_around_void_um, pixel_size_um = pixel_res)
    print(f"\nSTEP: visualize voids in the neighborhood of void id {void_id} at full detail")    

    cp.fft.config.clear_plan_cache()
    p_sel, r_fac = voids_b.export_grid(wd)    
    x_voids, p_voids = process_patches(projs, theta, center, fe, p_sel, rec_min_max)
    
    # export voids
    voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
    voids_b.write_to_disk(voids_lowres)    
    voids.write_to_disk(voids_highres)
    surf = voids.export_void_mesh_with_texture("sizes")
    surf.write_ply(ply_highres)

    # # complete: save stuff    
    # Vp = np.zeros(p_voids.vol_shape, dtype = np.uint8)
    # p_voids.fill_patches_in_volume(x_voids, Vp)
    # ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8, VERBOSITY=0)
    # ds_save.create_new(overwrite=True)
    # ds_save.write_full(Vp)
    # Vp_mask = np.zeros(p_voids.vol_shape, dtype = np.uint8) # Save for illustration purposes the guessed neighborhood of the surface
    # p_voids.fill_patches_in_volume(np.ones((len(p_voids),wd,wd,wd)), Vp_mask)
    # ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8, VERBOSITY=0)
    # ds_save.create_new(overwrite=True)
    # ds_save.write_full(Vp_mask)
    