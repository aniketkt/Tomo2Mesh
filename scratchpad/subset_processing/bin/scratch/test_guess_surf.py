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

sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs/')
from tomo_encoders import Grid
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d, recon_patches_3d_2
from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu
from params import model_path, get_model_params
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import tensorflow as tf
from cupyx.scipy.ndimage import zoom

# patch size
wd = 32
# guess surface parameters
b = 4
b_K = 4



if __name__ == "__main__":



    # read data and initialize output arrays
    ## to-do: ensure reconstructed object has dimensions that are a multiple of the (wd,wd,wd) !!    
    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'r')
    projs = np.asarray(hf["data"][:])
    theta = np.asarray(hf['theta'][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    # make sure projection shapes are divisible by the patch width (both binning and full steps)
    print(f'SHAPE OF PROJECTION DATA: {projs.shape}')
    
    ##### BEGIN ALGORITHM ########
    # guess surface
    print("STEP: guess surface")
    start_guess = cp.cuda.Event(); end_guess = cp.cuda.Event(); start_guess.record()

    V_rec = recon_binning(projs, theta, center, b_K, b)    


    
    
    
    end_guess.record(); end_guess.synchronize(); t_guess = cp.cuda.get_elapsed_time(start_guess,end_guess)
    print(f'TIME: guessing neighborhood of surface: {t_guess/1000.0:.2f} seconds')

    # complete: save stuff    
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_x_bin', tiff = True, d_shape = V_rec.shape, d_type = np.float32, VERBOSITY=0)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(V_rec)
    
