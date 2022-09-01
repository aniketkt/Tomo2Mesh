#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 


import numpy as np
import matplotlib.pyplot as plt
from tomo2mesh import DataFile
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.structures.grid import Grid
from tomo2mesh.misc.voxel_processing import modified_autocontrast
import sys
import os


# e.g. you can use
# fname = am316_ss_id105_2_c_rec_1x1_uint16_tiff_cropped_4001pix
# fname_out = am316_ss_id105_L3_cropped

model_path = '/data02/steel_am_project/models'
model_tag = "M_a04" # other options are M_a07, M_a02
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
wd = 32


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

    fname = str(sys.argv[1])
    fname_out = str(sys.arg[2]) if len(sys.argv) > 2 else fname
    fpath = os.path.join('/data02/steel_am_project/data/recons', fname)
    fpath_out = os.path.join('/data02/steel_am_project/data/binarized', fname_out)

    # load the U-net model
    model_params = get_model_params(model_tag)
    segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                            model_names = model_names, \
                            model_path = model_path)        
    
    # read the data
    dfile = DataFile(fpath, tiff = True)
    V = dfile.read_full()
    #V = V[::,:4000:,:4000:] #ONLY use this line for am316_ss_id105_L3_cropped
    
    nz, n, n = V.shape
    bin_num = 2
    V = V.reshape((nz//bin_num, bin_num, n//bin_num, bin_num, n//bin_num, bin_num)).mean(axis = (1,3,5))
    # shape of the volume now will be 512,1800,1800 #256,900,900
    nz, n, n = V.shape
    V = V[:(nz//wd)*wd, :(n//wd)*wd, :(n//wd)*wd]
    
    # segment using U-net
    rec_min_max = modified_autocontrast(V, s = 0.001, normalize_sampling_factor=4) # returns a tuple (min, max)
    grid = Grid(V.shape, width = 32) # a list of corner points (n, 3)
    x_rec = grid.extract(V) # returns a list of patches (n, 32, 32, 32)
    
    x_bin = segmenter.predict_patches("segmenter", x_rec[...,np.newaxis], 64, None, min_max = rec_min_max)[...,0]
    V = np.empty(V.shape, dtype = np.uint8)
    grid.fill_patches_in_volume(x_bin, V)
    
    dfile = DataFile(fpath_out, tiff = True, d_shape = V.shape, d_type = V.dtype)
    dfile.create_new(overwrite=True)
    dfile.write_full(V)
    

    # end
