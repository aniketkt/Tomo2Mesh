#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo_encoders import Patches, DataFile
import tensorflow as tf
import time

from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders.misc.feature_maps_vis import view_midplanes
from tomo_encoders.reconstruction.recon import recon_binning

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

# INPUT SIZE CHANGE
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import *

def fit(fe, Xs, Ys):
    
    t0_train = time.time()
    fe.train(Xs, Ys, \
             training_params["training_input_size"], \
             training_params["max_stride"], \
             training_params["batch_size"], \
             training_params["cutoff"], \
             training_params["random_rotate"], \
             training_params["add_noise"], \
             training_params["n_epochs"])
             
    fe.save_models(model_path)
    t1_train = time.time()
    tot_training_time = (t1_train - t0_train)/60.0
    print("\nTRAINING TIME: %.2f minutes"%tot_training_time)
    return fe

def infer(fe, Xs, Ys):
    return

if __name__ == "__main__":

    
    Vy_full = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_y', tiff = True).read_full()

    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'r')
    projs = np.asarray(hf["data"][:])
    theta = np.asarray(hf['theta'][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    Vx_full = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_x', tiff = True).read_full()
    Vx_bin4s = [recon_binning(projs, theta, center, b_K, 4) for b_K in [4,6,8]]    
    Vx_bin2s = [recon_binning(projs, theta, center, b_K, 2) for b_K in [2,4,8]]    


    

    
    # item_list = [((32,32,32), "M_a07"), ((32,32,32), "M_a02")]
    item_list = [((32,32,32), "M_a08")]
    for item in item_list:
        TRAINING_INPUT_SIZE, model_tag = item
        training_params = get_training_params(TRAINING_INPUT_SIZE)
        print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
        model_params = get_model_params(model_tag)
        fe = SurfaceSegmenter(model_initialization = 'define-new', \
                                 descriptor_tag = model_tag, \
                                 **model_params)
        fit(fe, Xs, Ys)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
