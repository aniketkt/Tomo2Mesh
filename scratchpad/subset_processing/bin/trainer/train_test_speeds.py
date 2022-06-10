#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import h5py 
from tomo_encoders import DataFile, Patches
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import os 
import tqdm
import pandas as pd

import matplotlib as mpl
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import *


#### THIS EXPERIMENT ####

patch_size = tuple([int(sys.argv[1])]*3)
chunk_size = int(sys.argv[2])
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 512
model_tag = sys.argv[4] if len(sys.argv) > 4 else "M_a01"
######### DEFINE EXPERIMENT ON 'nb'
def infer(fe):

#     Possible slowdown of first iteration due to tensorflow Dataset creation?
#     https://github.com/tensorflow/tensorflow/issues/46950
    fe.test_speeds(chunk_size,n_reps = 3, input_size = patch_size)

    print("#"*55,"\n")
    for jj in range(3):
        x = np.random.uniform(0, 1, tuple([nb]+list(patch_size)+[1])).astype(np.float32)
        t0 = time.time()
        y_pred = fe.predict_patches("segmenter", x, chunk_size, None, \
                                                min_max = (-1,1))
        t_unit = (time.time() - t0)*1.0e3/nb
        n_voxels = np.prod(patch_size)
        print(f"inf. time per patch {patch_size} = {t_unit:.2f} ms, nb = {nb}")
        print(f"inf. time per voxel {(t_unit/n_voxels*1.0e6):.2f} ns")
        print("\n")

def fit(fe):
    
    batch_size = training_params["batch_size"]
    n_epochs = training_params["n_epochs"]
    
    dg = fe.random_data_generator(batch_size)
    
    t0 = time.time()
    tot_steps = 1000
    val_split = 0.2
    steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
    validation_steps = int(val_split*tot_steps//batch_size)
    
    fe.models["segmenter"].fit(x = dg, epochs = n_epochs, batch_size = batch_size,\
              steps_per_epoch=steps_per_epoch,\
              validation_steps=validation_steps, verbose = 1)    
    t1 = time.time()
    training_time = (t1 - t0)
    print("training time per epoch = %.2f seconds"%(training_time/n_epochs))        
    return
        
    
if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    fe = SurfaceSegmenter(model_initialization = 'define-new', \
                         descriptor_tag = model_tag,\
                         **model_params) 
    print("EXPERIMENT WITH INPUT_SIZE = ", patch_size)
#     fe.print_layers("segmenter")    
#     fe.models["segmenter"].summary()

    infer(fe)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
