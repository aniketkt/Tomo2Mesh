import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo2mesh import DataFile
import tensorflow as tf
import time

from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.projects.eaton.params2 import *
from tomo2mesh.misc.voxel_processing import normalize_volume_gpu

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

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

data_list = ['b1_sample1_layer3', 'b1_sample1_layer4', 'b1_sample2_layer4']
if __name__ == "__main__":

    Xs = []
    Ys = []
    # Xs is a list of some volumes of some shape (nz, n, n) and Ys is a corresponding list of segmented volumes.
    # Each element in X could be a binned volume (b=4) or a full volume (b=1)


    for data_item in data_list:
        rec_path = os.path.join('/data01/Eaton_Polymer_AM/unet_data/', "test_data_rec_" + data_item)
        seg_path = os.path.join('/data01/Eaton_Polymer_AM/unet_data/', "test_data_seg_" + data_item)
        Vx = DataFile(rec_path, tiff = True).read_full()
        Vx = normalize_volume_gpu(Vx, chunk_size=1, use_autocontrast=True)

        Vy = DataFile(seg_path, tiff = True).read_full()
        Xs.append(Vx)
        Ys.append(Vy)
    
    TRAINING_INPUT_SIZE = (128,128,128) #(32,32,32)
    model_tag = "M_b02" 
    training_params = get_training_params(TRAINING_INPUT_SIZE)
    print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
    model_params = get_model_params(model_tag)
    fe = SurfaceSegmenter(model_initialization = 'define-new', \
                                descriptor_tag = model_tag, \
                                **model_params)
    fit(fe, Xs, Ys)