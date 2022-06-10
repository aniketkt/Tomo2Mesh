#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""

import pandas as pd
import os
import glob
import numpy as np


# from skimage.feature import match_template
# from tomopy import normalize, minus_log, angles, recon, circ_mask
# from scipy.ndimage.filters import median_filter

from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
from tensorflow.keras.models import load_model
import functools
import cupy as cp
import time

MAX_ITERS = 2000 # iteration max for find_patches(). Will raise warnings if count is exceeded.
# Parameters for weighted cross-entropy and focal loss - alpha is higher than 0.5 to emphasize loss in "ones" or metal pixels.

from tomo_encoders.neural_nets.keras_processor import Vox2VoxProcessor_fCNN
from tomo_encoders.neural_nets.Unet3D import build_Unet_3D
from tomo_encoders.misc.voxel_processing import _rescale_data, edge_map



class SurfaceSegmenter(Vox2VoxProcessor_fCNN):
    
    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        return
    
    def train(self, Xs, Ys, input_size, max_stride, batch_size, cutoff, random_rotate, add_noise, n_epochs):
        
        '''
        Parameters  
        ----------  
        
        '''
        n_vols = len(Xs)
        # instantiate data generator for use in training.  
        dg = self.data_generator(Xs, Ys, batch_size, input_size, max_stride, cutoff, random_rotate, add_noise)
        
        tot_steps = 1000
        val_split = 0.2
        steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
        validation_steps = int(val_split*tot_steps//batch_size)

        t0 = time.time()
        self.models["segmenter"].fit(x = dg, epochs = n_epochs,\
                  steps_per_epoch=steps_per_epoch,\
                  validation_steps=validation_steps, verbose = 1)    
        t1 = time.time()
        training_time = (t1 - t0)
        print("training time = %.2f seconds"%training_time)        
        return
    
    def _find_edges(self, patches, cutoff, Y_gt, input_size):
        
        assert Y_gt.shape == patches.vol_shape, "volume of Y_gt does not match vol_shape"
        y_tmp = patches.extract(edge_map(Y_gt), input_size)[...,np.newaxis]
        cond_list = np.sum(y_tmp, axis = (1,2,3)) > np.prod(input_size)*cutoff
        return cond_list.astype(bool)
    
    def _find_blanks(self, patches, cutoff, Y_gt, input_size):
        
        assert Y_gt.shape == patches.vol_shape, "volume of Y_gt does not match vol_shape"
        y_tmp = patches.extract(Y_gt, input_size)[...,np.newaxis]
        ystd = np.std(y_tmp, axis = (1,2,3))

        cond_list = ystd > np.max(ystd)*cutoff
        return cond_list.astype(bool)
    
    
    def get_training_patches(self, vol_shape, batch_size, input_size, max_stride, cutoff, Y_gt):
        
        ip = 0
        tot_len = 0
        patches = None
        
#         print(f'vol_shape: {vol_shape}, batch_size: {batch_size}, input_size: {input_size}, max_stride: {max_stride}, cutoff: {cutoff}, Y_gt: {Y_gt}')
        
        while tot_len < batch_size:
            
            # count iterations
            assert ip <= MAX_ITERS, "stuck in loop while finding patches to train on"
            ip+= 1
            
            p_tmp = Patches(vol_shape, initialize_by = "random", \
                            min_patch_size = input_size, \
                            max_stride = max_stride, \
                            n_points = batch_size)    
            
            # remove patches that do not contain a surface / edge
            if cutoff > 0.0:
                cond_list = self._find_blanks(p_tmp, cutoff, Y_gt, input_size)
            else:
                cond_list = np.asarray([True]*len(p_tmp)).astype(bool)
            
            if np.sum(cond_list) > 0:
                # do stuff
                p_tmp = p_tmp.filter_by_condition(cond_list)

                if patches is None:
                    patches = p_tmp.copy()
                else:
                    patches.append(p_tmp)
                    tot_len = len(patches)
            else:
                continue
        
        assert patches is not None, "get_patches() failed to return any patches with selected conditions"
        patches = patches.select_random_sample(batch_size)
        return patches

    def save_models(self, model_path):
        
        model_key = "segmenter"
        model = self.models[model_key]
        filepath = os.path.join(model_path, "%s_%s.hdf5"%(model_key, self.model_tag))
        tf.keras.models.save_model(model, filepath, include_optimizer=False)        
        return
    
    def _load_models(self, model_names = None, model_path = 'some/path'):
        
        '''
        Parameters
        ----------
        model_names : dict
            example {"segmenter" : "Unet"}
        model_path : str  
            example "some/path"
        custom_objects_dict : dict  
            dictionary of custom objects (usually pickled with the keras models)
            
        '''
        self.models = {} # clears any existing models linked to this class!!!!
        for model_key, model_name in model_names.items():
            self.models.update({model_key : \
                                load_model(os.path.join(model_path, \
                                                        model_name + '.hdf5'))})
        model_key = "segmenter"
        self.model_tag = "_".join(model_names[model_key].split("_")[1:])
        return
    


    def data_generator(self, Xs, Ys, batch_size, input_size, cutoff, random_rotate, add_noise):

        '''
        
        Parameters  
        ----------  
        vol : np.array  
            Volume from which patches are extracted.  
        batch_size : int  
            Size of the batch generated at every iteration.  
        sampling_method : str  
            Possible methods include "random", "random-fixed-width", "grid"  
        
        '''
        
        while True:
            n_vols = len(Xs)
            idx_vols = np.repeat(np.arange(0, n_vols), int(np.ceil(batch_size/n_vols)))
            idx_vols = idx_vols[:batch_size]
            
            xy = []
            for ivol in range(n_vols):
                
                Y_gt = Ys[ivol] if len(Ys) > 1 else Ys[0]
                sub_batch_size = np.sum(idx_vols == ivol)
#                 print(f"ivol: {ivol}, sub_batch_size: {sub_batch_size}") 
                if sub_batch_size < 1:
                    continue
                
                patches = self.get_training_patches(Xs[ivol].shape, \
                                                    sub_batch_size, \
                                                    input_size, \
                                                    max_stride, \
                                                    cutoff, \
                                                    Y_gt)
                
                xy.append(self.extract_training_patch_pairs(Xs[ivol], Y_gt, \
                                                            patches, add_noise, \
                                                            random_rotate, input_size))
                
            yield np.concatenate([_xy[0] for _xy in xy], axis = 0, dtype = 'float32'), np.concatenate([_xy[1] for _xy in xy], axis = 0, dtype = 'uint8')



    def _data_generator(self, Xs, Ys, batch_size, input_size, max_stride, cutoff, random_rotate, add_noise):
        
        '''
        
        Parameters  
        ----------  
        vol : np.array  
            Volume from which patches are extracted.  
        batch_size : int  
            Size of the batch generated at every iteration.  
        sampling_method : str  
            Possible methods include "random", "random-fixed-width", "grid"  
        max_stride : int  
            If method is "random" or "multiple-grids", then max_stride is required.  
        
        '''
        
        while True:
            n_vols = len(Xs)
            idx_vols = np.repeat(np.arange(0, n_vols), int(np.ceil(batch_size/n_vols)))
            idx_vols = idx_vols[:batch_size]
            
            xy = []
            for ivol in range(n_vols):
                
                Y_gt = Ys[ivol] if len(Ys) > 1 else Ys[0]
                sub_batch_size = np.sum(idx_vols == ivol)
#                 print(f"ivol: {ivol}, sub_batch_size: {sub_batch_size}") 
                if sub_batch_size < 1:
                    continue
                
                patches = self.get_training_patches(Xs[ivol].shape, \
                                                    sub_batch_size, \
                                                    input_size, \
                                                    max_stride, \
                                                    cutoff, \
                                                    Y_gt)
                
                xy.append(self.extract_training_patch_pairs(Xs[ivol], Y_gt, \
                                                            patches, add_noise, \
                                                            random_rotate, input_size))
                
            yield np.concatenate([_xy[0] for _xy in xy], axis = 0, dtype = 'float32'), np.concatenate([_xy[1] for _xy in xy], axis = 0, dtype = 'uint8')
    
    def predict_patches(self, model_key, x, chunk_size, out_arr, min_max = None):

        '''
        Predicts sub_vols. This is a wrapper around keras.model.predict() that speeds up inference on inputs lengths that are not factors of 2. Use this function to do multiprocessing if necessary.  
        
        '''
        assert x.ndim == 5, "x must be 5-dimensional (batch_size, nz, ny, nx, 1)."
        
        t0 = time.time()
#         print("call to predict_patches, len(x) = %i, shape = %s, chunk_size = %i"%(len(x), str(x.shape[1:-1]), chunk_size))
        nb = len(x)
        nchunks = int(np.ceil(nb/chunk_size))
        nb_padded = nchunks*chunk_size
        padding = nb_padded - nb

        if out_arr is None:
            out_arr = np.empty_like(x) # use numpy since return from predict is numpy
        else:
            # to-do: check dims
            assert out_arr.shape == x.shape, "x and out_arr shapes must be equal and 4-dimensional (batch_size, nz, ny, nx, 1)"

        for k in range(nchunks):

            sb = slice(k*chunk_size , min((k+1)*chunk_size, nb))
            x_in = x[sb,...]

            if min_max is not None:
                x_in = np.clip(x_in, *min_max)
                min_val, max_val = min_max
                x_in = _rescale_data(x_in, float(min_val), float(max_val))
            
            if padding != 0:
                if k == nchunks - 1:
                    x_in = np.pad(x_in, \
                                  ((0,padding), (0,0), \
                                   (0,0), (0,0), (0,0)), mode = 'edge')
                
                x_out = self.models[model_key].predict(x_in)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                x_out = self.models[model_key].predict(x_in)
            out_arr[sb,...] = x_out
        
        out_arr = np.round(out_arr).astype(np.uint8)
        t_unit = (time.time() - t0)*1000.0/nb
        
        return out_arr
    
            
    def _build_models(self, descriptor_tag = "misc", **model_params):
        '''
        
        Implementation of Segmenter_fCNN that removes blank volumes during training.  
        Parameters
        ----------
        model_keys : list  
            list of strings describing the model, e.g., ["segmenter"], etc.
        model_params : dict
            for passing any number of model hyperparameters necessary to define the model(s).
            
        '''
        if model_params is None:
            raise ValueError("Need model hyperparameters or instance of model. Neither were provided")
        else:
            self.models = {}

        # insert your model building code here. The models variable must be a dictionary of models with str descriptors as keys
            
        self.model_tag = "Unet_%s"%(descriptor_tag)

        model_key = "segmenter"
        self.models.update({model_key : None})
        # input_size here is redundant if the network is fully convolutional
        self.models[model_key] = build_Unet_3D(**model_params)
        self.models[model_key].compile(optimizer=tf.keras.optimizers.Adam(),\
                                         loss= tf.keras.losses.BinaryCrossentropy())
        return
            
    def random_data_generator(self, batch_size, input_size = (64,64,64)):

        while True:

            x_shape = tuple([batch_size] + list(input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            y = np.random.randint(0, 2, x_shape)#.astype(np.uint8)
            x[x == 0] = 1.0e-12
            yield x, y
            
    def test_speeds(self, chunk_size, n_reps = 3, input_size = None, model_key = "segmenter"):
        
        if input_size is None:
            input_size = (64,64,64)
        for jj in range(n_reps):
            x = np.random.uniform(0, 1, tuple([chunk_size] + list(input_size) + [1])).astype(np.float32)
            t0 = time.time()
            y_pred = self.predict_patches(model_key, x, chunk_size, None, min_max = (-1,1))

            t_unit = (time.time() - t0)*1000.0/len(x)
            print(f"inf. time per patch {input_size} = {t_unit:.2f} ms, nb = {len(x)}")
            print(f"inf. time per voxel {(t_unit/(np.prod(input_size))*1.0e6):.2f} ns")
            print("\n")
            
        return
            
            
            
if __name__ == "__main__":
    
    print('just a bunch of functions')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
