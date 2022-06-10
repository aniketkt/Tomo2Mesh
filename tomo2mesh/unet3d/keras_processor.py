#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""
from abc import abstractmethod
import pandas as pd
import os
import glob
import numpy as np


from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
from tensorflow.keras.models import load_model
import functools
import cupy as cp
import time
from tomo_encoders.misc.voxel_processing import _rescale_data, _find_min_max


class GenericKerasProcessor():
    
    def __init__(self,\
                 model_initialization = "define-new", \
                 descriptor_tag = "misc", **kwargs):
        
        '''
        
        Parameters
        ----------
        model_initialization : str
            either "define-new" or "load-model"
        
        descriptor_tag : str
            some description used while saving the models

        model_params : dict
            If "define-new", this contains the hyperparameters that define the architecture of the neural network model. If "load-model", it contains the paths to the models.
            
        models : dict of tf.keras.Models.model 
            example dict contains {"segmenter" : segmenter}

        '''

        # could be "data" or "labels" or "embeddings"
        self.input_type = "data"
        self.output_type = "data"
        
        model_getters = {"load-model" : self._load_models, \
                         "define-new" : self._build_models}

        # any function chosen must assign self.models, self.model_tag
        
        if model_initialization == "define-new":
            model_getters[model_initialization](descriptor_tag = descriptor_tag, \
                                                **kwargs)
        elif model_initialization == "load-model":
            model_getters[model_initialization](**kwargs)
        else:
            raise NotImplementedError("method is not implemented")
            
        return

    def print_layers(self, modelkey):
        
        txt_out = []
        for ii in range(len(self.models[modelkey].layers)):
            lshape = str(self.models[modelkey].layers[ii].output_shape)
            lname = str(self.models[modelkey].layers[ii].name)
            txt_out.append(lshape + "    ::    "  + lname)
        print('\n'.join(txt_out))
        return
    
    def predict_patches(self, model_key, x, chunk_size, out_arr, \
                         min_max = None, \
                         TIMEIT = False):

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
                min_val, max_val = min_max
                x_in = _rescale_data(x_in, float(min_val), float(max_val))
            
            if padding != 0:
                if k == nchunks - 1:
                    x_in = np.pad(x_in, \
                                  ((0,padding), (0,0), \
                                   (0,0), (0,0), (0,0)), mode = 'edge')
                
                if model_key == "autoencoder":
                    z = self.models["encoder"].predict(x_in)
                    x_out = self.models["decoder"].predict(z)
                else:
                    
                    x_out = self.models[model_key].predict(x_in)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                if self.output_type == "autoencoder":
                    z = self.models["encoder"].predict(x_in)
                    x_out = self.models["decoder"].predict(z)
                else:
                    x_out = self.models[model_key].predict(x_in)
            out_arr[sb,...] = x_out
        
        if self.output_type == "labels":
            out_arr = np.round(out_arr).astype(np.uint8)
        elif self.output_type == "embeddings":
            out_arr = np.round(out_arr).astype(x.dtype)
            
        t_unit = (time.time() - t0)*1000.0/nb
        
        if TIMEIT:
            print("inf. time p. input patch size %s = %.2f ms, nb = %i"%(str(x[0,...,0].shape), t_unit, nb))
            print("\n")
            return out_arr, t_unit
        else:
            return out_arr
    
    def calc_voxel_min_max(self, vol, sampling_factor, TIMEIT = False):

        '''
        returns min and max values for a big volume sampled at some factor
        '''

        return _find_min_max(vol, sampling_factor, TIMEIT = TIMEIT)
    
    def rescale_data(self, data, min_val, max_val):
        '''
        Recales data to values into range [min_val, max_val]. Data can be any numpy or cupy array of any shape.  

        '''
        xp = cp.get_array_module(data)  # 'xp' is a standard usage in the community
        eps = 1e-12
        data = (data - min_val) / (max_val - min_val + eps)
        return data
    
    def _msg_exec_time(self, func, t_exec):
        print("TIME: %s: %.2f seconds"%(func.__name__, t_exec))
        return

    @abstractmethod
    def test_speeds(self):
        pass
    
    @abstractmethod
    def save_models(self):
        pass
    
    @abstractmethod
    def _build_models(self):
        pass
        
    @abstractmethod
    def _load_models(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def get_patches(self):
        pass
    
    @abstractmethod
    def data_generator(self):
        pass
    
# examples: segmenter, enhancer, denoiser, etc.    
class Vox2VoxProcessor_fCNN(GenericKerasProcessor):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def test_speeds(self, chunk_size, n_reps = 3, input_size = None, model_key = "segmenter"):
        
        if input_size is None:
            input_size = (64,64,64)
        for jj in range(n_reps):
            x = np.random.uniform(0, 1, tuple([chunk_size] + list(input_size) + [1])).astype(np.float32)
            y_pred = self.predict_patches(model_key, x, chunk_size, None, min_max = (-1,1), TIMEIT = True)
        return

    def extract_training_patch_pairs(self, X, Y, patches, add_noise, random_rotate, input_size):
        '''
        Extract training pairs x and y from a given volume X, Y pair
        '''
        
        batch_size = len(patches)
        y = patches.extract(Y, input_size)[...,np.newaxis]            
        x = patches.extract(X, input_size)[...,np.newaxis]
        std_batch = np.random.uniform(0, add_noise, batch_size)
        x = x + np.asarray([np.random.normal(0, std_batch[ii], x.shape[1:]) for ii in range(batch_size)])

        if random_rotate:
            nrots = np.random.randint(0, 4, batch_size)
            for ii in range(batch_size):
                axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
                x[ii, ..., 0] = np.rot90(x[ii, ..., 0], k=nrots[ii], axes=axes)
                y[ii, ..., 0] = np.rot90(y[ii, ..., 0], k=nrots[ii], axes=axes)
#         print("DEBUG: shape x %s, shape y %s"%(str(x.shape), str(y.shape)))
        
        return x, y

    def random_data_generator(self, batch_size, input_size = (64,64,64)):

        while True:

            x_shape = tuple([batch_size] + list(input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            y = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            x[x == 0] = 1.0e-12
            y[y == 0] = 1.0e-12
            yield x, y
    
class EmbeddingLearner(GenericKerasProcessor):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        return

    @abstractmethod
    def predict_embeddings(self):
        pass
        
        
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
