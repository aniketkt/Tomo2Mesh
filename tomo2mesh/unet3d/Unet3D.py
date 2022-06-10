#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easily define U-net-like architectures using Keras layers

"""

import numpy as np
# from tensorflow import RunOptions
from tensorflow import keras
from tensorflow.keras.backend import random_normal
import tensorflow as tf
from tensorflow import map_fn, constant, reduce_max, reduce_min
from tensorflow.keras import layers as L

# tensorflow configs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def insert_activation(tensor_in, activation):
    """
    Returns
    -------
    tensor
        of rank 2 (FC layer), 4 (image) or 5 (volume) (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in : tensor
            input tensor
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
            
    """
    if activation is None:
        return tensor_in
    if activation == 'lrelu':
        tensor_out = L.LeakyReLU(alpha = 0.2)(tensor_in)
    else:
        tensor_out = L.Activation(activation)(tensor_in)
    
    return tensor_out
    
def custom_Conv3D(tensor_in, n_filters, kern_size, activation = None, batch_norm = False):
    
    """
    Define a custom 3D convolutional layer with batch normalization and custom activation function (includes lrelu)  

    This is the order chosen in our implementation:  
    
    -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->  
    
    See supmat in: https://dmitryulyanov.github.io/deep_image_prior

    Returns
    -------
    tensor
        of rank 5 (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_filters  : int
            number of filters in the first convolutional layer
    kern_size  : tuple
            kernel size, e.g. (3,3,3)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
            
    """
    
    tensor_out = L.Conv3D(n_filters, kern_size, activation = None, padding = "same")(tensor_in)
    
    if batch_norm:
        tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
    
    tensor_out = insert_activation(tensor_out, activation)
    return tensor_out
    
    
def analysis_block(tensor_in, n_filters, pool_size, \
                   kern_size = None, \
                   activation = None, \
                   batch_norm = False):

    """
    Define a block of 2 3D convolutional layers followed by a 3D max-pooling layer


    Returns
    -------
    tuple of two tensors (output, tensor to concatenate in synthesis path)
        of rank 5 (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_filters  : int
            number of filters in the first convolutional layer
    pool_size  : tuple
            max pooling e.g. (2,2,2)
    kern_size  : tuple
            kernel size, e.g. (3,3,3)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    kern_init  : str
            kernel initialization method
    batch_norm : bool
            True to insert a BN layer
            
    """
    
    # layer # 1
    tensor_out = custom_Conv3D(tensor_in, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    # layer # 2; 2x filters
    tensor_out = custom_Conv3D(tensor_out, 2*n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)

    assert type(pool_size) is int, "pool_size must be an integer"
    if pool_size > 1:
        # MaxPool3D
        return L.MaxPool3D(pool_size = pool_size, padding = "same")(tensor_out), tensor_out
    else:
        return tensor_out, tensor_out

def synthesis_block(tensor_in, n_filters, pool_size, \
                    concat_tensor = None, \
                    activation = None, \
                    kern_size = 3, \
                    kern_size_upconv = 2, \
                    batch_norm = False, \
                    concat_flag = True):
    """
    Define a 3D upsample block and concatenate the output of downsample block to it (skip connection)
    
    Returns
    -------
    tensor
        of rank 5 (batch_size, nz, ny, nx, n_channels)

    Parameters  
    ----------  
    tensor_in     : tensor  
            input tensor  
    concat_tensor : tensor  
            this will be concatenated to the output of the upconvolutional layer  
    n_filters  : int  
            number of filters in each convolutional layer after the transpose conv.  
    pool_size  : tuple  
            reverse the max pooling e.g. (2,2,2) with these many strides for transpose conv.  
    kern_size  : int  
            kernel size for conv, e.g. 3  
    kern_size_upconv  : int  
            kernel size for upconv, e.g. 2  
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
    concat_flag : bool
            True to concatenate layers (add skip connections)
    """
    assert type(pool_size) is int, "pool_size must be an integer"
    
    if pool_size > 1:
    
        # transpose convolution
        n_filters_upconv = tensor_in.shape[-1]
        tensor_out = L.Conv3DTranspose(n_filters_upconv, kern_size_upconv, padding = "same", activation = None, strides = pool_size) (tensor_in)
        tensor_out = insert_activation(tensor_out, activation)

        if concat_flag:
            tensor_out = L.concatenate([tensor_out, concat_tensor])
    else:
        tensor_out = tensor_in
    
    # layer # 1
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    # layer # 2
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    return tensor_out

    


def build_Unet_3D(n_filters = [16,32,64], \
                  n_blocks = 3, activation = 'lrelu',\
                  batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                  isconcat = None, pool_size = 2):
    """
    Define a 3D convolutional Unet, based on the arguments provided. Output image size is the same as input image size.  
    
    Returns
    -------
    tf.Keras.model
        keras model(s) for a 3D autoencoder-decoder architecture.  
        
    Parameters
    ----------
    vol_shape  : tuple
            input volume shape (nz,ny,nx,1)  
            
    n_filters : list
            a list of the number of filters in the convolutional layers for each block. Length must equal number of number of blocks.  
            
    n_blocks  : int
            Number of repeating blocks in the convolutional part  
            
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer  
            
    batch_norm : bool
            True to insert BN layer after the convolutional layers  
            
    kern_size  : tuple
            kernel size for conv. layers in downsampling block, e.g. (3,3,3).  
            
    kern_size_upconv  : tuple
            kernel size for conv. layers in upsampling block, e.g. (2,2,2).  
            
    isconcat : bool or list
            Selectively concatenate layers (skip connections)  
    
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
            
    """
    
    inp = L.Input((None,None,None,1))
    
    if isconcat is None:
        isconcat = [False]*n_blocks
    
    if type(pool_size) is int:
        pool_size = [pool_size]*n_blocks
    elif len(pool_size) != n_blocks:
        raise ValueError("list length must be equal to number of blocks")
            
    concats = []
    # downsampling path. e.g. n_blocks = 3, n_filters = [16,32,64], input volume is 64^3
    for ii in range(n_blocks): # 3 iterations
        
        if ii == 0:
            code = inp
            
        code, concat_tensor = analysis_block(code, \
                                             n_filters[ii], \
                                             pool_size[ii], \
                                             kern_size = kern_size, \
                                             activation = activation, \
                                             batch_norm = batch_norm)
        concats.append(concat_tensor)

    nf = code.shape[-1]
    code = custom_Conv3D(code, nf, kern_size, \
                         activation = activation, batch_norm = batch_norm)
    decoded = custom_Conv3D(code, 2*nf, kern_size, \
                         activation = activation, batch_norm = batch_norm)    
    
    # upsampling path. e.g. n_blocks = 3
    for ii in range(n_blocks-1, -1, -1):
        # ii is 2, 1, 0
        
        decoded = synthesis_block(decoded, \
                                  2*n_filters[ii], \
                                  pool_size[ii], \
                                  concat_tensor = concats[ii], \
                                  activation = activation, \
                                  kern_size = kern_size, \
                                  kern_size_upconv = kern_size_upconv, \
                                  batch_norm = batch_norm, \
                                  concat_flag = isconcat[ii])
        
    decoded = L.Conv3D(1, (1,1,1), activation = 'sigmoid', padding = "same")(decoded)
    
    segmenter = keras.models.Model(inp, decoded, name = "segmenter")
    
    return segmenter






    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

