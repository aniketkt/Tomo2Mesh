#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import tensorflow as tf
# from cupyx.scipy.fft import rfft, irfft, rfftfreq

from cupyx.scipy.fft import rfft, irfft, rfftfreq, get_fft_plan
from cupyx.scipy import ndimage
from tomo2mesh.fbp.retrieve_phase import paganin_filter
from tomo2mesh.misc.voxel_processing import TimerGPU
def calc_padding(data_shape):
    # padding, make sure the width of projection is divisible by four after padding
    [ntheta, nz, n] = data_shape
    n_pad = n*(1 + 0.25*2) # 1/4 padding
    n_pad = int(np.ceil(n_pad/8.0)*8.0) 
    pad_left = int((n_pad - n)//2)
    pad_right = n_pad - n - pad_left    
    
    # print(f'n: {n}, n_pad: {n_pad}')
    # print(f'pad_left: {pad_left}, pad_right: {pad_right}')    
    return pad_left, pad_right

def fbp_filter(data):
    """FBP filtering of projections"""
    
    timer = TimerGPU("ms")
    timer.tic()
    pad_left, pad_right = calc_padding(data.shape)    
    # padding
    data_padded = cp.pad(data, ((0,0),(0,0),(pad_left, pad_right)), mode = 'edge')

    # fft plan
    plan_fwd = get_fft_plan(data_padded, axes=2, value_type='R2C')
    plan_inv = get_fft_plan(rfft(data_padded,axis=2), axes=2, value_type='C2R')
    
    with plan_fwd:

        # filter mask
        t = rfftfreq(data_padded.shape[2])
        wfilter = t.astype(cp.float32) #* (1 - t * 2)**3  # parzen
        # fft
        data0 = wfilter*rfft(data_padded, axis=2)
        
    with plan_inv:
        # inverse fft
        data[:] = irfft(data0, axis=2)[...,pad_left:-pad_right]
        
    t_gpu = timer.toc()
    
    return t_gpu

def preprocess(data, dark, flat, outlier_kernel = 1):

    '''
    Parameters
    ----------
    data : np.ndarray  
        data has shape ntheta, nz, n  
        
    '''
    
    data = (data-dark)/(cp.maximum(flat-dark, 1.0e-6))                
    
    o = outlier_kernel
    fdata = ndimage.median_filter(data,[o,o,o])
    ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
    data[ids] = fdata[ids]        
    
    if 0:
        data = paganin_filter(data, alpha = 0.001, energy = 30.0, pixel_size = 3.10e-04)

    data = -cp.log(cp.maximum(data,1.0e-6))
    
    return data



if __name__ == "__main__":
    
    print('just a bunch of functions')

    
