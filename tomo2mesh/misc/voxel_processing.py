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


# from skimage.feature import match_template
# from tomopy import normalize, minus_log, angles, recon, circ_mask
# from scipy.ndimage.filters import median_filter


import tensorflow as tf
from multiprocessing import Pool, cpu_count
import functools
import cupy as cp
import h5py
import abc
import time


class TimerGPU():

    def __init__(self, unit):
        self.unit = unit
        pass

    def tic(self):
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()
        self.start.record()
        return
    
    def toc(self, msg = "execution"):
        self.end.record()
        self.end.synchronize()
        if self.unit == "secs":
            t_elapsed = cp.cuda.get_elapsed_time(self.start, self.end)/1000.0
        else:
            t_elapsed = cp.cuda.get_elapsed_time(self.start, self.end)

        if msg != "execution":
            print(f"\tTIME: {msg} {t_elapsed:.2f} {self.unit}")
        return t_elapsed

class TimerCPU():

    def __init__(self, unit):
        self.unit = unit
        pass

    def tic(self):
        self.start = time.time()
        return
    
    def toc(self, msg = "execution"):
        t_elapsed = time.time() - self.start
        if self.unit == "ms":
            t_elapsed *= 1000.0

        if msg != "execution":
            print(f"\tTIME: {msg} {t_elapsed:.2f} {self.unit}")
        return t_elapsed


def _msg_exec_time(func, t_exec):
    if type(func) is str:
        print("TIME: %s: %.2f seconds"%(func, t_exec))
    else:
        print("TIME: %s: %.2f seconds"%(func.__name__, t_exec))
    return


def _rescale_data(data, min_val, max_val):
    '''
    Recales data to values into range [min_val, max_val]. Data can be any numpy or cupy array of any shape.  

    '''
    xp = cp.get_array_module(data)  # 'xp' is a standard usage in the community
    eps = 1e-12
    data = (data - min_val) / (max_val - min_val + eps)
    return data

def _find_min_max(vol, sampling_factor, TIMEIT = False):

    '''
    returns min and max values for a big volume sampled at some factor
    '''
    t0 = time.time()
    ss = slice(None, None, sampling_factor)
    xp = cp.get_array_module(vol[ss,ss,ss])  # 'xp' is a standard usage in the community
    max_val = xp.max(vol[ss,ss,ss])
    min_val = xp.min(vol[ss,ss,ss])
    tot_time = time.time() - t0
    if TIMEIT:
        _msg_exec_time("find voxel min max", tot_time)            
    return min_val, max_val

def edge_map(Y):

    '''
    this algorithm was inspired by: https://github.com/tomochallenge/tomochallenge_utils/blob/master/foam_phantom_utils.py
    '''
    xp = cp.get_array_module(Y)
    msk = xp.zeros_like(Y)
    tmp = Y[:-1]!=Y[1:]
    msk[:-1][tmp] = 1
    msk[1:][tmp] = 1
    tmp = Y[:,:-1]!=Y[:,1:]
    msk[:,:-1][tmp] = 1
    msk[:,1:][tmp] = 1
    tmp = Y[:,:,:-1]!=Y[:,:,1:]
    msk[:,:,:-1][tmp] = 1
    msk[:,:,1:][tmp] = 1
    return msk > 0



def cylindrical_mask_slice(out_img, mask_fac, mask_val = 0):

    out_vol = out_img[np.newaxis,...].copy()

    cylindrical_mask(out_vol, mask_fac, mask_val = mask_val)
    out_img[:] = out_vol[0]
    return
    



def cylindrical_mask(out_vol, mask_fac, mask_val = 0):
    
    xp = cp.get_array_module(out_vol)

    vol_shape = out_vol.shape
    assert vol_shape[1] == vol_shape[2], "must be a tomographic volume where shape y = shape x"
    
    shape_yx = vol_shape[1]
    shape_z = vol_shape[0]
    rad = int(mask_fac*shape_yx/2)
    
    pts = xp.arange(-int(shape_yx//2), int(xp.ceil(shape_yx//2)))
    yy, xx = xp.meshgrid(pts, pts, indexing = 'ij')
    circ = (xp.sqrt(yy**2 + xx**2) < rad).astype(xp.uint8) # inside is positive
    circ = circ[xp.newaxis, ...]
    cyl = xp.repeat(circ, shape_z, axis = 0)
    out_vol[cyl == 0] = mask_val
    return

def get_values_cyl_mask(vol, mask_fac):

    xp = cp.get_array_module(vol)
    vol_shape = vol.shape
    assert vol_shape[1] == vol_shape[2], "must be a tomographic volume where shape y = shape x"
    
    shape_yx = vol_shape[1]
    shape_z = vol_shape[0]
    rad = int(mask_fac*shape_yx/2)
    
    pts = xp.arange(-int(shape_yx//2), int(xp.ceil(shape_yx//2)))
    yy, xx = xp.meshgrid(pts, pts, indexing = 'ij')
    circ = (xp.sqrt(yy**2 + xx**2) < rad).astype(xp.uint8) # inside is positive
    circ = circ[xp.newaxis, ...]
    cyl = xp.repeat(circ, shape_z, axis = 0)
    return vol[cyl > 0]


def modified_autocontrast(vol, s = 0.01, normalize_sampling_factor = 2):
    
    '''
    Returns
    -------
    tuple
        alow, ahigh values to clamp data  
    
    Parameters
    ----------
    s : float
        quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels
    
    '''
    xp = cp.get_array_module(vol)
    

    sbin = slice(None, None, normalize_sampling_factor)
    
    if vol.ndim == 4:
        intensity_vals = vol[:, sbin, sbin, sbin].reshape(-1)
    elif vol.ndim == 3:
        intensity_vals = vol[sbin, sbin, sbin].reshape(-1)
    elif vol.ndim == 1:
        intensity_vals = vol
    
    data_type  = xp.asarray(intensity_vals).dtype
    
    if type(s) == tuple and len(s) == 2:
        slow, shigh = s
    else:
        slow = s
        shigh = s

    h, bins = xp.histogram(intensity_vals, bins = 500)
    c = xp.cumsum(h)
    c_norm = c/xp.max(c)
    
    ibin_low = xp.argmin(xp.abs(c_norm - slow))
    ibin_high = xp.argmin(xp.abs(c_norm - 1 + shigh))
    
    alow = bins[ibin_low]
    ahigh = bins[ibin_high]
    
    return alow, ahigh


def normalize_volume_gpu(vol, chunk_size = 64, normalize_sampling_factor = 1, TIMEIT = False):
    '''
    Normalizes volume to values into range [0,1]  

    '''

    tot_len = vol.shape[0]
    nchunks = int(np.ceil(tot_len/chunk_size))
    min_val, max_val = _find_min_max(vol, normalize_sampling_factor)
    
    proc_times = []
    copy_to_times = []
    copy_from_times = []
    stream1 = cp.cuda.Stream()
    t0 = time.time()
    
    vol_gpu = cp.zeros((chunk_size, vol.shape[1], vol.shape[2]), dtype = cp.float32)
    for jj in range(nchunks):
        t01 = time.time()
        sz = slice(jj*chunk_size, min((jj+1)*chunk_size, tot_len))
        
        
        ## copy to gpu from cpu
        with stream1:    
            vol_gpu.set(vol[sz,...])
        stream1.synchronize()    
        t02 = time.time()
        copy_to_times.append(t02-t01)
        
        ## process
        with stream1:
             vol_gpu = _rescale_data(vol_gpu, min_val, max_val)
        stream1.synchronize()
        t03 = time.time()
        proc_times.append(t03-t02)
        
        ## copy from gpu to cpu
        with stream1:
            vol[sz,...] = vol_gpu.get()            
        stream1.synchronize()    
        t04 = time.time()
        copy_from_times.append(t04 - t03)
    
    if TIMEIT:
        print(f"time for normalizing volume: {float(time.time() - t0):.2f} secs")
    
    return vol



    
    

if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
