#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import h5py
# from cupyx.scipy.fft import rfft, irfft, rfftfreq

from cupy.fft import rfft, irfft, rfftfreq
from tomo_encoders import Patches
from tomo_encoders.reconstruction.recon import *
from tomo_encoders.misc.voxel_processing import modified_autocontrast, cylindrical_mask

def solver(projs, theta, center, dark, flat, \
           apply_fbp = True, \
           apply_darkflat_correction = True, \
           apply_minus_log = True,\
           normalize_sampling_factor = 2,\
           contrast_adjust_factor = 0.0, \
           mask_ratio = 0.95, \
           TIMEIT = False, \
           PLOTIT = False):
    
    '''
    reconstruct on epics pv stream. assumed data is on cpu in numpy arrays.  
    
    Parameters
    ----------
    projs : np.ndarray  
        array of projection images shaped as ntheta, nrows, ncols
    theta : np.ndarray
        array of theta values (length = ntheta)  
    center : float  
        center value for the projection data  
    
    Returns
    -------
    np.ndarray  
        vol, reconstructed object in a 3-d numpy voxel array (nz, ny, nx)  
        
    '''
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    
    stream_copy = cp.cuda.Stream()
    with stream_copy:
        data = cp.array(projs, dtype = 'float32')
        vol_shape = (data.shape[1], data.shape[2], data.shape[2])
        theta = cp.array(theta, dtype = 'float32')
        center = cp.float32(center)
        
        if apply_darkflat_correction:
            dark = cp.array(dark)
            flat = cp.array(flat)
            data = darkflat_correction(data, dark, flat)
        
        if apply_minus_log:
            data = minus_log(data)
        
        # PADDING
        # make sure the width of projection is divisible by four after padding
        proj_w = data.shape[-1]
        tot_width = proj_w*(1 + 0.25*2) # 1/4 padding
        tot_width = int(np.ceil(tot_width/8.0)*8.0) 
        padding = int((tot_width - proj_w)//2)
        padding_right = tot_width - data.shape[-1] - padding
        data = cp.pad(data, ((0,0),(0,0),(padding, padding_right)), mode = 'edge')
        stream_copy.synchronize()
    
    if apply_fbp:
        data = fbp_filter(data) # need to apply filter to full projection  
        
    # st* - start, p* - number of points
    stz, sty, stx = (0,0,0)
    pz, py, px = vol_shape
    st = time.time()
    obj = rec_patch(data, theta, center+padding, \
              stx+padding, px, \
              sty+padding, py, \
              0,           pz) # 0 since projections were cropped vertically
    
    device.synchronize()
#     print('total bytes: ', memory_pool.total_bytes())    
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        print("TIME reconstruction: %.2f ms"%t_gpu)
    
    vol_rec = obj.get()
    del obj
    
    if contrast_adjust_factor > 0.0:
        clip_vals = modified_autocontrast(vol_rec, \
                                          s = contrast_adjust_factor, \
                                          normalize_sampling_factor = normalize_sampling_factor)
        vol_rec = np.clip(vol_rec, *clip_vals)

    if mask_ratio < 1.0:
        s_ = slice(None,None,normalize_sampling_factor)
        cylindrical_mask(vol_rec, mask_ratio, mask_val = np.min(vol_rec[s_,s_,s_]))
        
    if PLOTIT:
        fig, ax = plt.subplots(1,3, figsize = (14,6))
        ax[0].imshow(vol_rec[int(vol_rec.shape[0]*0.2)], cmap = 'gray')
        ax[1].imshow(vol_rec[int(vol_rec.shape[0]*0.5)], cmap = 'gray')
        ax[2].imshow(vol_rec[int(vol_rec.shape[0]*0.8)], cmap = 'gray')            
        plt.show()
        plt.close()
        
    return vol_rec

if __name__ == "__main__":
    
    print('just a bunch of functions')

    
