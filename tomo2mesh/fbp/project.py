#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import h5py
import signal 

import tomocg as pt
import os


# def fast_pad(vol, padding, padding_z):

#     new_vol = []
    
#     while True:
#         img, vol = pad_one(vol, padding)
#         new_vol.append()

# def pad_one(vol, padding):
#     img = vol[0]
    
#     img = cp.pad(cp.array(img), ((padding,padding),(padding,padding))).get()
#     return img, vol[1:] if vol[1:].shape[0] > 0 else None
    

def get_projections(vol, theta, center, pnz):
    
    print("vol shape for projection compute", vol.shape)
    # make sure the width of projection is divisible by four after padding
    proj_w = vol.shape[-1]
    tot_width = int(proj_w*(1 + 0.25*2)) # 1/4 padding
    tot_width = int(np.ceil(tot_width/8)*8) 
    padding = int((tot_width - proj_w)//2)
    
    # make sure the height of projection is divisible by pnz  
    proj_h = vol.shape[0]
    tot_ht = int(np.ceil(proj_h/(2*pnz))*(2*pnz)) 
    padding_z = int(tot_ht - proj_h)
    
    
    vol = np.pad(vol, ((0,padding_z),\
                       (padding,padding),\
                       (padding,padding)), mode = 'edge')
    
    nz = vol.shape[0] 
    n = vol.shape[-1] 
    ntheta = theta.size
    center_padded = center + padding
    r = nz//2 # +overlap? because some border pixels are being missed by solver 
    s1 = slice(None,r,None) 
    s2 = slice(-r,None,None) 
    u0 = vol[s1] + 1j*vol[s2] 
    
    ngpus=1 
    # Class gpu solver 
    t0 = time.time() 
    with pt.SolverTomo(theta, ntheta, r, n, pnz, center_padded, ngpus) as slv: 
        # generate data 
        data = slv.fwd_tomo_batch(u0) 
        projs = np.zeros((ntheta, nz, n), dtype = 'float32') 
        projs[:,s1,:] = data.real 
        projs[:,s2,:] = data.imag 

    print("padded projections shape: %s"%str(projs.shape)) 
    
    if padding_z == 0:
        projs = projs[:, :, padding:-padding]     
    else:
        projs = projs[:, :-padding_z, padding:-padding]     
    
    print("time %.4f"%(time.time()- t0)) 
    print("projections shape: %s"%str(projs.shape)) 

    return projs, theta, center


def acquire_data(vol_full, projs_path, point, ntheta, FOV = (1920,1200), pnz = 4):
    
    '''
    Parameters
    ----------
    vol_full : np.ndarray  
        vol object having shape nz, ny, nx  
    point : tuple  
        iz, iy, ix coordinate where scan will be centered  
    FOV : tuple  
        projs_w, projs_h width and height of projections array
    
    '''
    
    
    assert vol_full.shape[1] == vol_full.shape[2], "axes 1 and 2 must be equal length - input must be tomographic volume"
    assert vol_full.shape[1]%2 == 0, "horizontal slice shape must be even"  
    
    iz, iy, ix = point
    projs_w, projs_h = FOV
    
    sz = slice(iz - projs_h//2, iz + projs_h//2)
    sy = slice(iy - projs_w//2, iy + projs_w//2)
    sx = slice(ix - projs_w//2, ix + projs_w//2)
    
    vol = vol_full[sz, sy, sx]
    assert vol.ndim == 3, "sliced volume is not three-dimensional. why?"
    assert vol.shape[0] == projs_h, "z axis of sliced volume does not match projs_h"
    assert vol.shape[1] == projs_w, "y axis of sliced volume does not match projs_w"
    assert vol.shape[2] == projs_w, "x axis of sliced volume does not match projs_w"

    theta = np.linspace(0,np.pi,ntheta,dtype='float32') 
    center = projs_w/2.0 

    
    projs, theta, center = get_projections(vol, theta, center, pnz)
    
    
    if projs_path is not None:
        save_fpath = os.path.join(projs_path, \
                                  'projs_point_z%i_y%i_x%i_ntheta%i_%ix%i.hdf5'%(iz, \
                                                                                 iy, ix, \
                                                                                 ntheta, FOV[1], FOV[0]))
        with h5py.File(save_fpath, 'w') as hf:
            hf.create_dataset('data', data = projs)
            hf.create_dataset('theta', data = theta)
            hf.create_dataset('center', data = center)
    
    return projs, theta, center

def read_data(projs_path, point, ntheta, FOV = (1920,1200)):
    
    iz, iy, ix = point
    read_fpath = os.path.join(projs_path, \
                              'projs_point_z%i_y%i_x%i_ntheta%i_%ix%i.hdf5'%(iz, \
                                                                             iy, ix, \
                                                                             ntheta, FOV[1], FOV[0]))
    if not os.path.exists(read_fpath):
        return None, None, None
        print("DEBUG because projections file not found")
        import pdb; pdb.set_trace()
        
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf['center'][()]))    
    
    return projs, theta, center

if __name__ == "__main__":
    
    print('just a bunch of functions')

    
