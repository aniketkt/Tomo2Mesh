#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from tomo2mesh.projects.eaton.recon import recon_binned
import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage
from scipy import ndimage as ndimage_cpu
from tomo2mesh.fbp.recon import recon_all

def void_map_gpu(projs, theta, center, dark, flat, b, pixel_size):
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    # fbp
    t_gpu.tic()
    V = recon_binned(projs, theta, center, dark, flat, b, pixel_size)
    V[:] = ndimage.gaussian_filter(V,0.5)
    # print(V.shape)
    # print(V[::2,::2,::2].shape)
    
    # binarize
    voxel_values = get_values_cyl_mask(V[::2,::2,::2], 1.0).get()
    rec_min_max = modified_autocontrast(voxel_values, s=0.01)
    thresh = cp.float32(threshold_otsu(voxel_values))    
    V[:] = (V<thresh).astype(cp.uint8)
    cylindrical_mask(V, 1, mask_val = 1)
    t_rec = t_gpu.toc('RECONSTRUCTION')

    # connected components labeling
    t_gpu.tic()
    V = cp.array(V, dtype = cp.uint32)
    V[:], n_det = ndimage.label(V,structure = cp.ones((3,3,3),dtype=cp.uint8))    
    
    voids_b = Voids().count_voids(V.get(), b, 2)    
    t_label = t_gpu.toc('LABELING')
    
    voids_b["rec_min_max"] = rec_min_max
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}

    del V
    memory_pool.free_all_blocks()    
    return voids_b


def void_map_all(projs, theta, center, dark, flat, b, pixel_size):
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    ###### Work ###########
    ntheta, nz, n = projs.shape
    projs = np.mean(projs.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4))

    projs = np.array(projs, dtype = np.float32)
    dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
    flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
    dark = np.array(dark.astype(np.float32), dtype = np.float32)
    flat = np.array(flat.astype(np.float32), dtype = np.float32)
    theta = np.array(theta, dtype = np.float32)
    center = np.float32(center/float(b))

    #######################

    # fbp
    t_gpu.tic()
    V = recon_all(projs, theta, center, 32, (dark,flat))
    # print(V.shape)
    # print(V[::4,::4,::4].shape)
    # V[:] = ndimage.gaussian_filter(V,0.5)
    
    # binarize
    voxel_values = get_values_cyl_mask(V[::4,::4,::4], 1.0)
    rec_min_max = modified_autocontrast(voxel_values, s=0.01)
    thresh = np.float32(threshold_otsu(voxel_values))    
    V = (V<thresh).astype(np.uint8)
    cylindrical_mask(V, 1, mask_val = 1)
    t_rec = t_gpu.toc('RECONSTRUCTION')

    print(V.shape)
    # pretty sure you are going to get oom here for b = 1, but b =2 will work
    # connected components labeling
    t_gpu.tic()
    V = cp.array(V, dtype = cp.uint32)
    V[:], n_det = ndimage.label(V,structure = cp.ones((3,3,3),dtype=cp.uint8))  
    voids_b = Voids().count_voids(V.get(), b, 2)    

    # V = np.array(V, dtype = np.uint32)
    # V[:], n_det = ndimage.label(V,structure = np.ones((3,3,3),dtype=np.uint8))   
    # voids_b = Voids().count_voids(V, b, 2)    

    t_label = t_gpu.toc('LABELING')
    
    voids_b["rec_min_max"] = rec_min_max
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}

    del V
    memory_pool.free_all_blocks()    
    return voids_b


