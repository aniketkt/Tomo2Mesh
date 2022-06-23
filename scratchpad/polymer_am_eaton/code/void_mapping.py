#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from recon import recon_binned
import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage


def void_map_gpu(projs, theta, center, dark, flat, b, pixel_size):
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    # fbp
    t_gpu.tic()
    V = recon_binned(projs, theta, center, dark, flat, b, pixel_size)
    V[:] = ndimage.gaussian_filter(V,0.5)
    
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

