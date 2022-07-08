#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from tomo2mesh.projects.eaton.recon import recon_binned, recon_all
from tomo2mesh.structures.patches import Patches
import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage
from scipy import ndimage as ndimage_cpu
import cc3d

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


def void_map_all(projs, theta, center, dark, flat, b, pixel_size, start, stop):
    ###### Work ###########
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    ntheta, nz, n = projs.shape
    projs = np.mean(projs.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4))
    projs = np.array(projs, dtype = np.float32)
    dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
    flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
    dark = np.array(dark.astype(np.float32), dtype = np.float32)
    flat = np.array(flat.astype(np.float32), dtype = np.float32)
    theta = np.array(theta, dtype = np.float32)
    center = np.float32(center/float(b))

    # fbp
    t_gpu.tic()
    V = recon_all(projs, theta, center, 32, dark, flat, pixel_size) 
    V_rec = V #[:(nz//28)*28,:(n//28)*28,:(n//28)*28] 

    t_rec = t_gpu.toc('RECONSTRUCTION')

    #######################
    
    # binarize
    #voxel_values = get_values_cyl_mask(V, 1.0)
    #rec_min_max = modified_autocontrast(voxel_values, s=0.01)
    
    p_size = 144
    patches = Patches(V_rec.shape, initialize_by = "regular-grid", patch_size = (p_size,p_size,p_size))
    x_vols = patches.extract(V_rec, (p_size,p_size,p_size))

    thresh_list = []
    for i in range(len(x_vols)):
        thresh_list.append((x_vols[i]<threshold_otsu(x_vols[i][::2,::2,::2])).astype(np.uint8))

    V_seg = np.empty(V_rec.shape, dtype = np.uint8)
    patches.fill_patches_in_volume(thresh_list, V_seg)
    #V_seg = np.median(V_empty, axis = 0)

    start += 1
    stop += 1
    if start > 0:
        V_seg[:start] = 1
    if stop < 1152:
        V_seg[stop:] = 1
    cylindrical_mask(V_seg, 1, mask_val = 1)

    t_gpu.tic()
    # V = cp.array(V, dtype = cp.uint32)
    # V[:], n_det = ndimage.label(V,structure = cp.ones((3,3,3),dtype=cp.uint8))
    # print(n_det)
    # voids_b = Voids().count_voids(V.get(), b, 2)
    
    V_seg = cc3d.connected_components(V_seg)
    # V_seg = np.array(V_seg, dtype = np.uint32)
    # V_seg[:], n_det = ndimage_cpu.label(V_seg,structure = np.ones((3,3,3),dtype=np.uint8))
    # print(n_det)
    voids_b = Voids().count_voids(V_seg, b, 2)

    t_label = t_gpu.toc('LABELING')
    
    #voids_b["rec_min_max"] = rec_min_max
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}

    del V
    del V_rec
    del V_seg
    memory_pool.free_all_blocks()    
    return voids_b


