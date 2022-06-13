#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask
from tomo2mesh.fbp.recon import recon_patches_3d
import cupy as cp
import numpy as np
from tomo2mesh import Grid
from tomo2mesh.fbp.recon import recon_all_gpu, recon_all
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage



def coarse_map(projs, theta, center, b, b_K, dust_thresh):
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    # fbp
    t_gpu.tic()
    raw_data = projs[::b_K,::b,::b], theta[::b_K,...], center/b
    _, nz, n = raw_data[0].shape    
    V = cp.empty((nz,n,n), dtype = cp.float32)
    recon_all_gpu(*raw_data, V)
    V[:] = ndimage.gaussian_filter(V,0.5)
    
    # binarize
    voxel_values = get_values_cyl_mask(V[::2,::2,::2], 1.0).get()
    rec_min_max = modified_autocontrast(voxel_values, s=0.01)
    thresh = cp.float32(threshold_otsu(voxel_values))    
    V[:] = (V<thresh).astype(cp.uint8)
    t_rec = t_gpu.toc('COARSE RECON')

    # connected components labeling
    t_gpu.tic()
    V = cp.array(V, dtype = cp.uint32)
    V[:], n_det = ndimage.label(V,structure = cp.ones((3,3,3),dtype=cp.uint8))    
    
    voids_b = Voids().count_voids(V.get(), b, dust_thresh)    
    t_label = t_gpu.toc('LABELING')
    
    voids_b["rec_min_max"] = rec_min_max
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}

    del V
    cp.fft.config.clear_plan_cache(); memory_pool.free_all_blocks()    
    return voids_b

def coarse_map2(projs, theta, center, b, b_K, wd):
    t_gpu = TimerGPU("secs")
    t_gpu.tic()
    V = recon_all(projs[::b_K,::b,::b], theta[::b_K,...], center/b, wd)
    t_rec = t_gpu.toc('COARSE RECON')
    t_gpu.tic()
    voxel_values = get_values_cyl_mask(V[::4,::4,::4], 1.0)
    thresh = np.float32(threshold_otsu(voxel_values))
    V = (V < thresh).astype(np.uint8)
    voids_b = Voids().count_voids(V, b)
    t_label = t_gpu.toc('LABELING')
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}
    return voids_b

def coarse_map_surface(V_bin, b, wd):
    
    # find patches on surface
    V_edge = edge_map(V_bin)
    wdb = int(wd//b)
    p3d = Grid(V_bin.shape, width = wdb)
    
    is_surf = (np.sum(p3d.extract(V_edge), axis = (1,2,3)) > 0.0).astype(np.uint8)
    is_zeros = (np.sum(p3d.extract(V_bin), axis = (1,2,3)) == 0.0).astype(np.uint8)
    
    p3d = p3d.rescale(b)
    p3d_surf = p3d.filter_by_condition(is_surf)
    p3d_zeros = p3d.filter_by_condition(is_zeros)
    
    eff = len(p3d_surf)*(wd**3)/np.prod(p3d_surf.vol_shape)
    print(f"\tSTAT: r value: {eff*100.0:.2f}")        
    return p3d_surf, p3d_zeros

def process_subset(projs, theta, center, fe, p_surf, min_max, seg_batch = False):

    if seg_batch:
        # SCHEME 1: integrate reconstruction and segmention (segments data on gpu in batches as they are reconstructed)
        x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, segmenter = fe, \
                                          segmenter_batch_size = 64, rec_min_max = min_max)
    else:
        # SCHEME 2: reconstruct and segment separately (copies rec data from gpu to cpu)
        x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf)
        print(f"x_surf min {x_surf.min()}; max {x_surf.max()}")
        print(f"clip to range: {min_max}")
        x_surf = fe.predict_patches("segmenter", x_surf[...,np.newaxis], 64, None, min_max = min_max)[...,0]
    print(f'\tSTAT: total patches in neighborhood: {len(p_surf)}')    
    return x_surf, p_surf
    
    
    

