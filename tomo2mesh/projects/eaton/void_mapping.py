#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask, TimerCPU
from tomo2mesh.projects.eaton.recon import recon_binned, recon_all
from tomo2mesh.structures.patches import Patches
import cupy as cp
import numpy as np
import pandas as pd
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage
from scipy import ndimage as ndimage_cpu
import cc3d
import matplotlib.pyplot as plt

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

def local_otsu(V, p_size):

    patches = Patches(V.shape, initialize_by = "bestfit_grid", patch_size = (p_size,p_size,p_size))
    x_vols = patches.extract(V, (p_size,p_size,p_size))

    thresh_list = []
    for i in range(len(x_vols)):
        thresh_list.append((x_vols[i]<threshold_otsu(x_vols[i][::3,::3,::3])).astype(np.uint8))
    
    V_seg = np.empty(V.shape, dtype = np.uint8)
    patches.fill_patches_in_volume(thresh_list, V_seg)
    return V_seg


def void_map_all(projs, theta, center, dark, flat, b, pixel_size, dust_thresh, z_crop = (None,None), local_otsu_psize = 144):
    timer = TimerCPU("secs")
    
    # tmp_path = '/data01/Eaton_Polymer_AM/reconstructed/tmp_rec'
    #Reconstruction
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    # down-sampling
    if b > 1:
        ntheta, nz, n = projs.shape
        projs = np.mean(projs.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4))
        dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
        flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
        theta = np.array(theta, dtype = np.float32)
        center = np.float32(center/float(b))
    
    projs = np.array(projs, dtype = np.float32)
    dark = np.array(dark.astype(np.float32), dtype = np.float32)
    flat = np.array(flat.astype(np.float32), dtype = np.float32)

    #FBP
    t_gpu.tic()
    V_rec = recon_all(projs, theta, center, 32, dark, flat, pixel_size*b, outlier_removal = True if b == 1 else False) 
    t_rec = t_gpu.toc('RECONSTRUCTION')

    timer = TimerCPU("secs")
    timer.tic()
    V_seg = local_otsu(V_rec, p_size = local_otsu_psize)
    timer.toc(f'BINARIZATION LOCAL OTSU PSIZE {local_otsu_psize}')

    V_seg = V_seg[slice(z_crop[0]//b, z_crop[1]//b),...]
    cylindrical_mask(V_seg, 1, mask_val = 0)
    timer.tic()
    # Connected components
    V_seg = cc3d.connected_components(V_seg)
    # print("Porosity before dust removal:", (np.sum((V_seg>0).astype(np.uint8)))/(np.prod(V_seg.shape)*np.pi/4))
    voids_b = Voids().count_voids(V_seg, b, dust_thresh)
    
    # Calculate porosity if b = 1
    if b==1:
        # Porosity calculation with dust removal
        V_bin = np.zeros(voids_b.vol_shape, dtype = np.uint8)
        for ii, s_void in enumerate(voids_b["s_voids"]):
            # V_bin[s_void] = 1
            V_bin[s_void] +=voids_b["x_voids"][ii]
        voids_b["porosity"] = np.sum(V_bin)/(np.prod(voids_b.vol_shape)*np.pi/4)
        voids_b["porosity_z"] = np.sum(V_bin, axis=(1,2))/(np.prod(voids_b.vol_shape[1:])*np.pi/4)
        print("Porosity after dust removal: ", voids_b["porosity"])

    if b!=1:
        voids_b["porosity"] = 0
        voids_b["porosity_z"] = 0

    # blow up volume enclosing voids back to full size before cropping
    voids_b.vol_shape = V_rec.shape
    voids_b.transform_linear_shift([z_crop[0]//b,0,0])
    t_label = timer.toc('LABELING')
    
    # assign some attributes to voids collection
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}

    del V_rec
    del V_seg
    memory_pool.free_all_blocks()    
    return voids_b


