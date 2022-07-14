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


def void_map_all(projs, theta, center, dark, flat, b, pixel_size, z_crop = (None,None)):
    timer = TimerCPU("secs")
    
    # tmp_path = '/data01/Eaton_Polymer_AM/reconstructed/tmp_rec'
    #Reconstruction
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

    #FBP
    t_gpu.tic()
    V = recon_all(projs, theta, center, 32, dark, flat, pixel_size) 
    V_rec = V #[:(nz//28)*28,:(n//28)*28,:(n//28)*28] 

    t_rec = t_gpu.toc('RECONSTRUCTION')

    timer = TimerCPU("secs")
    timer.tic()
    p_size = 144
    patches = Patches(V_rec.shape, initialize_by = "bestfit_grid", patch_size = (p_size,p_size,p_size))
    x_vols = patches.extract(V_rec, (p_size,p_size,p_size))
    timer.toc("Create patches")

    timer.tic()
    thresh_list = []
    for i in range(len(x_vols)):
        thresh_list.append((x_vols[i]<threshold_otsu(x_vols[i][::3,::3,::3])).astype(np.uint8))
    timer.toc("Create thresh list")

    timer = TimerCPU("secs")
    timer.tic()
    V_seg = np.empty(V_rec.shape, dtype = np.uint8)
    patches.fill_patches_in_volume(thresh_list, V_seg)
    V_seg = V_seg[slice(z_crop[0]//b, z_crop[1]//b),...]
    timer.toc("Fill patches")

    # Connected components
    cylindrical_mask(V_seg, 1, mask_val = 0)
    V_seg = cc3d.connected_components(V_seg)

    # Porosity without dust removal
    porosity = (np.sum((V_seg>0).astype(np.uint8)))/(np.prod(V_seg.shape)*np.pi/4)
    print("Porosity before dust removal:", porosity)

    timer.tic()
    voids_b = Voids().count_voids(V_seg, b, 2)
    
    # Porosity calculation with dust removal
    # try 1
    porosity = np.sum(voids_b["sizes"])/(np.prod(voids_b.vol_shape)*np.pi/4)
    print("Porosity after dust removal: ", porosity)
    voids_b["porosity"] = porosity

    # # try 2
    # counter = 0
    # for void in voids_b["x_voids"]:
    #     counter += np.sum(void)
    # porosity = counter/(np.prod(voids_b.vol_shape)*np.pi/4)
    # print("Porosity try2: ", porosity)

    # # try 3
    # V_seg = np.zeros(voids_b.vol_shape, dtype=np.uint8)
    # for iv, s_void in enumerate(voids_b["s_voids"]):
    #     V_seg[s_void] = voids_b["x_voids"][iv].copy()
    # print(V_seg.shape)
    # porosity = (np.sum(V_seg))/(np.prod(V_seg.shape)*np.pi/4)
    # print("Porosity try3: ", porosity)

    # from tomo2mesh import viewer
    # viewer.view_midplanes(vol = V_seg)
    # plt.savefig('/data01/tmp.png')
    # plt.close()

    
    # blow up volume enclosing voids back to full size before cropping
    voids_b.vol_shape = V_rec.shape
    voids_b.transform_linear_shift([z_crop[0]//b,0,0])
    t_label = timer.toc('LABELING')
    
    
    # assign some attributes to voids collection
    voids_b.compute_time = {"reconstruction" : t_rec, "labeling" : t_label}
    

    # V = cp.array(V, dtype = cp.uint32)
    # V[:], n_det = ndimage.label(V,structure = cp.ones((3,3,3),dtype=cp.uint8))
    # print(n_det)
    # voids_b = Voids().count_voids(V.get(), b, 2)
    
    # V_seg = np.array(V_seg, dtype = np.uint32)
    # V_seg[:], n_det = ndimage_cpu.label(V_seg,structure = np.ones((3,3,3),dtype=np.uint8))
    # print(n_det)
    
    #voids_b["rec_min_max"] = rec_min_max

    del V
    del V_rec
    del V_seg
    memory_pool.free_all_blocks()    
    return voids_b


