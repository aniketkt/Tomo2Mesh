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

def local_otsu(V, p_size, fac = 1):

    patches = Patches(V.shape, initialize_by = "bestfit_grid", patch_size = (p_size,p_size,p_size))
    x_vols = patches.extract(V, (p_size,p_size,p_size))

    thresh_list = []
    for i in range(len(x_vols)):
        thresh_list.append((x_vols[i]<threshold_otsu(x_vols[i][::fac,::fac,::fac])).astype(np.uint8))
    
    V_seg = np.empty(V.shape, dtype = np.uint8)
    patches.fill_patches_in_volume(thresh_list, V_seg)
    return V_seg

def get_annular_mask(n, ring_wd, xp):
    pts = xp.arange(-int(n//2), int(xp.ceil(n//2)))
    yy, xx = xp.meshgrid(pts, pts, indexing = 'ij')
    
    rad_max = n//2
    assert rad_max%ring_wd == 0, "incompatible arguments"
    circ = xp.zeros((n,n), dtype = xp.uint8)
    radii = xp.arange(rad_max,0,-ring_wd)
    for rad in radii:
        circ += (xp.sqrt(yy**2 + xx**2) < rad).astype(xp.uint8)   
#     cyl = xp.repeat(circ[xp.newaxis, ...], nc, axis = 0)
    return circ


def otsu_annular_regions(V_rec, ring_wd, nc, fac_nc):
    xp = cp.get_array_module(V_rec)
    nz, _, n = V_rec.shape
    n_rings = int(n/(2*ring_wd))
    V_seg = xp.zeros_like(V_rec, dtype = np.uint8)    
    
    circ = get_annular_mask(n, ring_wd, np)
    
    from tqdm import tqdm
    pbar = tqdm(total = int(np.ceil(nz/nc))*n_rings)
    for ic in range(int(np.ceil(nz/nc))):
        svert = slice(ic*nc, (ic+1)*nc)
        for ival in range(1,n_rings + 1):
            
            voxel_values = V_rec[ic*nc:(ic+1)*nc:fac_nc,circ == ival]
            if xp == cp: # no gpu implementation available for threshold_otsu
                voxel_values = voxel_values.get()
                
            thresh = threshold_otsu(voxel_values)
            if xp == cp: thresh = cp.float32(thresh)
                
            V_seg[svert, circ == ival] = V_rec[svert, circ == ival] < thresh
            pbar.update(1)
    pbar.close()
    
    return V_seg

def fbp(projs, theta, center, dark, flat, b, pixel_size, pg_pad = 8):

    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    
    if b == 1:
        projs = np.array(projs, dtype = np.float32)
        dark = np.array(dark.astype(np.float32), dtype = np.float32)
        flat = np.array(flat.astype(np.float32), dtype = np.float32)
        V_rec = recon_all(projs, theta, center, 32, dark, flat, pixel_size, outlier_removal = True if b == 1 else False, pg_pad = pg_pad) 

    elif b == 2:
        ntheta, nz, n = projs.shape
        projs = np.mean(projs.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4))
        dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
        flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
        theta = np.array(theta, dtype = np.float32)
        center = np.float32(center/float(b))
    
        projs = np.array(projs, dtype = np.float32)
        dark = np.array(dark.astype(np.float32), dtype = np.float32)
        flat = np.array(flat.astype(np.float32), dtype = np.float32)
        V_rec = recon_all(projs, theta, center, 32//b, dark, flat, pixel_size*b, outlier_removal = True if b == 1 else False, pg_pad = pg_pad//b) 
    
    elif b == 4:
        V_rec = recon_binned(projs, theta, center, dark, flat, b, pixel_size).get()
        
        

    V_rec = np.clip(V_rec, *modified_autocontrast(V_rec[::4,::4,::4], s = 0.005))
    cylindrical_mask(V_rec, 1, mask_val = V_rec[::4,::4,::4].min())
    
    memory_pool.free_all_blocks()    
    return V_rec


def void_map_all(projs, theta, center, dark, flat, b, pixel_size, dust_thresh, z_crop = (None,None)):
    timer = TimerCPU("secs")
    t_gpu = TimerGPU("secs")
    
    # ring_wd = 51 # 306//b
    # nc = 96#128//b
    # fac_nc = 1#4
    ring_wd = 306//b
    nc = 128//b
    fac_nc = 4
    mask_ratio = 0.9

    
    # tmp_path = '/data01/Eaton_Polymer_AM/reconstructed/tmp_rec'
    #Reconstruction
    t_gpu.tic()
    V_rec = fbp(projs, theta, center, dark, flat, b, pixel_size)
    t_rec = t_gpu.toc('RECONSTRUCTION')

    timer = TimerCPU("secs")
    timer.tic()
    V_seg = otsu_annular_regions(V_rec, ring_wd, nc, fac_nc)

    cylindrical_mask(V_seg, mask_ratio, mask_val = 0) # no need for mask when using annular otsu
    timer.toc(f'BINARIZATION ANNULAR OTSU')

    timer.tic()
    V_seg = V_seg[slice(z_crop[0]//b, z_crop[1]//b),...]
    # Connected components
    V_seg = cc3d.connected_components(V_seg)
    # print("Porosity before dust removal:", (np.sum((V_seg>0).astype(np.uint8)))/(np.prod(V_seg.shape)*np.pi/4))
    voids_b = Voids().count_voids(V_seg, b, dust_thresh)
    
    # Calculate porosity if b = 1
    if 1: # b==1
        # Porosity calculation with dust removal
        V_bin = np.zeros(voids_b.vol_shape, dtype = np.uint8)
        for ii, s_void in enumerate(voids_b["s_voids"]):
            # V_bin[s_void] = 1
            V_bin[s_void] +=voids_b["x_voids"][ii]
        voids_b["porosity"] = np.sum(V_bin)/(np.prod(voids_b.vol_shape)*np.pi/4*mask_ratio**2)
        voids_b["porosity_z"] = np.sum(V_bin, axis=(1,2))/(np.prod(voids_b.vol_shape[1:])*np.pi/4*mask_ratio**2)
        print("Porosity after dust removal: ", voids_b["porosity"])

    else:
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
    
    return voids_b


