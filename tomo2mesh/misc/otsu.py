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

def otsu_cyl_mask(V_rec, mask_frac, apply_mask_frac = None, inverted = True):

    xp = cp.get_array_module(V_rec)
    voxel_values = get_values_cyl_mask(V_rec[::2,::2,::2], mask_frac)
    if xp == cp:
        voxel_values = voxel_values.get()
        
    thresh = xp.float32(threshold_otsu(voxel_values))    
    V_seg = (V_rec<thresh).astype(xp.uint8)
    
    if apply_mask_frac is not None:
        cylindrical_mask(V_seg,apply_mask_frac,1)
    else:
        cylindrical_mask(V_seg,mask_frac,1)

    if not inverted:
        V_seg = V_seg^1
    else:        
        return V_seg




