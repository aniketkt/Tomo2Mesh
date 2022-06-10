#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import time
import seaborn as sns
import pandas as pd

from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning
from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d

ct_path = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/data/AM316_L205_fs_tomo_L5_rec_1x1_uint16_tiff'
seg_path = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/seg_data/AM316_L205_fs_tomo_L5/AM316_L205_fs_tomo_L5_GT'
rec_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_x_rec'
x_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_x'
gt_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_y'

if __name__ == "__main__":

    ds_ct = DataFile(ct_path, tiff = True)
    Vx = ds_ct.read_full()
    Vx = normalize_volume_gpu(Vx.astype(np.float32), chunk_size=1, normalize_sampling_factor=4, TIMEIT = True)
    Vx = Vx[50:-50-32,:-80,:-80].copy()    

    # Define patches for a cylindrical grid
    VOL_SHAPE = Vx.shape#(projs.shape[1], projs.shape[2], projs.shape[2])
    PATCH_SIZE = (32,32,32)
    p_grid = Patches(VOL_SHAPE, initialize_by='regular-grid', patch_size = PATCH_SIZE)
    p_grid = p_grid.filter_by_cylindrical_mask(mask_ratio=1)

    # Save Vx
    x = p_grid.extract(Vx, PATCH_SIZE)
    Vx = np.zeros(Vx.shape, dtype = np.float32)
    p_grid.fill_patches_in_volume(x, Vx)
    ds_x = DataFile(x_path, tiff=True, d_shape = Vx.shape, d_type = np.float32)
    ds_x.create_new(overwrite=True)
    ds_x.write_full(Vx.astype(np.float32))

    
    # Save GT: Initialize volume with ones, then fill
    ds_seg = DataFile(seg_path, tiff = True)
    Vy = ds_seg.read_full()
    Vy = Vy[50:-50-32,:-80,:-80].copy()    
    y = p_grid.extract(Vy, PATCH_SIZE)
    Vy = np.ones(Vy.shape, dtype = np.uint8)
    p_grid.fill_patches_in_volume(y, Vy)
    ds_gt = DataFile(gt_path, tiff=True, d_shape = Vy.shape, d_type = np.uint8)
    ds_gt.create_new(overwrite=True)
    ds_gt.write_full(Vy)
    
    
    
    
    # PROJECT
    theta = np.linspace(0,np.pi,3000,dtype='float32')
    pnz = 2
    center = Vx.shape[-1]//2.0
    projs, theta, center = get_projections(Vx, theta, center, pnz)

    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_L205L5', 'w')
    hf.create_dataset("data",data = projs)
    hf.create_dataset("theta", data = theta)
    hf.create_dataset("center", data = center)
    hf.close()    
    
    # RECONSTRUCT
    x_rec, p_grid = recon_patches_3d(projs, theta, center, p_grid, TIMEIT = True)
    print(f'total patches reconstructed: {x_rec.shape}')    
    # Rec: Initialize volume with minimum value from the x_rec array, then fill
    Vx_rec = np.ones(VOL_SHAPE)*x_rec[:,::4,::4,::4].min()
    p_grid.fill_patches_in_volume(x_rec, Vx_rec)    
    Vx_rec = normalize_volume_gpu(Vx_rec.astype(np.float32), chunk_size=1, normalize_sampling_factor=4, TIMEIT = True)
    ds_rec = DataFile(rec_path, tiff=True, d_shape = Vx_rec.shape, d_type = np.float32)        
    ds_rec.create_new(overwrite=True)
    ds_rec.write_full(Vx_rec.astype(np.float32))    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
