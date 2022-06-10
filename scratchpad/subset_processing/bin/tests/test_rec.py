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

import cupy as cp
from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
from tomo_encoders.reconstruction.recon import make_mask, rec_mask, extract_from_mask
from tomo_encoders.reconstruction.prep import fbp_filter
import pandas as pd
N_ITERS = 5
output_path = '/data02/MyArchive/aisteer_3Dencoders/voids_paper_data'

def run_func(data_cpu, theta, center, cpts_full, nc = 32):
    
    ntheta, nc, n = data_cpu.shape
    data = cp.empty((ntheta, nc, n), dtype = cp.float32)
    theta = cp.array(theta, dtype = cp.float32)
    center = cp.float32(center)
    obj_mask = cp.empty((nc, n, n), dtype = cp.float32)
    sub_vols = []
    times = []
    for ic in range(N_ITERS):
        # print(f'ITERATION {ic}')
        
#         p_sel must adjust coordinates based on the chunk size
        cpts = cpts_full
        
        # COPY DATA TO GPU
        start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
        stream = cp.cuda.Stream()
        with stream:
            data.set(data_cpu)
        end_gpu.record(); end_gpu.synchronize(); t_cpu2gpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
        # print(f"overhead for copying data to gpu: {t_cpu2gpu:.2f} ms")            
            
        # FBP FILTER
        t_filt = fbp_filter(data)
        
        # BACK-PROJECTION
        t_mask = make_mask(obj_mask, cpts,32)
        t_bp = rec_mask(obj_mask, data, theta, center)
        
        # EXTRACT PATCHES AND SEND TO CPU
        sub_vols_unit, t_gpu2cpu = extract_from_mask(obj_mask, cpts, 32)
        times.append([r_fac, ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_bp, t_gpu2cpu])
        sub_vols.append(sub_vols_unit)

    del obj_mask, data, theta, center    
    cp._default_memory_pool.free_all_blocks()    
    return np.asarray(sub_vols).reshape(-1,32,32,32), np.asarray(times)
        
def run(ntheta, nc, n, r_fac):
    n_sel = int(nc*n*n*r_fac/(32**3))
    # arguments to recon_chunk2: data, theta, center, p3d
    data_cpu = np.random.normal(0,1,(ntheta, nc, n)).astype(np.float32)
    theta = np.linspace(0, np.pi, ntheta, dtype = np.float32)
    center = n/2.0
    p_sel = Patches((nc,n,n), initialize_by = 'regular-grid', patch_size = (32,32,32), n_points = n_sel)
    print(f'r = N(P)/N(V): {len(p_sel)*32**3/(nc*n*n):.2f}')    
    sub_vols, times = run_func(data_cpu, theta, center, p_sel.points, nc = nc)
    print(f'returned sub_vols shape: {sub_vols.shape}')
    columns = ["r_fac", "ntheta", "nz", "n", "t_cpu2gpu", "t_filt", "t_mask", "t_backproj", "t_gpu2cpu"]
    df = pd.DataFrame(data = np.median(times, axis=0).reshape(1,-1), columns = columns)
    return df


if __name__ == "__main__":

    # experiment 1
    n = 2048
    ntheta = 1500
    dfs = []
    nc = 64
    sparsity = np.logspace(0,2,10)
    r_fac_list = np.sort(1.0/sparsity)
    # r_fac_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    
    for r_fac in r_fac_list:
        print(f'experiment: n={n}, nz={nc}, ntheta={ntheta}, 1/r={(1.0/r_fac):.2f}')
        dfs.append(run(ntheta, nc, n, r_fac))
    pd.concat(dfs, ignore_index = True).to_csv(os.path.join(output_path, 'output_exp1.csv'), index=False)
    
    # # experiment 2
    # r_fac = 0.2
    # nc = 32
    # items = [(750, 1024), (1500, 2048), (3000, 4096)]
    # dfs = []
    # for iter_item in items:
    #     ntheta, n = iter_item
    #     print(f'experiment: n={n}, nz={nc}, ntheta={ntheta}, r={r_fac}')
    #     dfs.append(run(ntheta, nc, n, r_fac))
    # pd.concat(dfs, ignore_index = True).to_csv(os.path.join(output_path, 'output_exp2.csv'), index=False)
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
