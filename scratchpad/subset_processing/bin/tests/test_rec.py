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
from tomo2mesh import Grid
from tomo2mesh.misc import viewer
from tomo2mesh import DataFile
from tomo2mesh.fbp.recon import make_mask, rec_mask, extract_from_mask, recon_patches_3d
import pandas as pd
from tomo2mesh.misc.voxel_processing import TimerGPU
from cupyx.scipy.fft import rfft, irfft, rfftfreq, get_fft_plan
from tomo2mesh.fbp.prep import calc_padding  

N_ITERS = 5
from utils import time_logs as output_path
n = 2048
ntheta = 1500
nc = 32
n_iter = 5
tag = 'p3d_fbp'

# def pinned_array(array):
#     # first constructing pinned memory
#     mem = cp.cuda.alloc_pinned_memory(array.nbytes)
#     src = np.frombuffer(
#                 mem, array.dtype, array.size).reshape(array.shape)
#     src[...] = array
#     return src


def run(data_cpu, theta, center, r_fac, n_iter, nc):
    # arguments to recon_chunk2: data, theta, center, p3d
    
    timer_tot = TimerGPU("ms")
    n_sel = int(nc*n*n*r_fac/(nc**3))    
    p_one = Grid((nc,n,n), width = nc, n_points = n_sel)
    p_one.vol_shape = (nc*n_iter, n, n)
    
    for ii in range(n_iter):

        if ii == 0:
            p_sel = p_one.copy()
        else:
            p_one.points[:,0] += nc
            p_sel.append(p_one.copy())

    
    timer_tot.tic()
    sub_vols, p_sel, times = recon_patches_3d(data_cpu, theta, center, p_sel, TIMEIT = True)
    timer_tot.toc()
    
    columns = ["ntheta", "nz", "n", "t_cpu2gpu", "t_filt", "t_mask", "t_bp", "t_gpu2cpu"]
    df = pd.DataFrame(data = np.median(times, axis=0).reshape(1,-1), columns = columns)
    df["r_fac"] = r_fac
    # print(df)
    print(pd.DataFrame(data = times, columns = columns))
    return df

if __name__ == "__main__":

    if 0:
        b = np.random.normal(0, 1, (ntheta, nc, n)).astype(np.float32)
        # b = pinned_array(b)
        timer_transfer = TimerGPU("ms")
        stream1 = cp.cuda.Stream(non_blocking=True)
        for ii in range(100):
            timer_transfer.tic()
            # with stream1:
            #     a = cp.empty(b.shape, dtype = cp.float32)
            #     a.set(b, stream = stream1)
            with stream1:
                a = cp.array(b, dtype = cp.float32)
            timer_transfer.toc("transfer time")

    # experiment 1
    dfs = []
    sparsity = np.logspace(0,2,10)
    r_fac_list = np.sort(1.0/sparsity)

    data_cpu = np.random.normal(0,1,(ntheta, nc*n_iter, n)).astype(np.float32)
    # data_cpu = pinned_array(data_cpu)
    theta = np.linspace(0, np.pi, ntheta, dtype = np.float32)
    center = n/2.0

    for r_fac in r_fac_list:
        print('\n'+'#'*30)
        print(f'experiment: n={n}, nz={nc}, ntheta={ntheta}, 1/r={(1.0/r_fac):.2f}\n')
        dfs.append(run(data_cpu, theta, center, r_fac, n_iter, nc))
        
    pd.concat(dfs, ignore_index = True).to_csv(os.path.join(output_path, f'{tag}_times_n{n}_ntheta{ntheta}_nc{nc}.csv'), index=False)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
