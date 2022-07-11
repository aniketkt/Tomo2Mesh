#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import pandas as pd
import cupy as cp
from tomo2mesh import Grid
from tomo2mesh.fbp.subset import recon_patches_3d
import pandas as pd
from tomo2mesh.misc.voxel_processing import TimerGPU


N_ITERS = 5
from tomo2mesh.projects.steel_am.rw_utils import time_logs as output_path
nc = 32
n_iter = 5
tag = 'subsetrec'


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
    
    columns = ["t_cpu2gpu", "t_filt", "t_mask", "t_bp", "t_gpu2cpu"]
    
    df = pd.DataFrame(data = np.median(times, axis=0)[3:].reshape(1,-1), columns = columns)
    df["r_fac"] = r_fac
    # print(df)
    print(pd.DataFrame(data = times[:,3:], columns = columns))
    return df

if __name__ == "__main__":

    n = 2048
    ntheta = 1500


    # experiment 1
    dfs = []
    sparsity = np.logspace(0,2,10)
    r_fac_list = np.sort(1.0/sparsity)

    projs = np.random.normal(0,1,(nc*n_iter, ntheta, n)).astype(np.float32)
    theta = np.linspace(0, np.pi, ntheta, dtype = np.float32)
    center = n/2.0

    for r_fac in r_fac_list:
        print('\n'+'#'*30)
        print(f'experiment: n={n}, nz={nc}, ntheta={ntheta}, 1/r={(1.0/r_fac):.2f}\n')
        dfs.append(run(projs, theta, center, r_fac, n_iter, nc))
        
    print(pd.concat(dfs, ignore_index = True))
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
